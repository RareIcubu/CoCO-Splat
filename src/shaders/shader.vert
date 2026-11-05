#version 460 core

layout(location = 0) in vec2 aQuadPos;

// Binding 0: Static buffer for splats (40 bytes = 10 uints per splat)
layout(std430, binding = 0) readonly buffer SplatBuffer {
  uint splatRaw[];
};

// Binding 1: Sorted splat indices
layout(std430, binding = 1) readonly buffer SplatIndexBuffer {
  uint sortedIndecies[];
};

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform vec2 viewport;

out vec3 fColor;
out float fOpacity;
out vec2 uv;

void main()
{
  uint splatIdx = sortedIndecies[gl_InstanceID];
  uint base = splatIdx * 7u;

  // 1. Read position and transform to view space
  vec2 u0 = unpackHalf2x16(splatRaw[base + 0u]); // pos.x, pos.y
  vec2 u1 = unpackHalf2x16(splatRaw[base + 1u]); // pos.z, effectiveRadius

  vec3 aInstancePos = vec3(u0.x, u0.y, u1.x);
  vec4 p_view = view * model * vec4(aInstancePos, 1.0f);

  // 2. Discard splats that are behind the camera
//if (p_view.z > -0.1f) {
//  gl_Position = vec4(0.0f, 0.0f, 2.0f, 1.0f);
//  return;
//}

  // 3. Read color, opacity and covariance matrix
  vec2 u2 = unpackHalf2x16(splatRaw[base + 2u]); // f_dc.r, f_dc.g
  vec2 u3 = unpackHalf2x16(splatRaw[base + 3u]); // f_dc.b, opacity (after sigmoid)

  vec3 aInstanceColor = vec3(u2.x, u2.y, u3.x);
  float opacity = u3.y;

  vec2 cov01 = unpackHalf2x16(splatRaw[base + 4u]);
  vec2 cov23 = unpackHalf2x16(splatRaw[base + 5u]);
  vec2 cov45 = unpackHalf2x16(splatRaw[base + 6u]);

  float c0 = cov01.x; 
  float c1 = cov01.y;
  float c2 = cov23.x;
  float c3 = cov23.y;
  float c4 = cov45.x;
  float c5 = cov45.y;
  
  mat3 Sigma = mat3(
      c0, c1, c2,
      c1, c3, c4,
      c2, c4, c5
  );

  // 4. Rzutowanie kowariancji 3D do przestrzeni ekranu 2D
  vec2 focal = vec2(projection[0][0] * viewport.x * 0.5f, projection[1][1] * viewport.y * 0.5f) * (1.0f/64.0f);
  float tz = -p_view.z;
  float tz2 = tz * tz;

  mat3 J = mat3(
      focal.x / tz, 0.0f, 0.0f,
      0.0f, focal.y / tz, 0.0f,
      (focal.x * p_view.x) / tz2, (focal.y * p_view.y) / tz2, 0.0f
  );

  mat3 W = mat3(view * model);
  mat3 T = J * W;
  mat3 cov2d = T * Sigma * transpose(T);

  // Low-pass filter
  cov2d[0][0] += 0.3f;
  cov2d[1][1] += 0.3f;

  // 5. Values and vectors for eigen decomposition of 2x2 covariance matrix
  float a = cov2d[0][0];
  float b = cov2d[0][1];
  float c = cov2d[1][1];

  float mid = 0.5f * (a + c);
  float radius = length(vec2(0.5f * (a - c), b));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1f);

  vec2 diagVec = (abs(b) > 1e-6f)
    ? ((a >= c) ? normalize(vec2(lambda1 - c, b)) : normalize(vec2(b, lambda1 - a)))
    : ((a >= c) ? vec2(1.0f, 0.0f) : vec2(0.0f, 1.0f));

  float cutoffFactor = 2.0f * log(255.0f * opacity);
 
  float cutoffRadius = sqrt(cutoffFactor);

  // 6. Stretching quad vertices along the eigenvectors of the covariance matrix
  vec2 v1 = min(sqrt(lambda1), 512.0f) * diagVec;
  vec2 v2 = min(sqrt(lambda2), 512.0f) * vec2(diagVec.y, -diagVec.x);

  vec4 p_clip = projection * p_view;
  vec2 ndcCenter = p_clip.xy / p_clip.w;
  vec2 pixelOffset = (aQuadPos.x * v1 + aQuadPos.y * v2) * cutoffRadius;
  vec2 ndcOffset = (pixelOffset / viewport) * 2.0f;

  gl_Position = vec4((ndcCenter + ndcOffset) * p_clip.w, p_clip.z, p_clip.w);

  // 7. Share data with fragment shader
  fColor = max(aInstanceColor, vec3(0.0f));
  uv = aQuadPos * cutoffRadius;
  fOpacity = opacity;
}
