#version 460 core
layout (location=0) in vec2 aQuadPos;
layout (location=1) in vec3 aInstancePos;
layout (location=2) in vec3 aInstanceColor;
layout (location=3) in float aInstanceOpacity;
layout (location=4) in vec3 aInstanceScale;
layout (location=5) in vec4 aInstanceRot;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fColor;
out vec2 uv;
out float fOpacity;

const float SH_C0 = 0.28209479177387814f;
const vec2 viewport = vec2(800.0f, 600.0f);

void main()
{
  // 1. Splat center transformation in view space
  vec4 p_view = view * model * vec4(aInstancePos, 1.0f);

  // 2. Discard splats that are behind the camera
  if(p_view.z > -0.1f){
    gl_Position = vec4(0.0f, 0.0f, 2.0f, 1.0f);
    return;
  }

  //Discard giant splats that are too far away
  float maxLogScale = max(aInstanceScale.x, max(aInstanceScale.y, aInstanceScale.z));
  if (maxLogScale > -3.0f){
    gl_Position = vec4(0.0f, 0.0f, 2.0f, 1.0f);
    return;
  }

  // 3. Create scale matrix S 
  vec3 s = exp(aInstanceScale);
  mat3 S = mat3(
      s.x, 0.0f, 0.0f,
      0.0f, s.y, 0.0f,
      0.0f, 0.0f, s.z
      );

  // 4. Create roation matrix R from quaternion
  vec4 q = normalize(aInstanceRot);
  float r = q.x;
  float x = q.y;
  float y = q.z;
  float z = q.w;

  mat3 R = mat3(
      1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + r * z), 2.0f * (x * z - r * y),
      2.0f * (x * y - r * z), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + x * r),
      2.0f * (x * z + y * r), 2.0f * (y * z - x * r), 1.0f - 2.0f * (x * x + y * y)
      );

  // 5. Project the 3D covariance matrix to 2D screen space
  mat3 M = R * S;
  mat3 Sigma = M * transpose(M);

  vec2 focal = vec2(projection[0][0] * viewport.x * 0.5f, projection[1][1] * viewport.y * 0.5f);
  float tz = -p_view.z;
  float tz2 = tz * tz;

  mat3 J = mat3(
      focal.x / tz, 0.0f, 0.0f,
      0.0f, focal.y / tz, 0.0f,
      -(focal.x * p_view.x) / tz2, -(focal.y * p_view.y) / tz2, 0.0f
      );

  mat3 W = mat3(view * model);
  mat3 T = J*W;
  mat3 cov2d = T * Sigma * transpose(T);

  // Low-pass filter 
  cov2d[0][0] += 0.3f;
  cov2d[1][1] += 0.3f;
  // 5. Calculate the eigenvalues and eigenvectors of the 2D covariance matrix
  float a = cov2d[0][0];
  float b = cov2d[0][1];
  float c = cov2d[1][1];

  float mid = 0.5f * (a + c);
  float radius = length(vec2(0.5f * (a - c), b));
  float lambda1 = mid + radius;
  float lambda2 = max(mid - radius, 0.1f);

  vec2 diagVec = normalize(vec2(b, lambda1 - a));
  if (abs(b) < 1e-6f){
    diagVec = (a >= c) ? vec2(1.0f, 0.0f) : vec2(0.0f, 1.0f);
  }

  float opacity = 1.0f / (1.0f + exp(-aInstanceOpacity));

  float cutoffFactor = 2.0f * log(255.0f * opacity);
  if(cutoffFactor < 0.0f){
    gl_Position = vec4(0.0f, 0.0f, 2.0f, 1.0f);
    return;
  }
  float cuttoffRadius = sqrt(cutoffFactor);

  // 7. Stretch the quad vertices along the eigenvectors to form an ellipse
  vec2 v1 = min(sqrt(lambda1), 512.0f) * diagVec;
  vec2 v2 = min(sqrt(lambda2), 512.0f) * vec2(diagVec.y, -diagVec.x);

  vec4 p_clip = projection * p_view;
  vec2 ndcCenter = p_clip.xy / p_clip.w;
  vec2 pixelOffest = (aQuadPos.x * v1 + aQuadPos.y * v2) * 2.0f;
  vec2 ndcOffset = (pixelOffest / viewport) * 2.0f;

  gl_Position = vec4((ndcCenter + ndcOffset) * p_clip.w, p_clip.z, p_clip.w);
  
  // 8. Pass the color and opacity to the fragment shader
  fColor = clamp(vec3(0.5f) + SH_C0 * aInstanceColor, 0.0f, 1.0f);
  uv = aQuadPos * cuttoffRadius;
  fOpacity = opacity;

}
