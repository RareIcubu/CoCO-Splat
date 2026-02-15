#version 460 core
out vec4 FragColor;
in vec3 fColor;
in vec2 uv;
in float fOpacity;

float sigmoid(float x){
  return 1.0f / (1.0f + exp(-x));
}

void main()
{
  float dis = dot(uv,uv);
  if(dis > 1.0f) discard;

  float alpha = min(0.99, sigmoid(fOpacity) * exp(-0.5 * dis * 4.0));

  if(alpha < 1.0/255.0) discard;

  FragColor = vec4(fColor, alpha);
}
