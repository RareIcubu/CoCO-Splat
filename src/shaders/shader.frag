#version 460 core
out vec4 FragColor;
in vec3 fColor;
in vec2 uv;
in float fOpacity;

void main()
{
  float dis = dot(uv,uv);
  float alpha = min(0.99f, fOpacity * exp(-0.5f * dis));

  if(alpha < 1.0f/255.0f) discard;

  FragColor = vec4(fColor, alpha);
}
