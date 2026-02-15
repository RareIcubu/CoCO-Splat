#version 460 core
layout (location=0) in vec2 aQuadPos;
layout (location=1) in vec3 aInstancePos;
layout (location=2) in vec3 aInstanceColor;
layout (location=3) in float aInstanceOpacity;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fColor;
out vec2 uv;
out float fOpacity;
void main()
{
  gl_Position = projection * view * model * vec4(aInstancePos+ vec3(aQuadPos * 2.0,0.0), 1.0);
  fColor = 0.5 + 0.5 * aInstanceColor;
  uv = aQuadPos;
  fOpacity = aInstanceOpacity;
}
