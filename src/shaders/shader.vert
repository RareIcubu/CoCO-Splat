#version 460 core
layout (location=0) in vec2 aQuadPos;
layout (location=2) in vec3 aInstancePos;
layout (location=3) in vec2 aInstanceColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 fColor;

void main()
{
  gl_Position = projection * view * model * vec4(aInstancePos+ vec3(aQuadPos * 0.5, 0.0), 1.0);
  fColor = 0.5 + 0.5 * aInstanceColor;
}
