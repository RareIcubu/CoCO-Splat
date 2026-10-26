#include <glad/glad.h>

#include "camera.h"
#include "parser.h"
#include "shader.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <iostream>
#include <math.h>

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
float cameraSpeed = 0.0f;

float lastX = 400, lastY = 300;
bool firstMouse = true;

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (firstMouse) {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom
                                // to top

  lastX = xpos;
  lastY = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
  if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_RELEASE)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
    cameraSpeed = 8.0f;
  else
    cameraSpeed = 1.0f;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::FORWARD, deltaTime * cameraSpeed);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::BACKWARD, deltaTime * cameraSpeed);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::LEFT, deltaTime * cameraSpeed);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::RIGHT, deltaTime * cameraSpeed);
}

float quadVertices[] = {
    // position
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};

int main() {

  GaussianSplatParser parser;

  parser.readPlyFile(getenv("PLY_FILE_PATH"));

  if (parser.splats.empty()) {
    std::cerr << "No splats found in the PLY file." << std::endl;
    return -1;
  }

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window = glfwCreateWindow(1920, 1080, "3DGS-OpenGL", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  // GLFW settings
  glfwMakeContextCurrent(window);
  glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glViewport(0, 0, 800, 600);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);

  GLuint VBO;
  glGenBuffers(1, &VBO);

  GLuint instanceVBO;
  glGenBuffers(1, &instanceVBO);

  GLuint VAO;
  glGenVertexArrays(1, &VAO);

  glBindVertexArray(VAO);

  // unsigned int EBO;
  // glGenBuffers(1, &EBO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  // instance VBO (optimized with FP16 - 28B for splat)
  glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
  glBufferData(GL_ARRAY_BUFFER,
               parser.splats.size() * sizeof(GaussianSplatFP16),
               parser.splats.data(), GL_STATIC_DRAW);

  parser.splats.clear();
  parser.splats.shrink_to_fit();

  // Atribute 1: position (3x GL_HALF_FLOAT)
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_HALF_FLOAT, GL_FALSE,
                        sizeof(GaussianSplatFP16),
                        (void *)offsetof(GaussianSplatFP16, pos));
  glVertexAttribDivisor(1, 1);
  // Atribute 2: Base color SH0 (f_dc[3])
  glEnableVertexAttribArray(2);
  glVertexAttribPointer(2, 3, GL_HALF_FLOAT, GL_FALSE,
                        sizeof(GaussianSplatFP16),
                        (void *)offsetof(GaussianSplatFP16, f_dc));
  glVertexAttribDivisor(2, 1);
  // Attribute 3: Opacity
  glEnableVertexAttribArray(3);
  glVertexAttribPointer(3, 1, GL_HALF_FLOAT, GL_FALSE,
                        sizeof(GaussianSplatFP16),
                        (void *)offsetof(GaussianSplatFP16, opacity));

  glVertexAttribDivisor(3, 1);
  // Attribute 4: Scale
  glEnableVertexAttribArray(4);
  glVertexAttribPointer(4, 3, GL_HALF_FLOAT, GL_FALSE,
                        sizeof(GaussianSplatFP16),
                        (void *)offsetof(GaussianSplatFP16, scale));
  glVertexAttribDivisor(4, 1);

  // Attribute 5: Quaternion rotation
  glEnableVertexAttribArray(5);
  glVertexAttribPointer(5, 4, GL_HALF_FLOAT, GL_FALSE,
                        sizeof(GaussianSplatFP16),
                        (void *)offsetof(GaussianSplatFP16, rot));
  glVertexAttribDivisor(5, 1);

  Shader ourShader("../shaders/shader.vert", "../shaders/shader.frag");

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  while (!glfwWindowShouldClose(window)) {
    processInput(window);

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Rotate 90 degress around the Z axis
    glm::quat myRotation =
        glm::angleAxis(glm::radians(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    // Convert the quaternion to model matrix
    glm::mat4 model =
        glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
    // Create view matrix
    glm::mat4 view = glm::mat4(1.0f);
    // Translate the view matrix to move the scene backwards (so we can see it)
    view = camera.GetViewMatrix();
    // Create projection matrix
    glm::mat4 projection;
    // Create the projection matrix using perspective projection with a 45
    // degreee field of view, an aspect ration of 800/600, and a near and far
    // plane of 0.1 and 100.0
    projection = glm::perspective(glm::radians(camera.Zoom), 800.0f / 600.0f,
                                  0.1f, 1000.0f);

    ourShader.use();

    // glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
    int modelLoc = glGetUniformLocation(ourShader.ID, "model");
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    int viewLoc = glGetUniformLocation(ourShader.ID, "view");
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    int projectionLoc = glGetUniformLocation(ourShader.ID, "projection");
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    glBindVertexArray(VAO);
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, parser.vertexCount);
    glBindVertexArray(0);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}
