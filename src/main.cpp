#include <glad/glad.h>

#include "camera.h"
#include "parser.h"
#include "shader.h"
#include <GLFW/glfw3.h>
#include <chrono>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sorter.h>

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
float cameraSpeed = 0.0f;

float lastX = 1920.0f / 2.0f, lastY = 1080.0f / 2.0f;
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

  const char *envPath = std::getenv("PLY_FILE_PATH");
  std::string filename = envPath ? envPath : "../../DELL.ply";

  parser.readPlyFile(filename);

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
  glfwSwapInterval(0); // Disable V-Sync for maximum performance
  glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glViewport(0, 0, 1920, 1080);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);

  unsigned int VBO;
  glGenBuffers(1, &VBO);

  unsigned int VAO;
  glGenVertexArrays(1, &VAO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);

  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  // Static SSBO for splats
  unsigned int splatSSBO;
  glGenBuffers(1, &splatSSBO);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, splatSSBO);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               parser.splats.size() * sizeof(GaussianSplatFP16),
               parser.splats.data(), GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, splatSSBO);

  parser.splats.clear();
  parser.splats.shrink_to_fit();

  // Dynamic SSBO for sorted indices
  unsigned int indexSSBO;
  glGenBuffers(1, &indexSSBO);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, indexSSBO);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<size_t>(parser.vertexCount) * sizeof(unsigned int),
               nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, indexSSBO);

  CudaSorter sorter;
  initCudaSorter(sorter, splatSSBO, indexSSBO, parser.vertexCount);

  // unsigned int EBO;
  // glGenBuffers(1, &EBO);

  // instance VBO (optimized with FP16 - 28B for splat)
  // glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);

  Shader ourShader("../shaders/shader.vert", "../shaders/shader.frag");

  const int modelLoc = glGetUniformLocation(ourShader.ID, "model");
  const int viewLoc = glGetUniformLocation(ourShader.ID, "view");
  const int projectionLoc = glGetUniformLocation(ourShader.ID, "projection");
  const int viewportLoc = glGetUniformLocation(ourShader.ID, "viewport");

  glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));

  ourShader.use();

  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  double statsTime = glfwGetTime();
  int frameCounter = 0;
  double lastSortsMs = 0.0;

  glm::mat4 lastMvp(0.0f);
  int visibleSplats = 0;

  lastFrame = glfwGetTime();

  while (!glfwWindowShouldClose(window)) {
    processInput(window);

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;
    int fbWidth, fbHeight;

    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
    if (fbWidth == 0 || fbHeight == 0) {
      glfwPollEvents();
      continue;
    }

    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Convert the quaternion to model matrix
    // Create view matrix
    glm::mat4 view = glm::mat4(1.0f);
    // Translate the view matrix to move the scene backwards (so we can see it)
    view = camera.GetViewMatrix();

    // Create projection matrix
    glm::mat4 projection;
    // Create the projection matrix using perspective projection with a 45
    // degreee field of view, an aspect ration of 800/600, and a near and far
    // plane of 0.1 and 100.0
    float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
    projection =
        glm::perspective(glm::radians(camera.Zoom), aspect, 0.1f, 1000.0f);
    glm::mat4 viewModel = view * model;
    glm::mat4 mvp = projection * viewModel;

    if (mvp != lastMvp) {
      auto t0 = std::chrono::high_resolution_clock::now();
      visibleSplats = sortSplatsCuda(sorter, glm::value_ptr(viewModel),
                                     glm::value_ptr(mvp));
      auto t1 = std::chrono::high_resolution_clock::now();
      lastSortsMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      lastMvp = mvp;
    }

    ourShader.use();

    // glUniform4f(vertexColorLocation, 0.0f, greenValue, 0.0f, 1.0f);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));

    glUniform2f(viewportLoc, static_cast<float>(fbWidth),
                static_cast<float>(fbHeight));

    glBindVertexArray(VAO);
    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    if (visibleSplats > 0) {
      glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, visibleSplats);
    }
    glBindVertexArray(0);
    glfwSwapBuffers(window);

    frameCounter++;
    double now = glfwGetTime();
    if (now - statsTime >= 0.5) {
      double avgFrameMs = ((now - statsTime) * 1000.0) / frameCounter;
      double fps = frameCounter / (now - statsTime);
      std::cout << "\r[PERF] FPS: " << std::fixed << std::setprecision(1) << fps
                << "| Frame: " << std::setprecision(2) << avgFrameMs << " ms"
                << "| CUDA Sort: " << lastSortsMs << "ms     "
                << "| Visible: " << visibleSplats << std::flush;
      frameCounter = 0;
      statsTime = now;
    }
    glfwPollEvents();
  }
  freeCudaSorter(sorter);
  glDeleteBuffers(1, &splatSSBO);
  glDeleteBuffers(1, &indexSSBO);
  glDeleteBuffers(1, &VBO);
  glDeleteVertexArrays(1, &VAO);
  glfwTerminate();
  return 0;
}
