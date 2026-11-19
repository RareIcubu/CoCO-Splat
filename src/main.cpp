#include <glad/glad.h>

#include <GLFW/glfw3.h>
#include <filesystem>
#include <glad/glad.h>
#include <iostream>

#include "EngineContext.h"
#include "camera.h"
#include "parser.h"
#include "shader.h"

Camera gCamera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = 1920.0f / 2.0f, lastY = 1080.0f / 2.0f;
bool firstMouse = true;

void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow *window, double xpos, double ypos) {
  if (glfwGetInputMode(window, GLFW_CURSOR) != GLFW_CURSOR_DISABLED) {
    firstMouse = true;
    return;
  }

  if (firstMouse) {
    lastX = static_cast<float>(xpos);
    lastY = static_cast<float>(ypos);
    firstMouse = false;
  }

  float xoffset = static_cast<float>(xpos) - lastX;
  float yoffset = lastY - static_cast<float>(ypos);
  lastX = static_cast<float>(xpos);
  lastY = static_cast<float>(ypos);

  gCamera.ProcessMouseMovement(xoffset, yoffset);
}

void scroll_callback(GLFWwindow *window, double xoffset, double yoffset) {
  gCamera.ProcessMouseScroll(static_cast<float>(yoffset));
}

static const float quadVertices[] = {-1.0f, -1.0f, 1.0f, -1.0f,
                                     -1.0f, 1.0f,  1.0f, 1.0f};

int main() {
  const char *envPath = std::getenv("PLY_FILE_PATH");
  std::string filename = envPath ? envPath : "../../DELL.ply";

  GaussianSplatParser parser;
  parser.readPlyFile(filename);
  if (parser.splats.empty()) {
    std::cerr << "No splats found in the PLY file." << std::endl;
    return -1;
  }

  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow *window =
      glfwCreateWindow(1920, 1080, "CoCO-Splat", nullptr, nullptr);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSwapInterval(0);
  glfwSetInputMode(window, GLFW_STICKY_KEYS, 1);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  glViewport(0, 0, 1920, 1080);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetScrollCallback(window, scroll_callback);

  EngineContext ctx;
  ctx.window = window;
  ctx.totalActiveSplats = parser.vertexCount;
  ctx.optimizedVramMiB =
      (parser.splats.size() * sizeof(GaussianSplatFP16)) / (1024.0f * 1024.0f);
  ctx.modelName = std::filesystem::path(filename).stem().string();

  glGenBuffers(1, &ctx.VBO);
  glGenVertexArrays(1, &ctx.VAO);
  glBindVertexArray(ctx.VAO);
  glBindBuffer(GL_ARRAY_BUFFER, ctx.VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices,
               GL_STATIC_DRAW);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glGenBuffers(1, &ctx.ssboSplats);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx.ssboSplats);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               parser.splats.size() * sizeof(GaussianSplatFP16),
               parser.splats.data(), GL_STATIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, ctx.ssboSplats);

  parser.splats.clear();
  parser.splats.shrink_to_fit();

  glGenBuffers(1, &ctx.ssboIndices);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, ctx.ssboIndices);
  glBufferData(GL_SHADER_STORAGE_BUFFER,
               static_cast<size_t>(ctx.totalActiveSplats) *
                   sizeof(unsigned int),
               nullptr, GL_DYNAMIC_DRAW);
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, ctx.ssboIndices);

  initCudaSorter(ctx.sorter, ctx.ssboSplats, ctx.ssboIndices,
                 ctx.totalActiveSplats);

  std::filesystem::path exeDir =
      std::filesystem::canonical("/proc/self/exe").parent_path();
  std::filesystem::path vertPath = exeDir / "../shaders/shader.vert";
  std::filesystem::path fragPath = exeDir / "../shaders/shader.frag";

  Shader ourShader(vertPath.string().c_str(), fragPath.string().c_str());
  ctx.shaderID = ourShader.ID;

  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  const char *modeEnv = std::getenv("RENDER_MODE");
  std::string mode = modeEnv ? modeEnv : "clean";

  if (mode == "imgui") {
    const char *camEnv = std::getenv("TEST_CAMERAS_PATH");
    runImGuiLoop(ctx, gCamera, camEnv ? camEnv : "test_cameras.txt");
  } else if (mode == "screenshot") {
    const char *camEnv = std::getenv("TEST_CAMERAS_PATH");
    const char *outEnv = std::getenv("SCREENSHOT_OUT_DIR");
    runScreenshotLoop(ctx, camEnv ? camEnv : "test_cameras.txt",
                      outEnv ? outEnv : "renders");
  } else {
    runCleanLoop(ctx, gCamera);
  }

  destroyEngineContext(ctx);
  return 0;
}
