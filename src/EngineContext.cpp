#include <glad/glad.h>

#include "EngineContext.h"
#include "camera.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iomanip>
#include <iostream>
#ifdef USE_IMGUI
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#endif

// ============================================================
// Implementation of BenchmarkRecorder
// ============================================================

bool setCameraViewModel = false;

void BenchmarkRecorder::start(double currentTime) {
  if (isRecording)
    return;
  isRecording = true;
  startTime = currentTime;
  lastSampleTime = currentTime;
  accumFrameMs = 0.0f;
  accumSortMs = 0.0f;
  accumVisible = 0;
  framesInWindow = 0;
  samples.clear();
  std::cout
      << "\n[F10 BENCHMARK] Rozpoczeto 5-sekundowe nagrywanie w locie...\n";
}

void BenchmarkRecorder::update(double currentTime, float frameMs, float sortMs,
                               int visible, const std::string &modelName) {
  if (!isRecording)
    return;

  accumFrameMs += frameMs;
  accumSortMs += sortMs;
  accumVisible += visible;
  framesInWindow++;

  if (currentTime - lastSampleTime >= sampleIntervalSec && framesInWindow > 0) {
    float avgFrame = accumFrameMs / framesInWindow;
    samples.push_back({static_cast<float>(currentTime - startTime),
                       1000.0f / avgFrame, avgFrame,
                       accumSortMs / framesInWindow,
                       static_cast<int>(accumVisible / framesInWindow)});
    lastSampleTime = currentTime;
    accumFrameMs = 0.0f;
    accumSortMs = 0.0f;
    accumVisible = 0;
    framesInWindow = 0;
  }

  if (currentTime - startTime >= durationSec) {
    isRecording = false;
    saveToFile(modelName);
  }
}

void BenchmarkRecorder::saveToFile(const std::string &modelName) {
  if (samples.empty())
    return;

  std::string filename = "benchmark_" + modelName + ".txt";
  std::ofstream out(filename, std::ios::app);

  float sumFps = 0.0f, sumFrame = 0.0f, sumSort = 0.0f;
  long long sumVis = 0;

  out << "================================================================\n";
  out << "Model: " << modelName << "\n";
  out << "Time(s)\tFPS\tFrame(ms)\tCUDASort(ms)\tVisibleSplats\n";

  for (const auto &s : samples) {
    out << std::fixed << std::setprecision(2) << s.timeOffsetSec << "\t"
        << s.fps << "\t" << s.frameTimeMs << "\t\t" << s.cudaSortMs << "\t\t"
        << s.visibleSplats << "\n";
    sumFps += s.fps;
    sumFrame += s.frameTimeMs;
    sumSort += s.cudaSortMs;
    sumVis += s.visibleSplats;
  }

  float n = static_cast<float>(samples.size());
  out << "----------------------------------------------------------------\n";
  out << std::fixed << std::setprecision(2) << "AVERAGE:\t" << (sumFps / n)
      << " FPS | "
      << "Frame: " << (sumFrame / n) << " ms | "
      << "CUDA Sort: " << (sumSort / n) << " ms | "
      << "Visible: " << (sumVis / samples.size()) << "\n";
  out << "================================================================\n\n";

  std::cout << "\n[F10 BENCHMARK] Zapisano " << samples.size()
            << " probek (Srednio: " << std::fixed << std::setprecision(1)
            << (sumFps / n) << " FPS, Sort: " << std::setprecision(2)
            << (sumSort / n) << " ms) do pliku: " << filename << "\n";
}

// ============================================================
// Internal movment input handler (WASD + Shift + ESC + F10)
// ============================================================

bool wasF5Pressed = false;

static void handleMovementInput(GLFWwindow *window, Camera &camera,
                                float deltaTime, BenchmarkRecorder &recorder) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  else if (glfwGetKey(window, GLFW_KEY_X) == GLFW_RELEASE)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  // Start recording benchmark when F10 is pressed
  if (glfwGetKey(window, GLFW_KEY_F10) == GLFW_PRESS)
    recorder.start(glfwGetTime());

  bool isf5preseed = (GLFW_PRESS == glfwGetKey(window, GLFW_KEY_F5));

  if (isf5preseed && !wasF5Pressed) {
    setCameraViewModel = !setCameraViewModel;
    std::cout << "[INFO] Camera view model set to: "
              << (setCameraViewModel ? "COLMAP" : "Free") << "\n";
  }

  wasF5Pressed = isf5preseed;

  float speed =
      (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 8.0f : 1.0f;

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::FORWARD, deltaTime * speed);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::BACKWARD, deltaTime * speed);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::LEFT, deltaTime * speed);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(CameraDirection::RIGHT, deltaTime * speed);
}

// ============================================================
// Loop 1: Main clean loop without any telemetry
// ============================================================

void runCleanLoop(EngineContext &ctx, Camera &camera) {
  int visibleSplats = 0;
  double lastSortsMs = 0.0;
  glm::mat4 lastMvp(0.0f);

  const int modelLoc = glGetUniformLocation(ctx.shaderID, "model");
  const int viewLoc = glGetUniformLocation(ctx.shaderID, "view");
  const int projectionLoc = glGetUniformLocation(ctx.shaderID, "projection");
  const int viewportLoc = glGetUniformLocation(ctx.shaderID, "viewport");

  glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
  glUseProgram(ctx.shaderID);
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  double statsTime = glfwGetTime();
  double lastFrame = statsTime;
  int frameCounter = 0;
  BenchmarkRecorder recorder;

  while (!glfwWindowShouldClose(ctx.window)) {
    double currentFrame = glfwGetTime();
    float deltaTime = static_cast<float>(currentFrame - lastFrame);
    lastFrame = currentFrame;

    handleMovementInput(ctx.window, camera, deltaTime, recorder);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(ctx.window, &fbWidth, &fbHeight);
    if (fbWidth == 0 || fbHeight == 0) {
      glfwPollEvents();
      continue;
    }

    glClearColor(ctx.bgColor[0], ctx.bgColor[1], ctx.bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view = camera.GetViewMatrix();
    float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
    glm::mat4 projection =
        glm::perspective(glm::radians(camera.Zoom), aspect, 0.1f, 1000.0f);
    glm::mat4 viewModel = view * model;
    glm::mat4 mvp = projection * viewModel;

    if (mvp != lastMvp || recorder.isRecording) {
      auto t0 = std::chrono::high_resolution_clock::now();
      float screenScalePromil = projection[1][1] * 500.0f;
      visibleSplats = sortSplatsCuda(ctx.sorter, glm::value_ptr(viewModel),
                                     glm::value_ptr(mvp), screenScalePromil);
      auto t1 = std::chrono::high_resolution_clock::now();
      lastSortsMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      lastMvp = mvp;
    }

    glUseProgram(ctx.shaderID);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform2f(viewportLoc, static_cast<float>(fbWidth),
                static_cast<float>(fbHeight));

    glBindVertexArray(ctx.VAO);
    if (visibleSplats > 0) {
      glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, visibleSplats);
    }
    glBindVertexArray(0);

    glfwSwapBuffers(ctx.window);

    double afterSwap = glfwGetTime();
    float frameMs = static_cast<float>((afterSwap - currentFrame) * 1000.0);
    recorder.update(afterSwap, frameMs, static_cast<float>(lastSortsMs),
                    visibleSplats, ctx.modelName);

    frameCounter++;
    if (afterSwap - statsTime >= 0.5) {
      double avgFrameMs = ((afterSwap - statsTime) * 1000.0) / frameCounter;
      double fps = frameCounter / (afterSwap - statsTime);
      std::cout << "\r[CLEAN] FPS: " << std::fixed << std::setprecision(1)
                << fps << " | Frame: " << std::setprecision(2) << avgFrameMs
                << " ms"
                << " | CUDA Sort: " << lastSortsMs << " ms"
                << " | Visible: " << visibleSplats << "      " << std::flush;
      frameCounter = 0;
      statsTime = afterSwap;
    }
    glfwPollEvents();
  }
}

// ============================================================
// Loop 2: imGui lopp with telemetry and HUD
// ============================================================
struct TestCamera {
  std::string imageName;
  int widht;
  int height;
  float fx;
  float fy;
  glm::mat4 w2c; // World-To-Camera transformation matrix
};

void readCamerasFromFile(const std::string &filePath,
                         std::vector<TestCamera> &cameras) {
  std::ifstream file(filePath);
  if (!file.is_open()) {
    std::cerr << "[ERROR] Cannot open cameras file: " << filePath << "\n";
    return;
  }

  cameras.clear();
  int numCameras = 0;
  if (!(file >> numCameras)) {
    std::cerr << "[ERROR] Failed to read number of cameras from file: "
              << filePath << "\n";
    return;
  }
  for (int i = 0; i < numCameras; ++i) {
    TestCamera cam;
    if (!(file >> cam.imageName >> cam.widht >> cam.height >> cam.fx >>
          cam.fy)) {
      std::cerr << "[ERROR] Failed to read camera parameters" << i << "\n";
      return;
    }
    float m[16];
    for (int j = 0; j < 16; ++j) {
      file >> m[j];
    }
    cam.w2c = glm::make_mat4(m);
    cameras.push_back(cam);
  }
  file.close();
}

void runImGuiLoop(EngineContext &ctx, Camera &camera,
                  const std::string &camerasFilepath) {
#ifndef USE_IMGUI
  std::cerr << "[WARN] Compiled with without USE_IMGUI flag! Preparing to run "
               "main loop...\n";
  runCleanLoop(ctx, camera);
#else
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(ctx.window, true);
  ImGui_ImplOpenGL3_Init("#version 460");

  int visibleSplats = 0;
  double lastSortsMs = 0.0;
  glm::mat4 lastMvp(0.0f);

  const int modelLoc = glGetUniformLocation(ctx.shaderID, "model");
  const int viewLoc = glGetUniformLocation(ctx.shaderID, "view");
  const int projectionLoc = glGetUniformLocation(ctx.shaderID, "projection");
  const int viewportLoc = glGetUniformLocation(ctx.shaderID, "viewport");

  glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
  glUseProgram(ctx.shaderID);
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  double lastFrame = glfwGetTime();
  float smoothedFrameMs = 16.0f;
  bool showHud = true;
  bool cursorUnlocked = false;
  bool tabPressedLastFrame = false;
  bool f1PressedLastFrame = false;
  BenchmarkRecorder recorder;

  std::vector<TestCamera> cameras;
  if (std::filesystem::exists(camerasFilepath)) {
    readCamerasFromFile(camerasFilepath, cameras);
    std::cout << "[INFO] Loaded " << cameras.size()
              << " cameras from: " << camerasFilepath << "\n";
  } else {
    std::cout << "Info: Cameras file not found: " << camerasFilepath
              << ". Continuing without camera view model.\n";
  }

  while (!glfwWindowShouldClose(ctx.window)) {
    double currentFrame = glfwGetTime();
    float deltaTime = static_cast<float>(currentFrame - lastFrame);
    lastFrame = currentFrame;
    smoothedFrameMs = 0.9f * smoothedFrameMs + 0.1f * (deltaTime * 1000.0f);

    bool tabNow = (glfwGetKey(ctx.window, GLFW_KEY_TAB) == GLFW_PRESS);
    if (tabNow && !tabPressedLastFrame) {
      cursorUnlocked = !cursorUnlocked;
      glfwSetInputMode(ctx.window, GLFW_CURSOR,
                       cursorUnlocked ? GLFW_CURSOR_NORMAL
                                      : GLFW_CURSOR_DISABLED);
    }
    tabPressedLastFrame = tabNow;

    bool f1Now = (glfwGetKey(ctx.window, GLFW_KEY_F1) == GLFW_PRESS);
    if (f1Now && !f1PressedLastFrame)
      showHud = !showHud;
    f1PressedLastFrame = f1Now;

    handleMovementInput(ctx.window, camera, deltaTime, recorder);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(ctx.window, &fbWidth, &fbHeight);
    if (fbWidth == 0 || fbHeight == 0) {
      glfwPollEvents();
      continue;
    }

    glClearColor(ctx.bgColor[0], ctx.bgColor[1], ctx.bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 view;
    glm::mat4 projection(0.0f);
    glm::mat4 viewModel;
    glm::mat4 mvp;
    if (setCameraViewModel && !cameras.empty()) {

      const TestCamera cam = cameras[0];

      view = model * cam.w2c * model;
      viewModel = model * cam.w2c;

      // 2. Projection matrix from COLMAP's intrinsics
      float zNear = 0.1f;
      float zFar = 1000.0f;
      float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
      projection[1][1] = (2.0f * cam.fy) / static_cast<float>(cam.height);
      projection[0][0] = projection[1][1] / aspect;
      projection[2][2] = -(zFar + zNear) / (zFar - zNear);
      projection[2][3] = -1.0f;
      projection[3][2] = -(2.0f * zFar * zNear) / (zFar - zNear);

      mvp = projection * viewModel;
    } else {

      view = camera.GetViewMatrix();
      float aspect = static_cast<float>(fbWidth) / static_cast<float>(fbHeight);
      projection =
          glm::perspective(glm::radians(camera.Zoom), aspect, 0.1f, 1000.0f);
      viewModel = view * model;
      mvp = projection * viewModel;
    }
    if (mvp != lastMvp || recorder.isRecording) {
      auto t0 = std::chrono::high_resolution_clock::now();
      float screenScalePromil = projection[1][1] * 500.0f;
      visibleSplats = sortSplatsCuda(ctx.sorter, glm::value_ptr(viewModel),
                                     glm::value_ptr(mvp), screenScalePromil);
      auto t1 = std::chrono::high_resolution_clock::now();
      lastSortsMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
      lastMvp = mvp;
    }

    glUseProgram(ctx.shaderID);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform2f(viewportLoc, static_cast<float>(fbWidth),
                static_cast<float>(fbHeight));

    glBindVertexArray(ctx.VAO);
    if (visibleSplats > 0) {
      glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, visibleSplats);
    }
    glBindVertexArray(0);

    if (showHud) {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      ImGui::SetNextWindowPos(ImVec2(15, 15), ImGuiCond_FirstUseEver);
      ImGui::SetNextWindowBgAlpha(0.75f);
      ImGuiWindowFlags flags =
          ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings;
      if (!cursorUnlocked)
        flags |= ImGuiWindowFlags_NoInputs;

      ImGui::Begin("CoCO-Splat Telemetry", nullptr, flags);
      ImGui::Text("FPS: %.1f (%.2f ms/frame)", 1000.0f / smoothedFrameMs,
                  smoothedFrameMs);
      ImGui::Text("CUDA Radix Sort: %.2f ms", lastSortsMs);
      ImGui::Separator();
      ImGui::Text("Total Active Splats: %d", ctx.totalActiveSplats);
      ImGui::Text("Visible (Frustum):   %d (%.1f%%)", visibleSplats,
                  ctx.totalActiveSplats > 0
                      ? (100.0f * visibleSplats / ctx.totalActiveSplats)
                      : 0.0f);
      ImGui::Text("VRAM Splat Buffer:   %.2f MiB (28 B/splat)",
                  ctx.optimizedVramMiB);
      ImGui::Separator();
      ImGui::ColorEdit3("BG Color", ctx.bgColor);
      if (recorder.isRecording) {
        ImGui::TextColored(ImVec4(1.0f, 0.3f, 0.3f, 1.0f),
                           "[REC F10] %.1f / 5.0 s",
                           currentFrame - recorder.startTime);
      } else {
        ImGui::TextDisabled(
            "[F10] Record 5s | [TAB] Cursor | [F1] HUD | [F5] Camera");
      }
      ImGui::End();

      ImGui::Render();
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    glfwSwapBuffers(ctx.window);
    recorder.update(glfwGetTime(), deltaTime * 1000.0f,
                    static_cast<float>(lastSortsMs), visibleSplats,
                    ctx.modelName);
    glfwPollEvents();
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
#endif
}

// ============================================================
// Loop 3: Screenshot loop
// ============================================================

void runScreenshotLoop(EngineContext &ctx, const std::string &camerasFilePath,
                       const std::string &outputDir) {

  std::vector<TestCamera> cameras;
  readCamerasFromFile(camerasFilePath, cameras);
  if (cameras.empty()) {
    std::cerr << "[ERROR] No cameras loaded. Exiting screenshot loop.\n";
    return;
  }

  std::filesystem::create_directories(outputDir);

  std::cout << "[SCREENSHOT MODE] Wczytywanie kamer z: " << camerasFilePath
            << "\n";

  const int modelLoc = glGetUniformLocation(ctx.shaderID, "model");
  const int viewLoc = glGetUniformLocation(ctx.shaderID, "view");
  const int projectionLoc = glGetUniformLocation(ctx.shaderID, "projection");
  const int viewportLoc = glGetUniformLocation(ctx.shaderID, "viewport");

  // Changing COLMAP's projection matrix to OpenGL's projection matrix
  glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(1.0f, -1.0f, -1.0f));
  glUseProgram(ctx.shaderID);
  glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));

  int currW = cameras[0].widht;
  int currH = cameras[0].height;

  GLuint screenshotFBO, colorTex;
  glGenFramebuffers(1, &screenshotFBO);
  glBindFramebuffer(GL_FRAMEBUFFER, screenshotFBO);

  glGenTextures(1, &colorTex);
  glBindTexture(GL_TEXTURE_2D, colorTex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, currW, currH, 0, GL_RGB,
               GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         colorTex, 0);
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "[ERROR] Screenshot framebuffer is not complete!\n";
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteTextures(1, &colorTex);
    glDeleteFramebuffers(1, &screenshotFBO);
    return;
  }

  // Critical for images that widht cannot be divided by 4 (like 1920x1080)
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  // Flipping point (0,0) from bottom-left to top-left
  stbi_flip_vertically_on_write(1);

  std::vector<unsigned char> pixelBuffer(currW * currH * 3);

  for (size_t i = 0; i < cameras.size(); ++i) {
    const auto &cam = cameras[i];

    if (cam.widht != currW || cam.height != currH) {
      currW = cam.widht;
      currH = cam.height;
      glBindTexture(GL_TEXTURE_2D, colorTex);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, currW, currH, 0, GL_RGB,
                   GL_UNSIGNED_BYTE, nullptr);
      pixelBuffer.resize(currW * currH * 3);
    }

    glViewport(0, 0, currW, currH);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 1. Conversion from COLMAP's world-to-camera to OpenGL's view matrix
    glm::mat4 view = model * cam.w2c * model;
    glm::mat4 viewModel = model * cam.w2c;

    // 2. Projection matrix from COLMAP's intrinsics
    float zNear = 0.1f;
    float zFar = 1000.0f;
    glm::mat4 projection(0.0f);
    projection[0][0] = (2.0f * cam.fx) / static_cast<float>(currW);
    projection[1][1] = (2.0f * cam.fy) / static_cast<float>(currH);
    projection[2][2] = -(zFar + zNear) / (zFar - zNear);
    projection[2][3] = -1.0f;
    projection[3][2] = -(2.0f * zFar * zNear) / (zFar - zNear);

    glm::mat4 mvp = projection * viewModel;

    // 3. sort splats using CUDA

    float screenScalePromil = projection[1][1] * 500.0f;
    int visibleSplats = sortSplatsCuda(ctx.sorter, glm::value_ptr(viewModel),
                                       glm::value_ptr(mvp), screenScalePromil);

    // 4. Drawing to FBO
    glUseProgram(ctx.shaderID);
    glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(projection));
    glUniform2f(viewportLoc, static_cast<float>(currW),
                static_cast<float>(currH));

    glBindVertexArray(ctx.VAO);
    if (visibleSplats > 0) {
      glDrawArraysInstanced(GL_TRIANGLE_STRIP, 0, 4, visibleSplats);
    }
    glBindVertexArray(0);
    glFinish();

    // 5. Read from FBO and save to PNG
    glReadPixels(0, 0, currW, currH, GL_RGB, GL_UNSIGNED_BYTE,
                 pixelBuffer.data());
    std::filesystem::path outPath =
        std::filesystem::path(outputDir) / cam.imageName;
    stbi_write_png(outPath.string().c_str(), currW, currH, 3,
                   pixelBuffer.data(), currW * 3);

    std::cout << "\r[SCREENSHOT MODE] Saved " << cam.imageName << " ("
              << (i + 1) << "/" << cameras.size() << ") with " << visibleSplats
              << " visible splats.            " << std::flush;
  }

  std::cout << "\n[SCREENSHOT MODE] Finished saving " << cameras.size()
            << " images to: " << outputDir << "\n";

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteTextures(1, &colorTex);
  glDeleteFramebuffers(1, &screenshotFBO);
}

void destroyEngineContext(EngineContext &ctx) {
  freeCudaSorter(ctx.sorter);
  glDeleteBuffers(1, &ctx.ssboSplats);
  glDeleteBuffers(1, &ctx.ssboIndices);
  glDeleteBuffers(1, &ctx.VBO);
  glDeleteVertexArrays(1, &ctx.VAO);
  glfwTerminate();
}
