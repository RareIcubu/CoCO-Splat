#ifndef ENGINECONTEXT_H
#define ENGINECONTEXT_H

#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <string>
#include <vector>

#include "camera.h"
#include "shader.h"
#include <sorter.h>

struct BenchmarkSample {
  float timeOffsetSec;
  float fps;
  float frameTimeMs;
  float cudaSortMs;
  int visibleSplats;
};

struct BenchmarkRecorder {
  bool isRecording = false;
  double startTime = 0.0;
  double lastSampleTime = 0.0;
  const double durationSec = 5.0;
  const double sampleIntervalSec = 0.2;

  float accumFrameMs = 0.0f;
  float accumSortMs = 0.0f;
  long long accumVisible = 0;
  int framesInWindow = 0;

  std::vector<BenchmarkSample> samples;

  void start(double currentTime);
  void update(double currentTime, float frameMs, float sortMs, int visible,
              const std::string &modelName);
  void saveToFile(const std::string &modelName);
};

struct EngineContext {
  GLFWwindow *window = nullptr;
  GLuint shaderID = 0;
  GLuint VAO = 0;
  GLuint VBO = 0;
  GLuint ssboSplats = 0;
  GLuint ssboIndices = 0;
  CudaSorter sorter{};

  int totalActiveSplats = 0;
  float rawPlySizeMiB = 0.0f;
  float optimizedVramMiB = 0.0f;
  std::string modelName = "model";
  float bgColor[3] = {0.0f, 0.0f, 0.0f};
};

void runCleanLoop(EngineContext &ctx, Camera &camera);
void runImGuiLoop(EngineContext &ctx, Camera &camera);
void runScreenshotLoop(EngineContext &ctx, const std::string &camerasFilePath,
                       const std::string &outputDir);

void destroyEngineContext(EngineContext &ctx);

#endif // ENGINECONTEXT_H
