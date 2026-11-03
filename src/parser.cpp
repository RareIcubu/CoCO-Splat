#include "parser.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <limits>
#include <omp.h>
#include <sstream>
#include <vector>

float opacityThreshold =
    std::numeric_limits<float>::min(); // Threshold for opacity pruning
float scaleThreshold =
    std::numeric_limits<float>::max(); // Threshold for scale pruning

const bool ENABLE_OPACITY_PRUNING =
    false;                              // Set to true to enable opacity pruning
const bool ENABLE_SCALE_PRUNING = true; // Set to true to enable scale pruning

void GaussianSplatParser::readPlyFile(const std::string &filename) {
  // Reading the PLY file is a two phase process:

  auto startTime = std::chrono::high_resolution_clock::now();

  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Cannot open file: " << filename << std::endl;
    return;
  }

  std::string line;
  size_t rawVertexCount = 0;
  PLYHeaderLayout layout;

  // Phase 1: Dynamically read header and column mapping
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back(); // Remove carriage return if present
    }

    std::stringstream ss(line);
    std::string keyword;
    ss >> keyword;

    if (keyword == "format") {
      std::string type;
      ss >> type;
      if (type != "binary_little_endian") {
        std::cerr << "What is this format? Are you using old IBM PC for "
                     "Gaussian Splatting? Are you mad? Little Endian is the "
                     "only supported format!"
                  << std::endl;
        return;
      }
    } else if (keyword == "element") {
      std::string type;
      ss >> type;
      if (type == "vertex") {
        ss >> rawVertexCount;
        std::cout << "Found: " << rawVertexCount << " splats." << std::endl;
      }
    } else if (keyword ==
               "property") { // <-- POPRAWKA 1: Wyciągnięte z "element"
      std::string type, name;
      ss >> type >> name;

      int idx = static_cast<int>(layout.floatsPerVertex);
      if (name == "x")
        layout.pos[0] = idx;
      else if (name == "y")
        layout.pos[1] = idx;
      else if (name == "z")
        layout.pos[2] = idx;
      else if (name == "f_dc_0")
        layout.f_dc[0] = idx;
      else if (name == "f_dc_1")
        layout.f_dc[1] = idx;
      else if (name == "f_dc_2")
        layout.f_dc[2] = idx;
      else if (name == "scale_0")
        layout.scale[0] = idx;
      else if (name == "scale_1")
        layout.scale[1] = idx;
      else if (name == "scale_2")
        layout.scale[2] = idx;
      else if (name == "opacity")
        layout.opacity = idx;
      else if (name == "rot_0")
        layout.rot[0] = idx;
      else if (name == "rot_1")
        layout.rot[1] = idx;
      else if (name == "rot_2")
        layout.rot[2] = idx;
      else if (name == "rot_3")
        layout.rot[3] = idx;

      layout.floatsPerVertex++; // <-- POPRAWKA 2: Zwiększanie licznika kolumn!
    } else if (keyword == "end_header") {
      break;
    }
  }

  if (!layout.isValid()) {
    std::cerr << "Invalid PLY header layout. Please check the file format."
              << std::endl;
    return;
  }

  size_t bytesPerVertex = layout.floatsPerVertex * sizeof(float);
  std::cout << "Detected PLY stride: " << layout.floatsPerVertex
            << " floats per vertex, " << bytesPerVertex << " bytes per splat."
            << std::endl;

  // Phase 2: Read the binary data
  if (rawVertexCount > 0) {
    // 1. Reserved space for the data
    splats.reserve(rawVertexCount / 2);

    const size_t CHUNK_SPLATS = 1048576; // 1M of splats
    std::vector<float> chunk(CHUNK_SPLATS * layout.floatsPerVertex);

    size_t remaining = rawVertexCount;
    size_t prunedScale = 0;
    size_t prunedOpacity = 0;

    while (remaining > 0) {
      size_t toRead = std::min(remaining, CHUNK_SPLATS);
      file.read(reinterpret_cast<char *>(chunk.data()),
                toRead * bytesPerVertex);
      size_t actuallyRead = file.gcount() / bytesPerVertex;
      if (actuallyRead == 0)
        break;

#pragma omp parallel reduction(+ : prunedScale, prunedOpacity)
      {
        std::vector<GaussianSplatFP16> localSplats;
        localSplats.reserve(actuallyRead / omp_get_num_threads() + 1024);

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i < actuallyRead; ++i) {
          const float *v = &chunk[i * layout.floatsPerVertex];

          float s0 = v[layout.scale[0]];
          float s1 = v[layout.scale[1]];
          float s2 = v[layout.scale[2]];
          if (std::max({s0, s1, s2}) > scaleThreshold && ENABLE_SCALE_PRUNING) {
            ++prunedScale;
            continue; // Skip this splat
          }

          float op = v[layout.opacity];
          if (op < opacityThreshold && ENABLE_OPACITY_PRUNING) {
            ++prunedOpacity;
            continue; // Skip this splat
          }

          GaussianSplatFP16 s;
          s.pos[0] = glm::packHalf1x16(v[layout.pos[0]]);
          s.pos[1] = glm::packHalf1x16(v[layout.pos[1]]);
          s.pos[2] = glm::packHalf1x16(v[layout.pos[2]]);

          for (int c = 0; c < 3; ++c) {
            s.f_dc[c] = glm::packHalf1x16(v[layout.f_dc[c]]);
            s.scale[c] = glm::packHalf1x16(v[layout.scale[c]]);
          }

          s.opacity = glm::packHalf1x16(op);

          for (int r = 0; r < 4; ++r) {
            s.rot[r] = glm::packHalf1x16(v[layout.rot[r]]);
          }
          localSplats.push_back(s);
        }

#pragma omp critical
        {
          splats.insert(splats.end(), localSplats.begin(), localSplats.end());
        }
      } // <-- POPRAWKA 3: Koniec bloku #pragma omp parallel

      remaining -= actuallyRead; // Odejmowanie tylko raz w wątku głównym!
    } // <-- POPRAWKA 4: Koniec pętli while (remaining > 0)

    splats.shrink_to_fit();
    vertexCount = static_cast<int>(splats.size());

    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSeconds =
        std::chrono::duration<double>(endTime - startTime).count();

    double rawMb = (static_cast<double>(rawVertexCount) * bytesPerVertex) /
                   (1024.0 * 1024.0);
    double optMB =
        (static_cast<double>(vertexCount) * sizeof(GaussianSplatFP16)) /
        (1024.0 * 1024.0);

    std::cout << "=== MEMORY OPTIMIZATION SUMMARY ===" << std::endl;
    std::cout << "Parsing time (OpenMP " << omp_get_max_threads()
              << " threads): " << elapsedSeconds << " seconds." << std::endl;
    std::cout << "Discarded (Scale Pruning):   " << prunedScale << " splats."
              << std::endl;
    std::cout << "Discarded (Opacity Pruning): " << prunedOpacity << " splats."
              << std::endl;
    std::cout << "Active splats remaining: " << vertexCount << " ("
              << (100.0 * vertexCount / rawVertexCount) << "%)" << std::endl;
    std::cout << "VRAM buffer size before optimization: " << rawMb
              << " MiB (Raw PLY)" << std::endl;
    std::cout << "VRAM buffer size after optimization:  " << optMB
              << " MiB (FP16 + SH0 + Pruning)" << std::endl;
  }
}
