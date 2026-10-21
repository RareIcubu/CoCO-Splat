#include "parser.h"
#include <algorithm>
#include <fstream>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <sstream>
#include <vector>

void GaussianSplatParser::readPlyFile(const std::string &filename) {
  // Reading the PLY file is a two pahse process:
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "Nie mogę otworzyć pliku: " << filename << std::endl;
    return;
  }

  std::string line;
  int rawVertexCount = 0;

  // Phase 1: Read the header (text)
  while (std::getline(file, line)) {
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
    } else if (keyword == "end_header") {
      break;
    }
  }

  //  Phase 2: Read the binary data

  if (rawVertexCount > 0) {
    // 1. Reserved space for the data
    splats.reserve(rawVertexCount / 2);

    const size_t CHUNK_SIZE = 65536;
    std::vector<GaussianSplatRaw> chunk(CHUNK_SIZE);

    int remaining = rawVertexCount;
    int prunedScale = 0;
    int prunedOpacity = 0;

    while (remaining > 0) {
      int toRead = std::min(remaining, static_cast<int>(CHUNK_SIZE));
      file.read(reinterpret_cast<char *>(chunk.data()),
                toRead * sizeof(GaussianSplatRaw));
      int actuallyRead = file.gcount() / sizeof(GaussianSplatRaw);

      for (size_t i = 0; i < actuallyRead; ++i) {
        const auto &raw = chunk[i];

        float maxScale = std::max({raw.scale[0], raw.scale[1], raw.scale[2]});
        if (maxScale > -1.0f) {
          ++prunedScale;
          continue; // Skip this splat
        }

        if (raw.opacity < -4.0f) {
          ++prunedOpacity;
          continue; // Skip this splat
        }

        GaussianSplatFP16 s;
        s.pos[0] = glm::packHalf1x16(raw.x);
        s.pos[1] = glm::packHalf1x16(raw.y);
        s.pos[2] = glm::packHalf1x16(raw.z);

        for (int c = 0; c < 3; ++c) {
          s.f_dc[c] = glm::packHalf1x16(raw.f_dc[c]);
          s.scale[c] = glm::packHalf1x16(raw.scale[c]);
        }

        s.opacity = glm::packHalf1x16(raw.opacity);

        for (int r = 0; r < 4; ++r) {
          s.rot[r] = glm::packHalf1x16(raw.rot[r]);
        }
        splats.push_back(s);
      }
      remaining -= actuallyRead;
      if (actuallyRead == 0)
        break;
    }
    splats.shrink_to_fit();
    vertexCount = static_cast<int>(splats.size());

    double rawMb =
        (static_cast<double>(rawVertexCount) * sizeof(GaussianSplatRaw)) /
        (1024.0 * 1024.0);
    double optMB =
        (static_cast<double>(vertexCount) * sizeof(GaussianSplatFP16)) /
        (1024.0 * 1024.0);

    std::cout << "=== MEMORY OPTIMIZATION SUMMARY ===" << std::endl;
    std::cout << "Discarded (Scale Pruning):   " << prunedScale << " splats."
              << std::endl;
    std::cout << "Discarded (Opacity Pruning): " << prunedOpacity << " splats."
              << std::endl;
    std::cout << "Active splats remaining: " << vertexCount << " ("
              << (100.0 * vertexCount / rawVertexCount) << "%" << std::endl;
    std::cout << "VRAM buffer size before optimization: " << rawMb
              << " MiB (FP32 + SH3)" << std::endl;
    std::cout << "VRAM buffer size after optimization:  " << optMB
              << " MiB (FP16 + SH0 + Pruning)" << std::endl;
  }
}
