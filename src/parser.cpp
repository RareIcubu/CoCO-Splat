#include "parser.h"
#include <fstream>
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
        ss >> vertexCount;
        std::cout << "Znaleziono " << vertexCount << " splatów." << std::endl;
      }
    } else if (keyword == "end_header") {
      break;
    }
  }

  //  Phase 2: Read the binary data

  if (vertexCount > 0) {
    // 1. Reserved space for the data
    splats.resize(vertexCount);

    // 2. Count how many bytes we need to read
    size_t sizeInBytes = vertexCount * sizeof(GaussianSplat);

    // 3. Read the binary data in one command
    file.read(reinterpret_cast<char *>(splats.data()), sizeInBytes);

    if (file) {
      std::cout << "Successfully read: " << sizeInBytes << " bytes of data"
                << std::endl;
    } else {
      std::cerr << "Read only " << file.gcount()
                << " bytes. File might be invalid or struct is wrong"
                << std::endl;
    }
  }
}
