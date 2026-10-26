#ifndef PARSER_H
// Create parser class for .ply files for 3D Gaussian Splatting
#include <cstdint>
#include <string>
#include <vector>

struct GaussianSplatFP16 {
  uint16_t pos[3];
  uint16_t f_dc[3];
  uint16_t opacity;
  uint16_t scale[3];
  uint16_t rot[4];
}; // Everything: 28 bytes

struct PLYHeaderLayout {
  int pos[3] = {-1, -1, -1};
  int f_dc[3] = {-1, -1, -1};
  int opacity = -1;
  int scale[3] = {-1, -1, -1};
  int rot[4] = {-1, -1, -1, -1};
  size_t floatsPerVertex = 0;

  bool isValid() const {
    return pos[0] >= 0 && pos[1] >= 0 && pos[2] >= 0 && f_dc[0] >= 0 &&
           f_dc[1] >= 0 && f_dc[2] >= 0 && opacity >= 0 && scale[0] >= 0 &&
           scale[1] >= 0 && scale[2] >= 0 && rot[0] >= 0 && rot[1] >= 0 &&
           rot[2] >= 0 && rot[3] >= 0 && floatsPerVertex >= 14;
  }
};

class GaussianSplatParser {
public:
  std::vector<GaussianSplatFP16> splats;
  int vertexCount = 0;

  void readPlyFile(const std::string &filename);
};
#endif // !PARSER_H
