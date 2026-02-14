#ifndef PARSER_H
// Create parser class for .ply files for 3D Gaussian Splatting
#include <string>
#include <vector>

struct GaussianSplat {
  float x, y, z; // Position
  // float nx, ny, nz; // Normal
  float f_dc[3]; // SH color coefficients for degree 3 (9 coefficients per color
                 // channel)
  float f_rest[45]; // If model has more SH degree 3, we need to store the
                    // remaining coefficients
  float opacity;    // Opacity before applying sigmoid function
  float scale[3];
  float rot[4]; // Quaternion rotation
};
class GaussianSplatParser {
public:
  std::vector<GaussianSplat> splats;
  int vertexCount;

  void readPlyFile(const std::string &filename);
};
#endif // !PARSER_H
