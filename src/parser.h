#ifndef PARSER_H
// Create parser class for .ply files for 3D Gaussian Splatting
#include <cstdint>
#include <string>
#include <vector>

struct GaussianSplatRaw {
  float x, y, z; // Position
  float f_dc[3]; // SH color coefficients for degree 3 (9 coefficients per color
                 // channel)
  float f_rest[45]; // If model has more SH degree 3, we need to store the
  //  remaining coefficients
  float opacity; // Opacity before applying sigmoid function
  float scale[3];
  float rot[4]; // Quaternion rotation
};

struct GaussianSplatFP16 {
  uint16_t pos[3];   // Position in 16-bit float 6B
  uint16_t f_dc[3];  // SH color 6B
  uint16_t opacity;  // Opacity before applying sigmoid function 2B
  uint16_t scale[3]; // Scale in 16-bit float 6B
  uint16_t rot[4];   // Quaternion rotation 8B
}; // Total 28B per splat

class GaussianSplatParser {
public:
  std::vector<GaussianSplatFP16> splats;
  int vertexCount = 0;

  void readPlyFile(const std::string &filename);
};
#endif // !PARSER_H
