#ifndef SORTER_H
#define SORTER_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

struct CudaSorter {
  size_t numSplats = 0;
  cudaGraphicsResource *cudaSplatRes = nullptr;
  cudaGraphicsResource *cudaIndexRes = nullptr;

  uint16_t *d_keys_in = nullptr;
  uint16_t *d_keys_out = nullptr;
  uint32_t *d_vals_in = nullptr;
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
};

void initCudaSorter(CudaSorter &s, unsigned int splatSSBO,
                    unsigned int indexSSBO, size_t count);

void sortSplatsCuda(CudaSorter &s, const float *viewMatrix);

void freeCudaSorter(CudaSorter &s);

#endif // SORTER_H
