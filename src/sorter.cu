#include "sorter.h"
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <iostream>

constexpr bool ENABLE_FRUSTUM_CULLING = true;
// constexpr bool ENABLE_SCALE_CULLING = true;
constexpr bool ENABLE_PIXEL_CULLING = true;
constexpr bool ENABLE_LOD_CULLING = true;
// constexpr float OPACITY_THRESHOLD_LOGIT = -4.0f;

// Cuda kernel for culling and compacting splats based on depth, frustum, pixel
// size, and LOD.
__global__ void cullAndCompactKernel(
    const __half *__restrict__ splatsHalf, uint16_t *__restrict__ keysOut,
    uint32_t *__restrict__ valsOut, uint32_t *__restrict__ globalVisibleCount,
    size_t numSplats, float screenScalePromil, float vm2x, float vm2y,
    float vm2z, float vm2w, float m0, float m1, float m2, float m3, float m4,
    float m5, float m6, float m7, float m8, float m9, float m10, float m11,
    float m12, float m13, float m14, float m15) {
  __shared__ uint32_t blockCount;
  __shared__ uint32_t blockBaseOffset;

  if (threadIdx.x == 0)
    blockCount = 0;

  __syncthreads();

  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  bool keep = false;
  uint16_t myKey = 0;
  uint32_t localOffset = 0;

  if (idx < numSplats) {
    size_t base = idx * 14;

    float x = __half2float(splatsHalf[base + 0]);
    float y = __half2float(splatsHalf[base + 1]);
    float z = __half2float(splatsHalf[base + 2]);

    // Near-plane culling.
    float z_view = vm2x * x + vm2y * y + vm2z * z + vm2w;
    float depth = -z_view;

    if (depth > 0.1f) {
      float clipW = m3 * x + m7 * y + m11 * z + m15;

      bool frustumPass = (clipW > 0.1f);

      // Frustum culling.
      if constexpr (ENABLE_FRUSTUM_CULLING) {
        float clipX = m0 * x + m4 * y + m8 * z + m12;
        float clipY = m1 * x + m5 * y + m9 * z + m13;

        // Level of detail culling: we allow a small margin outside the frustum
        // to avoid popping artifacts.
        float limit = 1.2f * clipW;

        frustumPass = frustumPass && clipX >= -limit && clipX <= limit &&
                      clipY >= -limit && clipY <= limit;
      }

      // If the splat passes the frustum culling, we proceed to pixel size and
      // LOD culling.
      if (frustumPass) {

        // Calculate the pixel size of the splat based on its world radius and
        // depth.
        float worldRadius = __half2float(splatsHalf[base + 3]);
        float screenPromil = (worldRadius * screenScalePromil) / clipW;

        // Hard pixel-size culling.
        bool pixelPass = true;

        if constexpr (ENABLE_PIXEL_CULLING)
          pixelPass =
              (screenPromil >= 1.0f); // Threshold: 0.1% of the screen height.

        // Stochastic LOD.
        bool lodPass = true;

        if constexpr (ENABLE_LOD_CULLING) {
          if (pixelPass && screenPromil < 3.5f) { // LOD threshold: from 0.1% to
                                                  // 0.35% of the screen height.
            float ratio = (screenPromil - 1.0f) / (3.5f - 1.0f);

            ratio = fminf(fmaxf(ratio, 0.0f), 1.0f);

            float keepProb = ratio * ratio;

            uint32_t hash =
                static_cast<uint32_t>(idx) * 747796405u + 2891336453u;

            hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;

            float rand01 =
                static_cast<float>(hash & 0xFFFFu) * (1.0f / 65536.0f);

            lodPass = (rand01 < keepProb);
          }
        }

        if (pixelPass && lodPass) {
          keep = true;

          uint16_t halfBits = __half_as_ushort(__float2half(depth));

          myKey = ~halfBits;

          localOffset = atomicAdd(&blockCount, 1u);
        }
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0 && blockCount > 0)
    blockBaseOffset = atomicAdd(globalVisibleCount, blockCount);

  __syncthreads();

  if (keep) {
    uint32_t finalIdx = blockBaseOffset + localOffset;

    keysOut[finalIdx] = myKey;
    valsOut[finalIdx] = static_cast<uint32_t>(idx);
  }
}
void initCudaSorter(CudaSorter &s, unsigned int splatSSBO,
                    unsigned int indexSSBO, size_t count) {
  s.numSplats = count;
  s.d_splats_mapped = nullptr;

  cudaGraphicsGLRegisterBuffer(&s.cudaSplatRes, splatSSBO,
                               cudaGraphicsRegisterFlagsReadOnly);

  cudaGraphicsGLRegisterBuffer(&s.cudaIndexRes, indexSSBO,
                               cudaGraphicsRegisterFlagsWriteDiscard);

  cudaMalloc(&s.d_visible_count, sizeof(uint32_t));

  cudaMalloc(&s.d_keys_in, count * sizeof(uint16_t));
  cudaMalloc(&s.d_keys_out, count * sizeof(uint16_t));
  cudaMalloc(&s.d_vals_in, count * sizeof(uint32_t));

  // Set up CUB temporary storage for sorting.
  s.temp_storage_bytes = 0;

  uint32_t *dummy_vals_out = nullptr;

  cub::DeviceRadixSort::SortPairs(nullptr, s.temp_storage_bytes, s.d_keys_in,
                                  s.d_keys_out, s.d_vals_in, dummy_vals_out,
                                  static_cast<int>(count));

  cudaMalloc(&s.d_temp_storage, s.temp_storage_bytes);

  std::cout << "[CUDA] Compact Sorter initialized. CUB temp buffer: "
            << (s.temp_storage_bytes / (1024.0 * 1024.0)) << " MiB"
            << std::endl;
}

int sortSplatsCuda(CudaSorter &s, const float *vm, const float *mvp,
                   float screenScalePromil) {
  if (s.numSplats == 0)
    return 0;

  // 1. Mapping splats SSBO to CUDA.
  cudaGraphicsMapResources(1, &s.cudaSplatRes, 0);

  size_t splatBytes = 0;

  cudaGraphicsResourceGetMappedPointer((void **)&s.d_splats_mapped, &splatBytes,
                                       s.cudaSplatRes);

  // 2. Zeroing the visible count before culling.
  cudaMemsetAsync(s.d_visible_count, 0, sizeof(uint32_t));

  // 3. Running the culling and compaction kernel.
  constexpr int threadsPerBlock = 256;

  int blocksPerGrid =
      static_cast<int>((s.numSplats + threadsPerBlock - 1) / threadsPerBlock);

  cullAndCompactKernel<<<blocksPerGrid, threadsPerBlock>>>(
      static_cast<const __half *>(s.d_splats_mapped), s.d_keys_in, s.d_vals_in,
      s.d_visible_count, s.numSplats, screenScalePromil,

      vm[2], vm[6], vm[10], vm[14],

      mvp[0], mvp[1], mvp[2], mvp[3], mvp[4], mvp[5], mvp[6], mvp[7], mvp[8],
      mvp[9], mvp[10], mvp[11], mvp[12], mvp[13], mvp[14], mvp[15]);

  // 4. Retrieving the number of visible splats after culling.
  uint32_t visibleCount = 0;

  cudaMemcpy(&visibleCount, s.d_visible_count, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  // 5. Sorting the visible splats by depth using CUB radix sort.
  if (visibleCount > 0) {
    cudaGraphicsMapResources(1, &s.cudaIndexRes, 0);

    uint32_t *d_sorted_indices = nullptr;
    size_t indexBytes = 0;

    cudaGraphicsResourceGetMappedPointer((void **)&d_sorted_indices,
                                         &indexBytes, s.cudaIndexRes);

    cub::DeviceRadixSort::SortPairs(
        s.d_temp_storage, s.temp_storage_bytes, s.d_keys_in, s.d_keys_out,
        s.d_vals_in, d_sorted_indices, static_cast<int>(visibleCount));

    cudaGraphicsUnmapResources(1, &s.cudaIndexRes, 0);
  }

  // 6. Unmapping the splats SSBO from CUDA.
  cudaGraphicsUnmapResources(1, &s.cudaSplatRes, 0);

  s.d_splats_mapped = nullptr;

  return static_cast<int>(visibleCount);
}

void freeCudaSorter(CudaSorter &s) {
  if (s.cudaSplatRes)
    cudaGraphicsUnregisterResource(s.cudaSplatRes);

  if (s.cudaIndexRes)
    cudaGraphicsUnregisterResource(s.cudaIndexRes);

  if (s.d_visible_count)
    cudaFree(s.d_visible_count);

  if (s.d_keys_in)
    cudaFree(s.d_keys_in);

  if (s.d_keys_out)
    cudaFree(s.d_keys_out);

  if (s.d_vals_in)
    cudaFree(s.d_vals_in);

  if (s.d_temp_storage)
    cudaFree(s.d_temp_storage);

  s.d_splats_mapped = nullptr;
}
