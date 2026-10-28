#include "sorter.h"
#include <algorithm>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <iostream>

__global__ void cullAndCompactKernel(
    const __half *__restrict__ splatsHalf, uint16_t *__restrict__ keysOut,
    uint32_t *__restrict__ valsOut, uint32_t *__restrict__ globalVisibleCount,
    size_t numSplats, float vm2x, float vm2y, float vm2z, float vm2w, float m0,
    float m1, float m2, float m3, float m4, float m5, float m6, float m7,
    float m8, float m9, float m10, float m11, float m12, float m13, float m14,
    float m15) {
  // Szybka agregacja w pamięci współdzielonej bloku (256 wątków)
  __shared__ uint32_t blockCount;
  __shared__ uint32_t blockBaseOffset;

  if (threadIdx.x == 0) {
    blockCount = 0;
  }
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

    // 1. Odległość od kamery (Near Plane Culling)
    float z_view = vm2x * x + vm2y * y + vm2z * z + vm2w;
    float depth = -z_view;

    if (depth > 0.1f) {
      // 2. Frustum Culling (czy splat mieści się na ekranie?)
      float clipX = m0 * x + m4 * y + m8 * z + m12;
      float clipY = m1 * x + m5 * y + m9 * z + m13;
      float clipW = m3 * x + m7 * y + m11 * z + m15;

      float limit = 1.2f * clipW;
      if (clipW > 0.1f && clipX >= -limit && clipX <= limit &&
          clipY >= -limit && clipY <= limit) {
        float op = __half2float(splatsHalf[base + 6]);

        // Próg rośnie z odległością, ale ma sufit na 0.0f (50% alfa),
        // więc z daleka NIGDY nie wytnie zielonego laminatu!
        float opThreshold = fminf(-2.0f + 0.1f * depth, 0.0f);

        if (op > opThreshold) {
          float s0 = __half2float(splatsHalf[base + 7]);
          float s1 = __half2float(splatsHalf[base + 8]);
          float s2 = __half2float(splatsHalf[base + 9]);

          float maxLogScale = fmaxf(s0, fmaxf(s1, s2));
          float worldRadius = __expf(maxLogScale) * 3.0f;
          float pixelSize = (worldRadius * 1080.0f) / clipW;

          // TWARDE ODCIĘCIE SUB-PIKSELI: wszystko poniżej 1.3 px wylatuje w
          // 100%!
          if (pixelSize >= 1.1f) {
            bool lodPass = true;

            // Dla splatów w przedziale 1.3 px - 3.5 px przerzedzamy je
            // stochastycznie
            if (pixelSize < 3.5f) {
              float ratio = (pixelSize - 1.3f) / (3.5f - 1.3f);
              float keepProb = ratio * ratio;

              uint32_t hash =
                  static_cast<uint32_t>(idx) * 747796405u + 2891336453u;
              hash = ((hash >> ((hash >> 28u) + 4u)) ^ hash) * 277803737u;
              float rand01 =
                  static_cast<float>(hash & 0xFFFFu) * (1.0f / 65536.0f);

              lodPass = (rand01 < keepProb);
            }

            if (lodPass) {
              keep = true;
              uint16_t halfBits = __half_as_ushort(__float2half(depth));
              myKey = ~halfBits;
              localOffset = atomicAdd(&blockCount, 1u);
            }
          }
        }
      }
    }
  }
  __syncthreads();

  // Jeden atomowy zapis na cały blok 256 wątków!
  if (threadIdx.x == 0 && blockCount > 0) {
    blockBaseOffset = atomicAdd(globalVisibleCount, blockCount);
  }
  __syncthreads();

  // Zapisujemy tylko widoczne splaty w zwartej tablicy (Stream Compaction)
  if (keep) {
    uint32_t finalIdx = blockBaseOffset + localOffset;
    keysOut[finalIdx] = myKey;
    valsOut[finalIdx] = static_cast<uint32_t>(idx);
  }
}

void initCudaSorter(CudaSorter &s, unsigned int splatSSBO,
                    unsigned int indexSSBO, size_t count) {
  s.numSplats = count;

  cudaGraphicsGLRegisterBuffer(&s.cudaSplatRes, splatSSBO,
                               cudaGraphicsRegisterFlagsReadOnly);
  cudaGraphicsGLRegisterBuffer(&s.cudaIndexRes, indexSSBO,
                               cudaGraphicsRegisterFlagsWriteDiscard);

  size_t num_bytes = 0;
  cudaGraphicsMapResources(1, &s.cudaSplatRes, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&s.d_splats_mapped, &num_bytes,
                                       s.cudaSplatRes);

  cudaMalloc(&s.d_visible_count, sizeof(uint32_t));
  cudaMalloc(&s.d_keys_in, count * sizeof(uint16_t));
  cudaMalloc(&s.d_keys_out, count * sizeof(uint16_t));
  cudaMalloc(&s.d_vals_in, count * sizeof(uint32_t));

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

int sortSplatsCuda(CudaSorter &s, const float *vm, const float *mvp) {
  if (s.numSplats == 0)
    return 0;

  // 1. Zerujemy licznik widocznych splatów
  cudaMemsetAsync(s.d_visible_count, 0, sizeof(uint32_t));

  // 2. Odpalamy Frustum + Sub-pixel Culling + Stream Compaction
  int threadsPerBlock = 256;
  int blocksPerGrid =
      static_cast<int>((s.numSplats + threadsPerBlock - 1) / threadsPerBlock);

  cullAndCompactKernel<<<blocksPerGrid, threadsPerBlock>>>(
      static_cast<const __half *>(s.d_splats_mapped), s.d_keys_in, s.d_vals_in,
      s.d_visible_count, s.numSplats, vm[2], vm[6], vm[10], vm[14], mvp[0],
      mvp[1], mvp[2], mvp[3], mvp[4], mvp[5], mvp[6], mvp[7], mvp[8], mvp[9],
      mvp[10], mvp[11], mvp[12], mvp[13], mvp[14], mvp[15]);

  // 3. Odczytujemy, ile splatów przetrwało selekcję
  uint32_t visibleCount = 0;
  cudaMemcpy(&visibleCount, s.d_visible_count, sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  if (visibleCount == 0)
    return 0;

  // 4. Sortujemy TYLKO widoczne splaty (visibleCount zamiast 15 milionów!)
  cudaGraphicsMapResources(1, &s.cudaIndexRes, 0);
  uint32_t *d_sorted_indices = nullptr;
  size_t num_bytes = 0;
  cudaGraphicsResourceGetMappedPointer((void **)&d_sorted_indices, &num_bytes,
                                       s.cudaIndexRes);

  cub::DeviceRadixSort::SortPairs(
      s.d_temp_storage, s.temp_storage_bytes, s.d_keys_in, s.d_keys_out,
      s.d_vals_in, d_sorted_indices, static_cast<int>(visibleCount));

  cudaGraphicsUnmapResources(1, &s.cudaIndexRes, 0);
  return static_cast<int>(visibleCount);
}

void freeCudaSorter(CudaSorter &s) {
  if (s.cudaSplatRes) {
    cudaGraphicsUnmapResources(1, &s.cudaSplatRes, 0);
    cudaGraphicsUnregisterResource(s.cudaSplatRes);
  }
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
}
