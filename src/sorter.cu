#include "sorter.h"
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <cuda_gl_interop.h>
#include <iostream>

// Kernel wyliczający głębokość w przestrzeni kamery (z_view) dla każdego splata
__global__ void computeDepthKeysKernel(const __half *__restrict__ splatsHalf,
                                       uint16_t *__restrict__ keysOut,
                                       uint32_t *__restrict__ valsOut,
                                       size_t numSplats, float r2x, float r2y,
                                       float r2z, float r2w) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= numSplats)
    return;

  // Każdy GaussianSplatFP16 ma 28 bajtów = 14 wartości typu __half (uint16_t)
  // Pierwsze 3 wartości to pos[0], pos[1], pos[2]
  size_t base = idx * 14;
  float x = __half2float(splatsHalf[base + 0]);
  float y = __half2float(splatsHalf[base + 1]);
  float z = __half2float(splatsHalf[base + 2]);

  // Głębokość w układzie kamery: z_view = dot(viewRow2, vec4(pos, 1.0))
  // Im dalej od kamery, tym bardziej ujemna wartość -> sortowanie rosnące daje
  // Back-to-Front!

  float z_view = r2x * x + r2y * y + r2z * z + r2w;

  float depth = -z_view;

  uint16_t key = 0xFFFF;
  if (depth > 0.1f) {
    // Mapujemy głębokość do zakresu [0, 65535] dla klucza
    uint16_t halfBits = __half_as_ushort(__float2half(depth));
    key = ~halfBits;
  }

  keysOut[idx] = key;
  valsOut[idx] = static_cast<uint32_t>(idx);
}

void initCudaSorter(CudaSorter &s, GLuint splatSSBO, GLuint indexSSBO,
                    size_t count) {
  s.numSplats = count;

  // 1. Rejestrujemy bufory OpenGL w przestrzeni adresowej CUDA
  cudaGraphicsGLRegisterBuffer(&s.cudaSplatRes, splatSSBO,
                               cudaGraphicsRegisterFlagsReadOnly);
  cudaGraphicsGLRegisterBuffer(&s.cudaIndexRes, indexSSBO,
                               cudaGraphicsRegisterFlagsWriteDiscard);

  // 2. Alokujemy tablice pomocnicze dla kluczy i indeksów wejściowych
  cudaMalloc(&s.d_keys_in, count * sizeof(uint16_t));
  cudaMalloc(&s.d_keys_out, count * sizeof(uint16_t));
  cudaMalloc(&s.d_vals_in, count * sizeof(uint32_t));

  // 3. Pytamy CUB, ile pamięci tymczasowej potrzebuje dla tej liczby splatów
  s.temp_storage_bytes = 0;
  uint32_t *dummy_vals_out = nullptr;
  cub::DeviceRadixSort::SortPairs(nullptr, s.temp_storage_bytes, s.d_keys_in,
                                  s.d_keys_out, s.d_vals_in, dummy_vals_out,
                                  static_cast<int>(count));

  cudaMalloc(&s.d_temp_storage, s.temp_storage_bytes);
  std::cout << "[CUDA] Sorter initialized. CUB temp buffer: "
            << (s.temp_storage_bytes / (1024.0 * 1024.0)) << " MiB"
            << std::endl;
}

void sortSplatsCuda(CudaSorter &s, const float *v) {
  if (s.numSplats == 0)
    return;

  // 1. Mapujemy oba zasoby OpenGL (zero kopiowania przez CPU!)
  cudaGraphicsResource *resources[2] = {s.cudaSplatRes, s.cudaIndexRes};
  cudaGraphicsMapResources(2, resources, 0);

  const __half *d_splats = nullptr;
  uint32_t *d_sorted_indices = nullptr;
  size_t num_bytes = 0;

  cudaGraphicsResourceGetMappedPointer((void **)&d_splats, &num_bytes,
                                       s.cudaSplatRes);
  cudaGraphicsResourceGetMappedPointer((void **)&d_sorted_indices, &num_bytes,
                                       s.cudaIndexRes);

  // 2. Wyciągamy 3. wiersz macierzy widoku (w GLM kolumnowym: indeksy 2, 6, 10,
  // 14)
  float r2x = v[2];
  float r2y = v[6];
  float r2z = v[10];
  float r2w = v[14];

  // 3. Odpalamy kernel liczący klucze głębokości
  int threadsPerBlock = 256;
  int blocksPerGrid =
      static_cast<int>((s.numSplats + threadsPerBlock - 1) / threadsPerBlock);

  computeDepthKeysKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_splats, s.d_keys_in, s.d_vals_in, s.numSplats, r2x, r2y, r2z, r2w);

  // 4. Sortujemy pary (klucz -> indeks) bezpośrednio do bufora indexSSBO!
  cub::DeviceRadixSort::SortPairs(
      s.d_temp_storage, s.temp_storage_bytes, s.d_keys_in, s.d_keys_out,
      s.d_vals_in, d_sorted_indices, static_cast<int>(s.numSplats));

  // 5. Odmapowujemy zasoby, oddając kontrolę potokowi OpenGL
  cudaGraphicsUnmapResources(2, resources, 0);
}

void freeCudaSorter(CudaSorter &s) {
  if (s.cudaSplatRes)
    cudaGraphicsUnregisterResource(s.cudaSplatRes);
  if (s.cudaIndexRes)
    cudaGraphicsUnregisterResource(s.cudaIndexRes);
  if (s.d_keys_in)
    cudaFree(s.d_keys_in);
  if (s.d_keys_out)
    cudaFree(s.d_keys_out);
  if (s.d_vals_in)
    cudaFree(s.d_vals_in);
  if (s.d_temp_storage)
    cudaFree(s.d_temp_storage);
}
