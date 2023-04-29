#ifndef __COMMON_HEADER__
#define __COMMON_HEADER__

#ifdef DEBUG
#define print_if_debugging(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define print_if_debugging(fmt, ...)
#endif

// Header
#include <cassert>
#include <cinttypes>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <thrust/sort.h>
#include "helper_math.h"
#include "host_memory.h"
#include "device_memory.h"

// Macro
#define KS_11 8
#define LOOP_LEN 1024

#define A(i, j) A[(i) + (j)*M]
#define B(i, j) B[(i) + (j)*N]
#define ptr_A(i, j) ptr_A[(i) + (j)*M]
#define ptr_B(i, j) ptr_B[(i) + (j)*N]
#define C(i, j) C[(i) + (j)*M]
#define sa11(i, j) sa11[((j) << 7) + (i)]
#define sb11(i, j) sb11[((j) << 7) + (i)]
#define ptr_sa11(i, j) ptr_sa11[((j) << 7) + (i)]
#define ptr_sb11(i, j) ptr_sb11[((j) << 7) + (i)]

// Device function
__device__ inline void vload(float4 &v, float const *addr)
{
  v = *((float4 *)(addr));
}

__device__ inline void vstore(float const *addr, float4 v)
{
  *((float4 *)(addr)) = v;
}

__global__ __launch_bounds__(128) void set_weights(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    const float *__restrict__ normB,
    const uint32_t k)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < k)
  {
    weight[idx] = normA[idx] * normB[idx];
  }
}

__global__ __launch_bounds__(128) void update_weights(
    float *__restrict__ weight,
    const float *total_weight,
    const uint32_t c,
    const uint32_t k)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < k)
  {
    weight[idx] = (*total_weight) / (weight[idx] * ((float)c));
  }
}

#endif
