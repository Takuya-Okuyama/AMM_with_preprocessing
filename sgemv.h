#ifndef __MY_SGEMV__
#define __MY_SGEMV__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <unistd.h>

#ifdef DEBUG
#define print_if_debugging(fmt, ...) printf(fmt, ##__VA_ARGS__);
#else
#define print_if_debugging(fmt, ...)
#endif

#define NUM_BLOCKY 2
#define NUM_UNROLL 32

__device__ __forceinline__ void
load_vector(float *__restrict__ addr1, const float *__restrict__ addr2)
{
  *((float4 *)(addr1)) = *((float4 *)(addr2));
}

__global__ __launch_bounds__(DIM_M) void kernel_without_beta(const int64_t m,
                                                             const int64_t n,
                                                             const int64_t c,
                                                             const float alpha,
                                                             const float *__restrict__ A,
                                                             const float *__restrict__ x,
                                                             float *__restrict__ y)
{
  const int64_t tx = threadIdx.x, bx = blockIdx.x, by = blockIdx.y;

  A = A + bx * DIM_M + by * n * m;
  y = y + bx * DIM_M;
  x = x + by * n;
  float resY = 0.;

  __shared__ float sx[NUM_UNROLL];

#pragma unroll
  for (int64_t j = 0; j < c; j += NUM_UNROLL)
  {
    __syncthreads();

#pragma unroll
    for (int64_t k = 4 * tx; k < NUM_UNROLL; k += 4 * DIM_M)
    {
      load_vector(&sx[k], &x[j + k]);
    }

    __syncthreads();

#pragma unroll
    for (int64_t i = 0; i < NUM_UNROLL; ++i)
    {
      resY = __fmaf_rn(A[tx + m * (j + i)], sx[i], resY);
    }
  }

  /* ============================== */
  __syncthreads();

#pragma unroll
  for (int64_t k = c + 4 * tx; k < n; k += 4 * DIM_M)
  {
    load_vector(&sx[k - c], &x[k]);
  }

  __syncthreads();

#pragma unroll
  for (int64_t j = c; j < n; j++)
  {
    resY = __fmaf_rn(A[tx + m * j], sx[j - c], resY);
  }

  atomicAdd(&y[tx], alpha * resY);
}

void sgemv(const int m,
           const int n,
           const float alpha,
           const float *A,
           const float *x,
           const float beta,
           float *y)
{
  assert(m % DIM_M == 0);
  assert(n % NUM_BLOCKY == 0);

  dim3 grid(m / DIM_M, NUM_BLOCKY);
  dim3 threads(DIM_M, 1);

  int sub_n = n / NUM_BLOCKY;
  const int c = sub_n - sub_n % NUM_UNROLL;

  //cudaFuncSetAttribute(&sgemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_UNROLL * sizeof(float));
  //cudaFuncSetCacheConfig(&sgemv_kernel, cudaFuncCachePreferL1);
  if (beta == 0.0f)
  {
    kernel_without_beta<<<grid, threads>>>(m, sub_n, c, alpha, A, x, y);
  }
}

#endif
