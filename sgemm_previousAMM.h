#ifndef __SGEMV_WITH_previousAMM__
#define __SGEMV_WITH_previousAMM__

#include "common.h"
#include "sgemm.h"
#include <thrust/sort.h>

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
    weight[idx] = (*total_weight) / (weight[idx] * ((float) c));
  }
}

template <uint32_t dim_M>
void sgemm_previousAMM(
    device_memory &dm,
    const bool sanity_check = false)
{
  static bool is_init = true;

  if (is_init)
  {
    /*------------------------------------------------------------
     * Set weight
     *------------------------------------------------------------*/
    if (sanity_check && dm.k == dm.c)
    {
      set_value_for_sanity_check<<<DIV_CEIL(dm.k, (uint64_t)dim_M), dim_M, 0, dm.stream_2>>>(
          dm.d_weight,
          dm.k);
    }
    else
    {
      set_weights<<<DIV_CEIL(dm.k, (uint64_t)128), 128, 0, dm.stream_2>>>(
          dm.d_weight,
          dm.d_normA,
          dm.d_normB,
          dm.k);
    }

    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(
        dm.d_tmp_inclusiveSum,
        dm.storageBytes_inclusiveSum,
        dm.d_weight,
        dm.d_acc_weight,
        dm.k,
        dm.stream_2);

    /*------------------------------------------------------------
     * Update weight
     *------------------------------------------------------------*/
    update_weights<<<DIV_CEIL(dm.k, (uint64_t)128), 128, 0, dm.stream_2>>>(
        dm.d_weight,
        dm.d_acc_weight + dm.k - 1,
        dm.c,
        dm.k);

    is_init = false;
    thrust::cuda::par.on(dm.stream_2);
  }

  /*------------------------------------------------------------
   * Select columns / rows
   *------------------------------------------------------------*/
  if (sanity_check && dm.k == dm.c)
  {
    set_sequential<<<DIV_CEIL(dm.k, (uint64_t)32), 32, 0, dm.stream_2>>>(
        dm.d_pos,
        dm.k);
  }
  else
  {
    pick_index<<<DIV_CEIL(dm.c, (uint64_t)DIM_M), DIM_M, 0, dm.stream_2>>>(
        dm.d_pos,
        dm.d_rnd,
        dm.d_acc_weight,
        dm.k,
        dm.c);

    thrust::sort(thrust::device, dm.d_pos, dm.d_pos + dm.c);
  }

  // cudaFuncSetAttribute(&sgemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_UNROLL * sizeof(float));
  // cudaFuncSetCacheConfig(&sgemv_kernel, cudaFuncCachePreferL1);

  dim3 gridDim(DIV_CEIL(dm.m, (uint64_t)128), DIV_CEIL(dm.n, (uint64_t)128));
  mysgemm<<<gridDim, 256, 0, dm.stream_2>>>(
      dm.m, dm.n, dm.k, dm.c,
      dm.alpha, dm.dA, dm.dB, dm.beta, dm.dY,
      dm.d_pos,
      dm.d_weight);
}

#endif
