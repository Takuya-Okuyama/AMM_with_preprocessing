#ifndef __SGEMV_WITH_BASICAMM__
#define __SGEMV_WITH_BASICAMM__

#include "common.h"

#define NUM_UNROLL 32
#define NUM_THREAD_BASIC 128

template <uint32_t dim_M,
          uint32_t nblocks_y>
__global__ __launch_bounds__(dim_M) void kernel_basicAMM(
    const uint64_t m,
    const float alpha,
    const float *__restrict__ A,
    const float *__restrict__ x,
    float *__restrict__ y,
    const uint32_t *__restrict__ d_ary,
    const float *__restrict__ d_weight,
    const float *__restrict__ d_acc_weight,
    const uint32_t nsamples)
{
  const uint32_t tx = threadIdx.x;
  const uint64_t by = blockIdx.y;
  const uint32_t offset = nsamples / nblocks_y;
  const uint32_t c = offset - offset % NUM_UNROLL;

  __shared__ float sx[NUM_UNROLL];
  __shared__ uint32_t idx[NUM_UNROLL];
  __shared__ float sum_of_weight;

  A = A + blockIdx.x * dim_M + tx;
  y = y + blockIdx.x * dim_M + tx;
  float resY = 0;

  if (tx == 0)
  {
    sum_of_weight = *d_acc_weight;
  }

#pragma unroll
  for (uint32_t j = 0; j < c; j += NUM_UNROLL)
  {
    __syncthreads();

#pragma unroll
    for (uint32_t k = tx; k < NUM_UNROLL; k += dim_M)
    {
      const uint32_t pos = d_ary[offset * by + j + k];
      idx[k] = pos;

      float probability = d_weight[pos] / sum_of_weight;
      sx[k] = x[pos] / (nsamples * probability);
    }

    __syncthreads();

#pragma unroll
    for (uint32_t i = 0; i < NUM_UNROLL; ++i)
    {
      resY += A[idx[i] * m] * sx[i];
    }
  }

  /* ============================== */
  __syncthreads();

#pragma unroll
  for (uint32_t k = tx; k < offset - c; k += dim_M)
  {
    const uint32_t pos = d_ary[offset * by + c + k];
    idx[k] = pos;

    float probability = d_weight[pos] / sum_of_weight;
    sx[k] = x[pos] / (nsamples * probability);
  }

  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < offset - c; ++i)
  {
    resY += A[idx[i] * m] * sx[i];
  }

  atomicAdd(y, alpha * resY);
}

__global__ __launch_bounds__(NUM_THREAD_BASIC) void set_weight_basic(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    const float *__restrict__ x,
    const uint32_t n)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    weight[idx] = normA[idx] * abs(x[idx]);
  }
}

template <uint32_t dim_M>
void sgemv_with_basicAMM(
    device_memory &dm,
    const bool sanity_check = false)
{
  static bool graphCreated = false;
  static cudaGraph_t graph;
  static cudaGraphExec_t instance;

  if (!graphCreated)
  {
    cudaEventCreate(&dm.event_scale);
    cudaStreamBeginCapture(dm.stream_2, cudaStreamCaptureModeGlobal);

    /*------------------------------------------------------------
     * Set weight
     *------------------------------------------------------------*/
    if (sanity_check && dm.nslices_select == dm.nslices_total)
    {
      set_value_for_sanity_check<<<DIV_CEIL(dm.nslices_total, dim_M), dim_M, 0, dm.stream_2>>>(
          dm.d_weight,
          dm.nslices_total);
    }
    else
    {
      set_weight_basic<<<DIV_CEIL(dm.nslices_total, (uint32_t)NUM_THREAD_BASIC), NUM_THREAD_BASIC, 0, dm.stream_2>>>(
          dm.d_weight,
          dm.d_normA,
          dm.dB,
          dm.n);
    }

    /*------------------------------------------------------------
     * Select columns / rows
     *------------------------------------------------------------*/

    // Run inclusive prefix sum
    cub::DeviceScan::InclusiveSum(dm.d_tmp_inclusiveSum,
                                  dm.storageBytes_inclusiveSum,
                                  dm.d_weight,
                                  dm.d_acc_weight,
                                  dm.nslices_total,
                                  dm.stream_2);

    if (sanity_check && dm.nslices_select == dm.nslices_total)
    {
      set_sequential<<<DIV_CEIL(dm.nslices_total, (uint32_t)32), 32, 0, dm.stream_2>>>(dm.d_pos, dm.nslices_total);
    }
    else
    {
      pick_index<<<DIV_CEIL(dm.nslices_select, (uint32_t)DIM_M), DIM_M, 0, dm.stream_2>>>(
          dm.d_pos,
          dm.d_rnd,
          dm.d_acc_weight,
          dm.nslices_total,
          dm.nslices_select);
    }

    //cudaFuncSetAttribute(&sgemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_UNROLL * sizeof(float));
    //cudaFuncSetCacheConfig(&sgemv_kernel, cudaFuncCachePreferL1);

    cudaStreamWaitEvent(dm.stream_2, dm.event_scale);

    constexpr uint32_t nblocks_y = 16;

    kernel_basicAMM<dim_M, nblocks_y>
        <<<dim3(dm.m / dim_M, nblocks_y), dim_M, 0, dm.stream_2>>>(
            dm.m,
            dm.alpha,
            dm.dA,
            dm.dB,
            dm.dY,
            dm.d_pos,
            dm.d_weight,
            dm.d_acc_weight + dm.nslices_total - 1,
            dm.nslices_select);

    cudaStreamEndCapture(dm.stream_2, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated = true;
  }

  cudaGraphLaunch(instance, dm.stream_2);

  scale<<<dm.m / dim_M, dim_M, 0, dm.stream_1>>>(dm.dY, dm.beta);
}

#endif
