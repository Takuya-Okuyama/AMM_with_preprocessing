#ifndef __SGEMV_proposedAMM__
#define __SGEMV_proposedAMM__

#include "common.h"
#include "core_proposedAMM.cuh"
#include <thrust/sort.h>

template <uint32_t dim_M>
void sgemm_proposedAMM(
    device_memory &dm,
    const bool sanity_check = false)
{
  static bool is_init = true;
  static cudaGraph_t graph;
  static cudaGraphExec_t instance;

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
          dm.d_normAprime,
          dm.d_normBprime,
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

    cudaEventCreate(&dm.event_scale);
    cudaStreamBeginCapture(dm.stream_2, cudaStreamCaptureModeGlobal);

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

      cub::DeviceRadixSort::SortKeys(
          dm.d_tmp_sort,
          dm.storageBytes_sort,
          dm.d_pos,
          dm.d_sorted_pos,
          dm.c,
          0, sizeof(uint32_t) * 8,
          dm.stream_2);
    }

    // cudaFuncSetAttribute(&sgemv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, NUM_UNROLL * sizeof(float));
    // cudaFuncSetCacheConfig(&sgemv_kernel, cudaFuncCachePreferL1);

    dim3 gridDim(DIV_CEIL(dm.m, (uint64_t)128), DIV_CEIL(dm.n, (uint64_t)128));
    proposedAMM<<<gridDim, 256, 0, dm.stream_2>>>(
        dm.m, dm.n, dm.c,
        dm.dA, dm.dB, dm.dY,
        dm.d_sorted_pos,
        dm.d_weight,
        dm.d_alpha, dm.d_beta,
        dm.d_vr, dm.d_vc, dm.d_w);

    cudaStreamEndCapture(dm.stream_2, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
    is_init = false;
  }

  cudaGraphLaunch(instance, dm.stream_2);
}

#endif
