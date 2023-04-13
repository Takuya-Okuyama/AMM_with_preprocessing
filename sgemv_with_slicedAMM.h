#ifndef __SGEMV_WITH_SLICEDAMM__
#define __SGEMV_WITH_SLICEDAMM__

#include "common.h"

template <uint32_t dim_M,
          uint32_t slicesize,
          uint32_t num_unroll, // num_unroll >= slicesize
          uint32_t nblocks_y>
__global__ __launch_bounds__(dim_M) void kernel_slicedAMM_for_small_slicesize(
    const uint64_t m,
    const float alpha,
    const float *__restrict__ A,
    const float *__restrict__ x,
    float *__restrict__ y,
    const uint32_t *__restrict__ d_ary,
    const float *__restrict__ d_weight,
    const float *__restrict__ d_acc_weight,
    const uint32_t num_selected_slices)
{
  const uint32_t tx = threadIdx.x;
  const uint32_t by = blockIdx.y;
  const uint32_t slice_src = (by * num_selected_slices) / nblocks_y;
  const uint32_t slice_dst = ((by + 1) * num_selected_slices) / nblocks_y;

  const uint32_t offset = slice_dst - slice_src;
  const uint32_t c = offset - offset % (num_unroll / slicesize);

  __shared__ float sx[num_unroll];
  __shared__ uint64_t base_idx[num_unroll / slicesize];
  __shared__ float scale_weight[num_unroll / slicesize];
  __shared__ float sum_of_weight;

  A = A + blockIdx.x * dim_M + tx;
  y = y + blockIdx.x * dim_M + tx;
  d_ary = d_ary + slice_src;
  float resY = 0;

  if (tx == 0)
  {
    sum_of_weight = *d_acc_weight;
  }

#pragma unroll
  for (uint32_t j = 0; j < c; j += num_unroll / slicesize)
  {
    __syncthreads();

#pragma unroll
    for (uint32_t k = tx; k < num_unroll / slicesize; k += dim_M)
    {
      const uint32_t selected_pos = d_ary[j + k];
      scale_weight[k] = num_selected_slices * d_weight[selected_pos];
      base_idx[k] = selected_pos * slicesize;

      float *ptr_sx = sx + k * slicesize;
      const float *ptr_x = x + selected_pos * slicesize;

#pragma unroll
      for (uint32_t l = 0; l < slicesize; ++l)
      {
        *ptr_sx = *ptr_x;
        ++ptr_sx;
        ++ptr_x;
      }
    }

    __syncthreads();

    const float *ptr_sx = sx;

#pragma unroll
    for (uint32_t k = 0; k < num_unroll / slicesize; ++k)
    {
      float sub_resY = 0.0;
      const float *ptr_A = A + base_idx[k] * m;

#pragma unroll
      for (uint64_t l = 0; l < slicesize; ++l)
      {
        sub_resY += (*ptr_A) * (*ptr_sx);
        ptr_A += m;
        ++ptr_sx;
      }

      resY += (sub_resY * sum_of_weight) / scale_weight[k];
    }
  }

  /* ============================== */
  __syncthreads();

#pragma unroll
  for (uint32_t k = tx; k < offset - c; k += dim_M)
  {
    const uint32_t selected_pos = d_ary[c + k];
    scale_weight[k] = num_selected_slices * d_weight[selected_pos];
    base_idx[k] = selected_pos * slicesize;

    float *ptr_sx = sx + k * slicesize;
    const float *ptr_x = x + selected_pos * slicesize;

#pragma unroll
    for (uint32_t l = 0; l < slicesize; ++l)
    {
      *ptr_sx = *ptr_x;
      ++ptr_sx;
      ++ptr_x;
    }
  }

  __syncthreads();

  const float *ptr_sx = sx;

#pragma unroll
  for (uint32_t k = 0; k < offset - c; ++k)
  {
    float sub_resY = 0.0;
    const float *ptr_A = A + base_idx[k] * m;

#pragma unroll
    for (uint64_t l = 0; l < slicesize; ++l)
    {
      sub_resY += (*ptr_A) * (*ptr_sx);
      ptr_A += m;
      ++ptr_sx;
    }

    resY += (sub_resY * sum_of_weight) / scale_weight[k];
  }

  atomicAdd(y, alpha * resY);
}

template <uint32_t dim_M,
          uint32_t slicesize,
          uint32_t num_unroll, // num_unroll < slicesize
          uint32_t nblocks_y>
__global__ __launch_bounds__(dim_M) void kernel_slicedAMM_for_large_slicesize(
    const uint64_t m,
    const float alpha,
    const float *__restrict__ A,
    const float *__restrict__ x,
    float *__restrict__ y,
    const uint32_t *__restrict__ d_ary,
    const float *__restrict__ d_weight,
    const float *__restrict__ d_acc_weight,
    const uint32_t num_selected_slices)
{
  const uint32_t tx = threadIdx.x;
  const uint32_t by = blockIdx.y;
  const uint32_t slice_src = (by * num_selected_slices) / nblocks_y;
  const uint32_t slice_dst = ((by + 1) * num_selected_slices) / nblocks_y;

  __shared__ float sx[num_unroll];
  __shared__ uint64_t base_idx;
  __shared__ float scale_weight;
  __shared__ float sum_of_weight;

  A = A + blockIdx.x * dim_M + tx;
  y = y + blockIdx.x * dim_M + tx;
  float resY = 0;

  if (tx == 0)
  {
    sum_of_weight = *d_acc_weight;
  }

#pragma unroll
  for (uint32_t j = slice_src; j < slice_dst; ++j)
  {
    __syncthreads();

    if (tx == 0)
    {
      const uint32_t selected_pos = d_ary[j];
      scale_weight = num_selected_slices * d_weight[selected_pos];
      base_idx = selected_pos * slicesize;
    }

#pragma unroll
    for (uint32_t b = 0; b < slicesize; b += num_unroll)
    {
      __syncthreads();

#pragma unroll
      for (uint32_t k = tx; k < num_unroll; k += dim_M)
      {
        sx[k] = x[base_idx + k];
      }

      __syncthreads();

      float sub_resY = 0.0;
      const float *ptr_A = A + base_idx * m;

#pragma unroll
      for (uint32_t k = 0; k < num_unroll; ++k)
      {
        sub_resY += (*ptr_A) * sx[k];
        ptr_A += m;
      }

      resY += (sub_resY * sum_of_weight) / scale_weight;

      base_idx += num_unroll;
    }
  }

  atomicAdd(y, alpha * resY);
}

template <uint32_t slicesize>
__global__ __launch_bounds__(slicesize / 4) void set_weight_sliced(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  typedef cub::BlockReduce<float, slicesize / 4> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  register float ary[4];
  reinterpret_cast<float4 *>(ary)[0] = reinterpret_cast<float4 *>(x)[blockIdx.x * (slicesize / 4) + threadIdx.x];

  float local_sum = ary[0] * ary[0];
  local_sum += ary[1] * ary[1];
  local_sum += ary[2] * ary[2];
  local_sum += ary[3] * ary[3];

  float norm2 = BlockReduce(temp_storage).Sum(local_sum, slicesize / 4);

  if (threadIdx.x == 0)
  {
    weight[blockIdx.x] = normA[blockIdx.x] * sqrtf(norm2);
  }
}

__global__ __launch_bounds__(128) void set_weight_sliced_slicesize2(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nslices_total)
  {
    register float ary[2];
    reinterpret_cast<float2 *>(ary)[0] = reinterpret_cast<float2 *>(x)[idx];

    //weight[idx] = normA[idx] * sqrtf(ary[0] * ary[0] + ary[1] * ary[1]);
    weight[idx] = normA[idx] * normf(2, ary);
  }
}

__global__ __launch_bounds__(128) void set_weight_sliced_slicesize4(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nslices_total)
  {
    register float ary[4];
    reinterpret_cast<float4 *>(ary)[0] = reinterpret_cast<float4 *>(x)[idx];

    //const float s1 = ary[0] * ary[0] + ary[1] * ary[1];
    //const float s2 = ary[2] * ary[2] + ary[3] * ary[3];
    //weight[idx] = normA[idx] * norm4df(s1 + s2);
    weight[idx] = normA[idx] * norm4df(ary[0], ary[1], ary[2], ary[3]);
  }
}

__global__ __launch_bounds__(256) void set_weight_sliced_slicesize8(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nslices_total)
  {
    register float ary[8];
    reinterpret_cast<float4 *>(ary)[0] = reinterpret_cast<float4 *>(x)[2 * idx];
    reinterpret_cast<float4 *>(ary)[1] = reinterpret_cast<float4 *>(x)[2 * idx + 1];
    weight[idx] = normA[idx] * normf(8, ary);
  }
}

__global__ __launch_bounds__(256) void set_weight_sliced_slicesize16(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nslices_total)
  {
    register float ary[16];
    reinterpret_cast<float4 *>(ary)[0] = reinterpret_cast<float4 *>(x)[4 * idx];
    reinterpret_cast<float4 *>(ary)[1] = reinterpret_cast<float4 *>(x)[4 * idx + 1];
    reinterpret_cast<float4 *>(ary)[2] = reinterpret_cast<float4 *>(x)[4 * idx + 2];
    reinterpret_cast<float4 *>(ary)[3] = reinterpret_cast<float4 *>(x)[4 * idx + 3];
    weight[idx] = normA[idx] * normf(16, ary);
  }
}

template <uint32_t slicesize>
__global__ __launch_bounds__(256) void set_weight_sliced_slicesize_any(
    float *__restrict__ weight,
    const float *__restrict__ normA,
    float *__restrict__ x,
    const uint32_t nslices_total)
{
  const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < nslices_total)
  {
    register float ary[slicesize];
    const uint32_t offset = (slicesize / 4) * idx;

#pragma unroll (slicesize / 4)
    for (uint32_t i = 0; i < slicesize / 4; ++i)
    {
      reinterpret_cast<float4 *>(ary)[i] = reinterpret_cast<float4 *>(x)[offset + i];
    }

    weight[idx] = normA[idx] * normf(slicesize, ary);
  }
}

template <uint32_t dim_M>
void sgemv_with_slicedAMM(
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
      switch (dm.slicesize)
      {
      case 2:
        set_weight_sliced_slicesize2<<<DIV_CEIL(dm.nslices_total, (uint32_t)128), 128, 0, dm.stream_2>>>(
            dm.d_weight,
            dm.d_normA,
            dm.dB,
            dm.nslices_total);
        break;

      case 4:
        set_weight_sliced_slicesize4<<<DIV_CEIL(dm.nslices_total, (uint32_t)128), 128, 0, dm.stream_2>>>(
            dm.d_weight,
            dm.d_normA,
            dm.dB,
            dm.nslices_total);
        break;

      case 8:
        set_weight_sliced_slicesize8<<<DIV_CEIL(dm.nslices_total, (uint32_t)256), 256, 0, dm.stream_2>>>(
            dm.d_weight,
            dm.d_normA,
            dm.dB,
            dm.nslices_total);
        break;

      case 16:
        set_weight_sliced_slicesize_any<16>
            <<<DIV_CEIL(dm.nslices_total, (uint32_t)64), 128, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;

      case 32:
        set_weight_sliced_slicesize_any<32><<<DIV_CEIL(dm.nslices_total, (uint32_t)128), 128, 0, dm.stream_2>>>(
            dm.d_weight,
            dm.d_normA,
            dm.dB,
            dm.nslices_total);
        break;

      case 64:
        set_weight_sliced_slicesize_any<64><<<DIV_CEIL(dm.nslices_total, (uint32_t)128), 128, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;

      case 128:
        set_weight_sliced_slicesize_any<128><<<DIV_CEIL(dm.nslices_total, (uint32_t)128), 128, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;

      case 256:
        set_weight_sliced<256>
            <<<dm.nslices_total, 256 / 4, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;

      case 512:
        set_weight_sliced<512>
            <<<dm.nslices_total, 512 / 4, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;

      case 1024:
        set_weight_sliced<1024>
            <<<dm.nslices_total, 1024 / 4, 0, dm.stream_2>>>(
                dm.d_weight,
                dm.d_normA,
                dm.dB,
                dm.nslices_total);
        break;
      }
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

    switch (dm.slicesize)
    {
    case 2:
      kernel_slicedAMM_for_small_slicesize<32, 2, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 4:
      kernel_slicedAMM_for_small_slicesize<32, 4, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 8:
      kernel_slicedAMM_for_small_slicesize<32, 8, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 16:
      kernel_slicedAMM_for_small_slicesize<32, 16, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 32:
      kernel_slicedAMM_for_small_slicesize<32, 32, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 64:
      kernel_slicedAMM_for_large_slicesize<32, 64, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 128:
      kernel_slicedAMM_for_large_slicesize<32, 128, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 256:
      kernel_slicedAMM_for_large_slicesize<32, 256, 32, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 512:
      kernel_slicedAMM_for_large_slicesize<32, 512, 64, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;

    case 1024:
      kernel_slicedAMM_for_large_slicesize<32, 1024, 64, nblocks_y>
          <<<dim3(dm.m / 32, nblocks_y), 32, 0, dm.stream_2>>>(
              dm.m,
              dm.alpha,
              dm.dA,
              dm.dB,
              dm.dY,
              dm.d_pos,
              dm.d_weight,
              dm.d_acc_weight + dm.nslices_total - 1,
              dm.nslices_select);
      break;
    }

    cudaStreamEndCapture(dm.stream_2, &graph);
    cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
    graphCreated = true;
  }

  cudaGraphLaunch(instance, dm.stream_2);

  scale<<<dm.m / dim_M, dim_M, 0, dm.stream_1>>>(dm.dY, dm.beta);
}

#endif
