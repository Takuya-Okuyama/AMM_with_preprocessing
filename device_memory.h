#ifndef __DEVICE_MEMORY__
#define __DEVICE_MEMORY__

#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cub/cub.cuh>

#include <vector>
#include <algorithm>
#include <random>
#include <sstream>

struct Item
{
  uint32_t ID[8];
  uint32_t rnd;
  float norm;
  float weight;

  __host__ __device__ Item operator+(const Item rhs) const
  {
    uint32_t y = rnd;

    Item tc;
    //tc.ID = (total_weight / rhs.total_weight < y / RAND_MAX) ? this->ID : rhs.ID;

    const auto threshold = weight * (float)UINT32_MAX;

#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
      y ^= (y << 13);
      y ^= (y >> 17);
      y ^= (y << 5);

      tc.ID[i] = (threshold < (float)y * rhs.weight) ? this->ID[i] : rhs.ID[i];
    }

    tc.rnd = y;
    tc.weight = this->weight + rhs.weight;
    return tc;
  }

  void set(
      const uint32_t _ID,
      const uint32_t _rnd,
      const float _norm)
  {
    for (int i = 0; i < 8; ++i)
    {
      ID[i] = _ID;
    }

    rnd = _rnd;
    norm = _norm;
  }

  __host__ __device__ Item()
  {
    for (int i = 0; i < 8; ++i)
    {
      ID[i] = 2;
    }

    rnd = 2;
    norm = 2;
    weight = 2;
  }
};

struct device_memory
{
  float *dA = nullptr;
  float *dB = nullptr;
  float *dY = nullptr;
  float *dY_ref = nullptr;
  float *dY_avg = nullptr;

  float *d_normA = nullptr;

  float *d_weight = nullptr;
  float *d_acc_weight = nullptr;
  void *d_tmp_inclusiveSum = NULL;

  Item *d_item = nullptr;
  Item *d_item_result = nullptr;
  void *d_tmp_sum = NULL;

  std::size_t storageBytes_inclusiveSum = 0;
  std::size_t storageBytes_sum = 0;

  uint32_t *d_rnd = nullptr;
  uint32_t *d_pos = nullptr;

  cublasHandle_t handle;
  curandGenerator_t gen;
  cudaStream_t stream_1;
  cudaStream_t stream_2;
  cudaEvent_t event_scale;

  uint64_t m = 0;
  uint64_t n = 0;
  float alpha = 0.0;
  float beta = 0.0;
  std::string kernel;

  uint32_t nslices_total = 0;
  uint32_t nslices_select = 0;
  uint32_t slicesize = 0;

  device_memory(host_memory &p)
  {
    m = p.m;
    n = p.n;
    seed = p.seed;

    alpha = p.alpha;
    beta = p.beta;
    kernel = p.kernel;
    slicesize = p.slicesize;

    nslices_total = n;
    nslices_select = p.nsamples;

    if (kernel == "slicedAMM")
    {
      nslices_select /= p.slicesize;
    }

    cudaMalloc((void **)&dA, sizeof(float) * m * n);
    cudaMalloc((void **)&dB, sizeof(float) * n);
    cudaMalloc((void **)&dY, sizeof(float) * m);
    cudaMalloc((void **)&dY_ref, sizeof(float) * m);
    cudaMalloc((void **)&dY_avg, sizeof(float) * m);

    cudaMalloc((void **)&d_normA, sizeof(float) * nslices_total);
    cudaMalloc((void **)&d_weight, sizeof(float) * nslices_total);
    cudaMalloc((void **)&d_acc_weight, sizeof(float) * nslices_total);

    cudaMalloc((void **)&d_item, sizeof(Item) * nslices_total);
    cudaMalloc((void **)&d_item_result, sizeof(Item));

    cudaMalloc((void **)&d_rnd, sizeof(uint32_t) * nslices_select);
    cudaMalloc((void **)&d_pos, sizeof(uint32_t) * nslices_select);

    cublasCreate(&handle);
    cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);
    //cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(d_tmp_inclusiveSum,
                                  storageBytes_inclusiveSum,
                                  d_weight,
                                  d_acc_weight,
                                  nslices_total,
                                  stream_2);

    cub::DeviceReduce::Sum(d_tmp_sum,
                           storageBytes_sum,
                           d_item,
                           d_item_result,
                           nslices_total);

    // Allocate temporary storage
    cudaMalloc(&d_tmp_inclusiveSum, storageBytes_inclusiveSum);
    cudaMalloc(&d_tmp_sum, storageBytes_sum);
  }

  ~device_memory()
  {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dY);
    cudaFree(dY_ref);
    cudaFree(dY_avg);

    cudaFree(d_normA);
    cudaFree(d_weight);
    cudaFree(d_acc_weight);

    cudaFree(d_item);
    cudaFree(d_item_result);

    cudaFree(d_rnd);
    cudaFree(d_pos);

    cudaFree(d_tmp_inclusiveSum);
    cudaFree(d_tmp_sum);

    curandDestroyGenerator(gen);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    //cudaEventDestroy(event_scale);
  }

  void write_data(host_memory &p)
  {
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    p.total_normA = generate_matrix(dA, m, n, p.matrixType_A);
    p.total_normB = generate_matrix(dB, n, 1, p.matrixType_B);

    const float const_zero = 0.0;
    cublasSscal(handle, m, &const_zero, dY, 1);
    cublasSscal(handle, m, &const_zero, dY_ref, 1);

    // calculate norm of A
    float *h_normA = new float[nslices_total];
    float *ptr_A = dA;
    const std::size_t size = m * (n / nslices_total);
    for (uint32_t i = 0; i < nslices_total; ++i)
    {
      cublasSnrm2(handle, size, ptr_A, 1, h_normA + i);
      ptr_A += size;
    }

    cudaMemcpy(d_normA, h_normA, sizeof(float) * nslices_total, cudaMemcpyDefault);
    delete h_normA;
  }

  /*float get_error() const
  {
    float ret = 0.0f;
    cublasSnrm2(handle, m, dY, int incx, float  *result)
  }*/

  void set_internal_randomness()
  {
    curandSetPseudoRandomGeneratorSeed(gen, seed + 1234);
    curandGenerate(gen, d_rnd, nslices_select);
  }

private:
  uint32_t seed;

  float generate_matrix(float *d_ptr,
                        const uint64_t m,
                        const uint64_t n,
                        const std::string &type) const
  {
    std::string matrixType;
    float param1, param2;
    extract_parames(type, matrixType, param1, param2);

    if (matrixType == "gaussian")
    {
      curandGenerateNormal(gen, d_ptr, m * n, param1, param2);
    }
    else if (matrixType == "lognormal")
    {
      curandGenerateLogNormal(gen, d_ptr, m * n, param1, param2);
    }

    float norm = 0.0f;
    cublasSnrm2(handle, m * n, d_ptr, 1, &norm);
    return norm;
  }

  std::vector<std::string> split(const std::string &str, char sep) const
  {
    std::vector<std::string> v;
    std::stringstream ss(str);
    std::string buffer;
    while (std::getline(ss, buffer, sep))
    {
      v.push_back(buffer);
    }
    return v;
  }

  void extract_parames(
      const std::string str,
      std::string &matrixType,
      float &param1,
      float &param2) const
  {
    auto ary = split(str, '_');
    matrixType = ary[0];
    param1 = (ary.size() <= 1) ? 0.0 : std::atof(ary[1].c_str());
    param2 = (ary.size() <= 2) ? 0.0 : std::atof(ary[2].c_str());
  }
};

__global__ __launch_bounds__(32) void scale(
    float *x,
    const float beta)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  x[idx] *= beta;
}

__global__ void set_value_for_sanity_check(
    float *d_w,
    const uint32_t len)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < len)
  {
    d_w[idx] = 1.0;
  }
}

float get_error(
    const float *h1,
    const float *h2,
    const uint64_t len)
{
  double ret = 0.0;
  for (uint64_t i = 0; i < len; ++i)
  {
    //printf("%f, %f\n", h1[i], h2[i]);
    double v = (double)h1[i] - (double)h2[i];
    ret += v * v;
  }

  return std::sqrt(ret);
}

__device__ __forceinline__ float get_rand(uint32_t *x)
{
  uint32_t y = *x;
  y ^= (y << 13);
  y ^= (y >> 17);
  y ^= (y << 5);
  *x = y;
  return __int_as_float((y & 0x007FFFFF) | 0x3f800000) - 1.0f;
}

__global__ __launch_bounds__(DIM_M) void pick_index(
    uint32_t *d_pos,
    uint32_t *__restrict__ d_rnd,
    const float *__restrict__ d_acc_weight,
    const uint32_t nslices_total,
    const uint32_t nslices_select)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nslices_select)
  {
    return;
  }

  int bound_idx[2] = {-1, (int)nslices_total - 1};
  float bound_val[2] = {0.0f, d_acc_weight[bound_idx[1]]};
  const float target_val = bound_val[1] * get_rand(&d_rnd[idx]);

  while (bound_idx[0] + 1 != bound_idx[1])
  {
    const int middle_idx = (bound_idx[0] + bound_idx[1]) / 2;
    const float middle_val = d_acc_weight[middle_idx];

    int offset = (middle_val <= target_val) ? 0 : 1;
    bound_idx[offset] = middle_idx;
    bound_val[offset] = middle_val;
  }

  d_pos[idx] = bound_idx[1];
}

__global__ __launch_bounds__(32) void set_sequential(
    uint32_t *d_pos,
    const uint32_t n)
{
  uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    d_pos[idx] = idx;
  }
}

#endif
