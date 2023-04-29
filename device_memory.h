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

__global__ __launch_bounds__(128) void get_norms(
    float *d_normA,
    float *d_normB,
    float *d_normAprime,
    float *d_normBprime,
    float *d_alpha,
    float *d_beta,
    const float *dA,
    const float *dB,
    const uint64_t m,
    const uint64_t n,
    const uint64_t k)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < k)
  {
    double sum = 0.0, sum_pow2 = 0.0;
    uint64_t pos = m * idx;
    for (int i = 0; i < m; i++)
    {
      double v = dA[pos];
      sum += v;
      sum_pow2 = fma(v, v, sum_pow2);
      pos++;
    }

    double mean = sum / (double)m;
    d_alpha[idx] = (float)mean;
    d_normA[idx] = (float)sqrt(sum_pow2);
    d_normAprime[idx] = (float)sqrt(sum_pow2 - mean * mean * (double)m);

    sum = 0.0;
    sum_pow2 = 0.0;
    pos = n * idx;
    for (int i = 0; i < n; i++)
    {
      double v = dB[pos];
      sum += v;
      sum_pow2 = fma(v, v, sum_pow2);
      pos++;
    }

    mean = sum / (double)n;
    d_beta[idx] = (float)mean;
    d_normB[idx] = (float)sqrt(sum_pow2);
    d_normBprime[idx] = (float)sqrt(sum_pow2 - mean * mean * (double)n);
  }
}
__global__ __launch_bounds__(128) void update_matrices(
    float *dA,
    float *dB,
    const float *d_alpha,
    const float *d_beta,
    const uint64_t m,
    const uint64_t n,
    const uint64_t k)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < k)
  {
    uint64_t pos = m * idx;
    float val = d_alpha[idx];
    for (int i = 0; i < m; i++)
    {
      dA[pos] -= val;
      pos++;
    }

    pos = n * idx;
    val = d_beta[idx];
    for (int i = 0; i < n; i++)
    {
      dB[pos] -= val;
      pos++;
    }
  }
}

struct device_memory
{
  float *dA = nullptr;
  float *dB = nullptr;
  float *dY = nullptr;
  float *dY_ref = nullptr;
  float *dY_avg = nullptr;

  float *d_normA = nullptr;
  float *d_normB = nullptr;
  float *d_normAprime = nullptr;
  float *d_normBprime = nullptr;

  float *d_alpha = nullptr;
  float *d_beta = nullptr;
  float *d_vc = nullptr;
  float *d_vr = nullptr;
  float *d_w = nullptr;

  float *d_weight = nullptr;
  float *d_acc_weight = nullptr;
  void *d_tmp_inclusiveSum = nullptr;
  void *d_tmp_sort = nullptr;

  std::size_t storageBytes_inclusiveSum = 0;
  std::size_t storageBytes_sum = 0;
  std::size_t storageBytes_sort = 0;

  uint32_t *d_rnd = nullptr;
  uint32_t *d_pos = nullptr;
  uint32_t *d_sorted_pos = nullptr;

  cublasHandle_t handle;
  curandGenerator_t gen;
  cudaStream_t stream_1;
  cudaStream_t stream_2;
  cudaEvent_t event_scale;

  uint64_t m = 0;
  uint64_t k = 0;
  uint64_t n = 0;
  uint64_t c = 1;
  float const_one = 1.0f;
  float const_zero = 0.0f;
  std::string kernel;

  device_memory(host_memory &p)
  {
    m = p.m;
    k = p.k;
    n = p.n;
    c = p.c;
    seed = p.seed;
    kernel = p.kernel;

    cudaMalloc((void **)&dA, sizeof(float) * m * k);
    cudaMalloc((void **)&dB, sizeof(float) * k * n);
    cudaMalloc((void **)&dY, sizeof(float) * m * n);
    cudaMalloc((void **)&dY_ref, sizeof(float) * m * n);
    cudaMalloc((void **)&dY_avg, sizeof(float) * m * n);

    cudaMalloc((void **)&d_normA, sizeof(float) * k);
    cudaMalloc((void **)&d_normB, sizeof(float) * k);
    cudaMalloc((void **)&d_normAprime, sizeof(float) * k);
    cudaMalloc((void **)&d_normBprime, sizeof(float) * k);

    cudaMalloc((void **)&d_alpha, sizeof(float) * k);
    cudaMalloc((void **)&d_beta, sizeof(float) * k);
    cudaMalloc((void **)&d_vc, sizeof(float) * k);
    cudaMalloc((void **)&d_vr, sizeof(float) * k);
    cudaMalloc((void **)&d_w, sizeof(float));
    cudaMalloc((void **)&d_weight, sizeof(float) * k);
    cudaMalloc((void **)&d_acc_weight, sizeof(float) * k);

    cudaMalloc((void **)&d_rnd, sizeof(uint32_t) * c);
    cudaMalloc((void **)&d_pos, sizeof(uint32_t) * c);
    cudaMalloc((void **)&d_sorted_pos, sizeof(uint32_t) * c);

    cublasCreate(&handle);
    // cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_ALLOWED);
    // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);

    cudaStreamCreate(&stream_1);
    cudaStreamCreate(&stream_2);

    // Determine temporary device storage requirements
    cub::DeviceScan::InclusiveSum(
        d_tmp_inclusiveSum,
        storageBytes_inclusiveSum,
        d_weight,
        d_acc_weight,
        k,
        stream_2);

    cub::DeviceRadixSort::SortKeys(
        d_tmp_sort,
        storageBytes_sort,
        d_pos,
        d_sorted_pos,
        c,
        0, sizeof(uint32_t) * 8,
        stream_2);

    // Allocate temporary storage
    cudaMalloc(&d_tmp_inclusiveSum, storageBytes_inclusiveSum);
    cudaMalloc(&d_tmp_sort, storageBytes_sort);
  }

  ~device_memory()
  {
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dY);
    cudaFree(dY_ref);
    cudaFree(dY_avg);

    cudaFree(d_normA);
    cudaFree(d_normB);
    cudaFree(d_normAprime);
    cudaFree(d_normBprime);

    cudaFree(d_alpha);
    cudaFree(d_beta);
    cudaFree(d_vc);
    cudaFree(d_vr);
    cudaFree(d_w);

    cudaFree(d_weight);
    cudaFree(d_acc_weight);

    cudaFree(d_rnd);
    cudaFree(d_pos);
    cudaFree(d_sorted_pos);

    cudaFree(d_tmp_inclusiveSum);
    cudaFree(d_tmp_sort);

    curandDestroyGenerator(gen);
    cudaStreamDestroy(stream_1);
    cudaStreamDestroy(stream_2);
    // cudaEventDestroy(event_scale);
  }

  void write_data(host_memory &p)
  {
    curandSetPseudoRandomGeneratorSeed(gen, seed);

    p.FrobeniusNorm_A = generate_matrix(dA, m, k, p.matrixType_A);
    p.FrobeniusNorm_B = generate_matrix(dB, n, k, p.matrixType_B);

    if (p.verbose >= 2)
    {
      printf("[info] Frobenius Norm of A = %lf\n", p.FrobeniusNorm_A);
      printf("[info] Frobenius Norm of B = %lf\n", p.FrobeniusNorm_B);
    }

    // zero clear
    const float const_zero = 0.0;
    cublasSscal(handle, m * n, &const_zero, dY, 1);
    cublasSscal(handle, m * n, &const_zero, dY_ref, 1);

    // calculate norms
    get_norms<<<DIV_CEIL(k, (uint64_t)128), 128>>>(
        d_normA,
        d_normB,
        d_normAprime,
        d_normBprime,
        d_alpha, d_beta,
        dA, dB,
        m, n, k);

    cudaMemcpy(p.h_normA, d_normA, sizeof(float) * k, cudaMemcpyDefault);
    cudaMemcpy(p.h_normB, d_normB, sizeof(float) * k, cudaMemcpyDefault);
    cudaMemcpy(p.h_normAprime, d_normAprime, sizeof(float) * k, cudaMemcpyDefault);
    cudaMemcpy(p.h_normBprime, d_normBprime, sizeof(float) * k, cudaMemcpyDefault);
  }

  void set_internal_randomness()
  {
    curandSetPseudoRandomGeneratorSeed(gen, seed + 1234);
    curandGenerate(gen, d_rnd, c);
  }

private:
  uint32_t seed;

  float generate_matrix(float *d_ptr,
                        const uint64_t size_v,
                        const uint64_t size_h,
                        const std::string &type) const
  {
    std::string matrixType;
    float param1, param2;
    extract_parames(type, matrixType, param1, param2);

    if (matrixType == "gaussian")
    {
      curandGenerateNormal(gen, d_ptr, size_v * size_h, param1, param2);
    }
    else if (matrixType == "lognormal")
    {
      curandGenerateLogNormal(gen, d_ptr, size_v * size_h, param1, param2);
    }

    float norm = 0.0f;
    cublasSnrm2(handle, size_v * size_h, d_ptr, 1, &norm);
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
    uint32_t *__restrict__ d_pos,
    uint32_t *__restrict__ d_rnd,
    const float *__restrict__ d_acc_weight,
    const uint64_t ntotal,
    const uint64_t nselect)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= nselect)
  {
    return;
  }

  int bound_idx[2] = {-1, (int)ntotal - 1};
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
    const uint64_t n)
{
  uint64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n)
  {
    d_pos[idx] = idx;
  }
}

#endif
