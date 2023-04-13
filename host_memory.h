#ifndef __HOST_MEMORY__
#define __HOST_MEMORY__

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <cuda_runtime.h>

template <typename T>
inline T DIV_CEIL(const T x, const T y)
{
  return (x + y - 1) / y;
}

class host_memory
{
public:
  uint64_t m = DIM_M;
  uint64_t n = DIM_M;
  uint32_t nsamples = DIM_M;
  uint32_t slicesize = 32;

  std::string kernel = "exact";
  std::string matrixType_A = "gaussian_0.0_1.0";
  std::string matrixType_B = "gaussian_0.0_1.0";

  int verbose = 0;
  uint32_t nreps = 10;
  uint32_t seed;
  float alpha = 1.0f;
  float beta = 0.0f;

  float *hY = nullptr;
  float *hY_ref = nullptr;

  float total_normA = 0.0f;
  float total_normB = 0.0f;
  float total_normAB = 0.0f;

  bool sanity_check = false;
  bool show_all_errors = false;

  std::vector<float> error;

  host_memory()
  {
    seed = time(NULL);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~host_memory()
  {
    delete hY;
    delete hY_ref;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  bool parser(int argc, char **argv)
  {
    for (int i = 1; i < argc; ++i)
    {
      if (!strcmp(argv[i], "-m"))
      {
        m = std::atoi(argv[++i]);
        m = DIV_CEIL(m, (uint64_t)DIM_M) * DIM_M;
      }
      else if (!strcmp(argv[i], "-n"))
      {
        n = std::atoi(argv[++i]);
        n = DIV_CEIL(n, (uint64_t)DIM_M) * DIM_M;
      }
      else if (!strcmp(argv[i], "-s"))
      {
        nsamples = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-r"))
      {
        nreps = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-slicesize"))
      {
        slicesize = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-kernel"))
      {
        kernel = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-matrixType_A"))
      {
        matrixType_A = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-matrixType_B"))
      {
        matrixType_B = std::string(argv[++i]);
      }
      else if (!strcmp(argv[i], "-seed"))
      {
        seed = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-sanity"))
      {
        sanity_check = true;
      }
      else if (!strcmp(argv[i], "-alpha"))
      {
        alpha = std::atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-beta"))
      {
        beta = std::atof(argv[++i]);
      }
      else if (!strcmp(argv[i], "-verbose"))
      {
        verbose = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-show_all_errors"))
      {
        show_all_errors = true;
      }
      else
      {
        printf("[warning] unknown parameter: %s\n", argv[i]);
      }
    }

    if (kernel == "exact")
    {
      sanity_check = false;
    }
    else if (kernel != "basicAMM" && kernel != "slicedAMM")
    {
      printf("[error] '-kernel %s' is not valid.\n", kernel.c_str());
      return false;
    }

    bool check = false;
    const uint32_t ary_of_slicesize[] = {1, 2, 4};
    for (std::size_t i = 0; i < sizeof(ary_of_slicesize) / sizeof(uint32_t); ++i)
    {
      if (ary_of_slicesize[i] == slicesize)
      {
        check = true;
        break;
      }
    }
    if (!check)
    {
      printf("[error] '-slicesize %d' is not valid.\n", slicesize);
      return false;
    }

    nsamples = DIV_CEIL(nsamples, slicesize) * slicesize;

    assert(0 < nsamples);
    assert(nsamples <= n);

    if (verbose >= 2)
    {
      printf("[info] Parameters are as follows.\n");
      printf("\tkernel    : %s\n", kernel.c_str());
      printf("\tm         = %ld\n", m);
      printf("\tn         = %ld\n", n);
      printf("\tnsamples  = %u\n", nsamples);
      printf("\tslicesize = %u\n", slicesize);
      printf("\tnreps     = %u\n", nreps);
      printf("\talpha     = %f\n", alpha);
      printf("\tbeta      = %f\n", beta);
      printf("\tseed      = %d\n", seed);

      printf("[info] Sanity check : %s\n", sanity_check ? "on" : "off");
    }

    return true;
  }

  void allocate_memory()
  {
    hY = new float[m];
    hY_ref = new float[m];
  }

  void start_timer(bool reset = true)
  {
    if (reset)
    {
      elapsed_time = 0.0f;
    }
    cudaEventRecord(start);
  }

  void stop_timer()
  {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float t;
    cudaEventElapsedTime(&t, start, stop);
    elapsed_time += t;
  }

  float get_elapsed_millisecond_time() const
  {
    return elapsed_time;
  }

  void print_result()
  {
    if (verbose >= 1)
    {
      // basic information
      printf("DIM_M,");
      printf("-m,");
      printf("-n,");
      printf("-s,");
      printf("-r,");
      printf("-slicesize,");
      printf("-kernel,");
      printf("-matrixType_A,");
      printf("-matrixType_B,");
      printf("-seed,");
      printf("-alpha,");
      printf("-beta,");

      // result
      printf("Frobenius norm of A,");
      printf("Frobenius norm of B,");
      printf("Frobenius norm of AB,");
      printf("total time [ms],");
      printf("average time [ms],");
      printf("50%%-tile error,");
      printf("25%%-tile error,");
      printf("75%%-tile error,");
      printf("error norms\n");
    }

    if (kernel != "exact")
    {
      std::sort(error.begin(), error.end());
    }

    // basic information
    printf("%d,", DIM_M);
    printf("%ld,", m);
    printf("%ld,", n);
    printf("%d,", nsamples);
    printf("%d,", nreps);
    printf("%d,", slicesize);
    printf("%s,", kernel.c_str());
    printf("%s,", matrixType_A.c_str());
    printf("%s,", matrixType_B.c_str());
    printf("%d,", seed);
    printf("%f,", alpha);
    printf("%f,", beta);

    // result
    printf("%lf,", total_normA);
    printf("%lf,", total_normB);
    printf("%lf,", total_normAB);
    printf("%f,", get_elapsed_millisecond_time());
    printf("%f,", get_elapsed_millisecond_time() / nreps);
    printf("%f,", (kernel == "exact") ? 0.0 : error[int(0.50 * error.size())]);
    printf("%f,", (kernel == "exact") ? 0.0 : error[int(0.25 * error.size())]);
    printf("%f", (kernel == "exact") ? 0.0 : error[int(0.75 * error.size())]);

    if (show_all_errors && kernel != "exact")
    {
      for (int i = 0; i < nreps; ++i)
      {
        printf(",%f", error[i]);
      }
    }
    printf("\n");
  }

private:
  float elapsed_time;

  cudaEvent_t start;
  cudaEvent_t stop;
};

#endif
