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
  uint64_t k = DIM_M;
  uint64_t n = DIM_M;
  uint32_t c = DIM_M;

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
  float *h_normA = nullptr;
  float *h_normB = nullptr;

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

    delete h_normA;
    delete h_normB;

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
      else if (!strcmp(argv[i], "-k"))
      {
        k = std::atoi(argv[++i]);
        k = DIV_CEIL(k, (uint64_t)DIM_M) * DIM_M;
      }
      else if (!strcmp(argv[i], "-n"))
      {
        n = std::atoi(argv[++i]);
        n = DIV_CEIL(n, (uint64_t)DIM_M) * DIM_M;
      }
      else if (!strcmp(argv[i], "-c"))
      {
        c = std::atoi(argv[++i]);
      }
      else if (!strcmp(argv[i], "-r"))
      {
        nreps = std::atoi(argv[++i]);
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
    else if (kernel != "previousAMM" && kernel != "proposedAMM")
    {
      printf("[error] '-kernel %s' is not valid.\n", kernel.c_str());
      return false;
    }

    assert(1 <= c);
    assert(c <= k);

    if (verbose >= 2)
    {
      printf("[info] Parameters are as follows.\n");
      printf("\tkernel: %s\n", kernel.c_str());
      printf("\tm     = %" PRIu64 "\n", m);
      printf("\tk     = %" PRIu64 "\n", k);
      printf("\tn     = %" PRIu64 "\n", n);
      printf("\tc     = %u\n", c);
      printf("\tnreps = %u\n", nreps);
      printf("\talpha = %f\n", alpha);
      printf("\tbeta  = %f\n", beta);
      printf("\tseed  = %d\n", seed);

      printf("[info] Sanity check : %s\n", sanity_check ? "on" : "off");
    }

    return true;
  }

  void allocate_memory()
  {
    hY = new float[m * n];
    hY_ref = new float[m * n];

    h_normA = new float[k];
    h_normB = new float[k];
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

  double get_theoretical_Frobenius_error() const
  {
    double sum = 0.0;
    for (int i = 0; i < k; i++)
    {
      sum += h_normA[i] * h_normB[i];
      //printf("%f, %f\n", h_normA[i], h_normB[i]);
    }
    //printf("sum = %f\n", sum);
    //printf("total_normAB = %f\n", total_normAB);

    return sqrt((sum * sum - total_normAB * total_normAB) / (double)c);
  }

  void print_result()
  {
    if (verbose >= 1)
    {
      // basic information
      printf("DIM_M,");
      printf("-m,");
      printf("-k,");
      printf("-n,");
      printf("-s,");
      printf("-r,");
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
      printf("theoretical error,");
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
    printf("%" PRIu64 ",", m);
    printf("%" PRIu64 ",", k);
    printf("%" PRIu64 ",", n);
    printf("%d,", c);
    printf("%d,", nreps);
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
    printf("%lf,", get_theoretical_Frobenius_error());
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
    else
    {
      printf(",-");
    }
    printf("\n");
  }

private:
  float elapsed_time;

  cudaEvent_t start;
  cudaEvent_t stop;
};

#endif
