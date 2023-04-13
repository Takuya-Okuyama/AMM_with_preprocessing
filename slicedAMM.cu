#include "common.h"
#include "sgemv_with_basicAMM.h"
#include "sgemv_with_slicedAMM.h"

void test(
    host_memory &p,
    device_memory &dm)
{
  if (p.kernel == "basicAMM" || p.slicesize == 1)
  {
    sgemv_with_basicAMM<DIM_M>(dm, true);
  }
  else if (p.kernel == "slicedAMM")
  {
    sgemv_with_slicedAMM<DIM_M>(dm, true);
  }

  cublasSgemv(dm.handle,
              CUBLAS_OP_N,
              p.m, p.n,
              &p.alpha,
              dm.dA, p.m,
              dm.dB, 1,
              &p.beta,
              dm.dY_ref, 1);

  cudaDeviceSynchronize();
  cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m, cudaMemcpyDeviceToHost);
  cudaMemcpy(p.hY_ref, dm.dY_ref, sizeof(float) * p.m, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("[info] elements obtained by MM and AMM\n");
  printf("my kernel,cuBLAS\n");
  for (int i = 0; i < p.m; i++)
  {
    printf("%f, %f\n", p.hY[i], p.hY_ref[i]);
  }
}

void execute_selected_kernel(
    host_memory &p,
    device_memory &dm,
    const bool warmup = false)
{
  if (p.kernel == "exact" || warmup)
  {
    if (warmup)
    {
      cublasSgemv(dm.handle,
                  CUBLAS_OP_N,
                  p.m, p.n,
                  &p.alpha,
                  dm.dA, p.m,
                  dm.dB, 1,
                  &p.beta,
                  dm.dY_ref, 1);

      cudaMemcpy(p.hY_ref, dm.dY_ref, sizeof(float) * p.m, cudaMemcpyDefault);

      // calculate Frobenius norm of AB
      cublasSnrm2(dm.handle, dm.m, dm.dY_ref, 1, &p.total_normAB);
    }
    else
    {
      // measure time
      p.start_timer();
      for (int i = 0; i < p.nreps; ++i)
      {
        cublasSgemv(dm.handle,
                    CUBLAS_OP_N,
                    p.m, p.n,
                    &p.alpha,
                    dm.dA, p.m,
                    dm.dB, 1,
                    &p.beta,
                    dm.dY, 1);
      }
      p.stop_timer();
    }
  }

  if (dm.kernel == "basicAMM" || dm.slicesize == 1)
  {
    if (warmup)
    {
      sgemv_with_basicAMM<DIM_M>(dm);
    }
    else
    {
      // get error of Frobenius norm only, not measure time
      p.error.clear();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemv_with_basicAMM<DIM_M>(dm);

        cudaDeviceSynchronize();
        cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m, cudaMemcpyDefault);
        double err = get_error(p.hY_ref, p.hY, p.m);
        p.error.push_back(err);

        if (p.sanity_check)
        {
          const float const_one = 1.0;
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m, &const_one, dm.dY, 1, dm.dY_avg, 1);
          }
        }
      }

      // measure time
      p.start_timer();
      cudaDeviceSynchronize();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemv_with_basicAMM<DIM_M>(dm);
      }
      cudaDeviceSynchronize();
      p.stop_timer();
    }
  }
  else if (p.kernel == "slicedAMM")
  {
    if (warmup)
    {
      sgemv_with_slicedAMM<DIM_M>(dm);
    }
    else
    {
      // get error of Frobenius norm only, not measure time
      p.error.clear();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemv_with_slicedAMM<DIM_M>(dm);

        cudaDeviceSynchronize();
        cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m, cudaMemcpyDefault);
        double err = get_error(p.hY_ref, p.hY, p.m);
        p.error.push_back(err);

        if (p.sanity_check)
        {
          const float const_one = 1.0;
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m, &const_one, dm.dY, 1, dm.dY_avg, 1);
          }
        }
      }
    }

    // measure time
    cudaDeviceSynchronize();
    p.start_timer();
    for (int i = 0; i < p.nreps; ++i)
    {
      sgemv_with_slicedAMM<DIM_M>(dm);
    }
    cudaDeviceSynchronize();
    p.stop_timer();
  }

  if (p.sanity_check && !warmup)
  {
    cudaMemcpy(p.hY, dm.dY_avg, sizeof(float) * p.m, cudaMemcpyDefault);

    printf("[info] elements obtained by MM and AMM\n");
    printf("my kernel,cuBLAS\n");
    for (int i = 0; i < p.m; i++)
    {
      printf("%f, %f\n", p.hY[i] / p.nreps, p.hY_ref[i]);
    }
  }
}

int main(int argc, char *argv[])
{
  // set host memory
  host_memory p;
  assert(p.parser(argc, argv));

  p.allocate_memory();

  // set device memory
  device_memory dm(p);
  dm.write_data(p);
  dm.set_internal_randomness();

  // check validity of hand-written kernel
  if (p.sanity_check && p.n == p.nsamples)
  {
    test(p, dm);
    return 0;
  }

  curandSetPseudoRandomGeneratorSeed(dm.gen, 1234ULL);

  // warm-up
  execute_selected_kernel(p, dm, true);
  cudaDeviceSynchronize();

  // measure time
  execute_selected_kernel(p, dm);

  // display info
  p.print_result();

  return 0;
}