#include "common.h"
#include "sgemm_previousAMM.h"
//#include "sgemv_with_proposedAMM.h"

void test(
    host_memory &p,
    device_memory &dm)
{
  if (p.kernel == "previousAMM")
  {
    sgemm_previousAMM<DIM_M>(dm, true);
  }
  else if (p.kernel == "proposedAMM")
  {
    //sgemv_with_proposedAMM<DIM_M>(dm, true);
  }

  cublasSgemm(dm.handle,
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              p.m, p.n, p.k,
              &p.alpha,
              dm.dA, p.m,
              dm.dB, p.k,
              &p.beta,
              dm.dY_ref, p.m);

  cudaDeviceSynchronize();
  cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDeviceToHost);
  cudaMemcpy(p.hY_ref, dm.dY_ref, sizeof(float) * p.m * p.n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  printf("[info] elements obtained by MM and AMM\n");
  printf("my kernel,cuBLAS\n");
  for (int i = 0; i < p.m * p.n; i++)
  {
    printf("%f, %f\n", p.hY[i], p.hY_ref[i]);
  }
}

void execute_kernel(
    host_memory &p,
    device_memory &dm,
    const bool warmup = false)
{
  if (p.kernel == "exact" || warmup)
  {
    if (warmup)
    {
      cublasSgemm(dm.handle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  p.m, p.n, p.k,
                  &p.alpha,
                  dm.dA, p.m,
                  dm.dB, p.k,
                  &p.beta,
                  dm.dY_ref, p.m);

      cudaMemcpy(p.hY_ref, dm.dY_ref, sizeof(float) * p.m * p.n, cudaMemcpyDefault);

      // calculate Frobenius norm of AB
      cublasSnrm2(dm.handle, dm.m * p.n, dm.dY_ref, 1, &p.total_normAB);
    }
    else
    {
      // measure time
      p.start_timer();
      for (int i = 0; i < p.nreps; ++i)
      {
        cublasSgemm(dm.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    p.m, p.n, p.k,
                    &p.alpha,
                    dm.dA, p.m,
                    dm.dB, p.k,
                    &p.beta,
                    dm.dY_ref, p.m);
      }
      p.stop_timer();
    }
  }

  if (dm.kernel == "previousAMM")
  {
    if (warmup)
    {
      sgemm_previousAMM<DIM_M>(dm);
    }
    else
    {
      // get error of Frobenius norm only, not measure time
      p.error.clear();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemm_previousAMM<DIM_M>(dm);

        cudaDeviceSynchronize();
        cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
        double err = get_error(p.hY_ref, p.hY, p.m * p.n);
        p.error.push_back(err);

        if (p.sanity_check)
        {
          const float const_one = 1.0;
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m * p.n, &const_one, dm.dY, 1, dm.dY_avg, 1);
          }
        }
      }

      // measure time
      p.start_timer();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemm_previousAMM<DIM_M>(dm);
      }
      p.stop_timer();
    }
  }
  else if (p.kernel == "proposedAMM")
  {
    if (warmup)
    {
      //sgemv_with_proposedAMM<DIM_M>(dm);
    }
    else
    {
      // get error of Frobenius norm only, not measure time
      p.error.clear();
      for (int i = 0; i < p.nreps; ++i)
      {
        //sgemv_with_proposedAMM<DIM_M>(dm);

        cudaDeviceSynchronize();
        cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
        double err = get_error(p.hY_ref, p.hY, p.m * p.n);
        p.error.push_back(err);

        if (p.sanity_check)
        {
          const float const_one = 1.0;
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m * p.n, &const_one, dm.dY, 1, dm.dY_avg, 1);
          }
        }
      }
    }

    // measure time
    cudaDeviceSynchronize();
    p.start_timer();
    for (int i = 0; i < p.nreps; ++i)
    {
      //sgemv_with_proposedAMM<DIM_M>(dm);
    }
    cudaDeviceSynchronize();
    p.stop_timer();
  }

  if (p.sanity_check && !warmup)
  {
    cudaMemcpy(p.hY, dm.dY_avg, sizeof(float) * p.m * p.n, cudaMemcpyDefault);

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
  if (p.sanity_check && p.k == p.c)
  {
    test(p, dm);
    return 0;
  }

  curandSetPseudoRandomGeneratorSeed(dm.gen, 1234ULL);

  // warm-up
  execute_kernel(p, dm, true);
  cudaDeviceSynchronize();

  // measure time
  execute_kernel(p, dm);

  // display info
  p.print_result();

  return 0;
}