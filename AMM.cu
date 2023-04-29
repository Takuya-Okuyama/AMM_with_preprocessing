#include "common.h"
#include "sgemm_previousAMM.h"
#include "sgemm_proposedAMM.h"

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
    sgemm_proposedAMM<DIM_M>(dm, true);
  }

  cublasSgemm(dm.handle,
              CUBLAS_OP_N,
              CUBLAS_OP_T,
              p.m, p.n, p.k,
              &p.const_one,
              dm.dA, p.m,
              dm.dB, p.n,
              &p.const_zero,
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
                  CUBLAS_OP_T,
                  p.m, p.n, p.k,
                  &p.const_one,
                  dm.dA, p.m,
                  dm.dB, p.n,
                  &p.const_zero,
                  dm.dY_ref, p.m);

      cudaMemcpy(p.hY_ref, dm.dY_ref, sizeof(float) * p.m * p.n, cudaMemcpyDefault);

      // calculate Frobenius norm of AB
      cublasSnrm2(dm.handle, dm.m * p.n, dm.dY_ref, 1, &p.FrobeniusNorm_AB);
    }
    else
    {
      // measure time
      p.start_timer();
      for (int i = 0; i < p.nreps; ++i)
      {
        cublasSgemm(dm.handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    p.m, p.n, p.k,
                    &p.const_one,
                    dm.dA, p.m,
                    dm.dB, p.n,
                    &p.const_zero,
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
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m * p.n, &p.const_one, dm.dY, 1, dm.dY_avg, 1);
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
      cublasSgemv(dm.handle,
                  CUBLAS_OP_N,
                  p.m, p.k,
                  &p.const_one,
                  dm.dA, p.m,
                  dm.d_beta, 1,
                  &p.const_zero,
                  dm.d_vc, 1);

      cublasSgemv(dm.handle,
                  CUBLAS_OP_N,
                  p.n, p.k,
                  &p.const_one,
                  dm.dB, p.n,
                  dm.d_alpha, 1,
                  &p.const_zero,
                  dm.d_vr, 1);

      cublasSdot(dm.handle,
                 p.k,
                 dm.d_alpha, 1,
                 dm.d_beta, 1,
                 dm.d_w);

      sgemm_proposedAMM<DIM_M>(dm);
    }
    else
    {
      // get error of Frobenius norm only, not measure time
      p.error.clear();
      for (int i = 0; i < p.nreps; ++i)
      {
        sgemm_proposedAMM<DIM_M>(dm);

        cudaDeviceSynchronize();
        cudaMemcpy(p.hY, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
        double err = get_error(p.hY_ref, p.hY, p.m * p.n);
        p.error.push_back(err);

        if (p.sanity_check)
        {
          if (i == 0)
          {
            cudaMemcpy(dm.dY_avg, dm.dY, sizeof(float) * p.m * p.n, cudaMemcpyDefault);
          }
          else
          {
            cublasSaxpy(dm.handle, p.m * p.n, &p.const_one, dm.dY, 1, dm.dY_avg, 1);
          }
        }
      }
    }

    // measure time
    p.start_timer();
    for (int i = 0; i < p.nreps; ++i)
    {
      sgemm_proposedAMM<DIM_M>(dm);
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

  if (p.kernel == "proposedAMM")
  {
    cudaDeviceSynchronize();
    update_matrices<<<DIV_CEIL(dm.k, (uint64_t)128), 128>>>(
        dm.dA, dm.dB,
        dm.d_alpha, dm.d_beta,
        dm.m, dm.n, dm.k);

    cublasSgemm(dm.handle,
                CUBLAS_OP_N,
                CUBLAS_OP_T,
                p.m, p.n, p.k,
                &p.const_one,
                dm.dA, p.m,
                dm.dB, p.n,
                &p.const_zero,
                dm.dY_ref, p.m);

    // calculate Frobenius norm of A'B'
    cublasSnrm2(dm.handle, p.m * p.n, dm.dY_ref, 1, &p.FrobeniusNorm_modifiedAB);
  }

  // display info
  p.print_result();

  return 0;
}

