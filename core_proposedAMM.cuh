#ifndef __CORE_PROPOSEDAMM___
#define __CORE_PROPOSEDAMM___

// Calculate approximate matrix product of A and B^T
// by our proposed Monte Carlo AMM
__global__ __launch_bounds__(256) void proposedAMM(
    const int M,
    const int N,
    const int nsamples,
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    const uint32_t *__restrict__ d_ary,
    const float *__restrict__ d_weight,
    const float *__restrict__ d_alpha,
    const float *__restrict__ d_beta,
    const float *__restrict__ d_vr,
    const float *__restrict__ d_vc,
    const float *__restrict__ d_w)
{
  const int tx = threadIdx.x;
  const int bx = blockIdx.x, by = blockIdx.y;
  const int col_a = tx >> 5;
  const int row_c = ((col_a & 3) << 5) + ((tx & 3) << 3);
  const int col_c = ((col_a >> 2) << 6) + ((tx & 28) << 1);
  const int row_a = (tx & 31) << 2;
  const float w = *d_w;

  float4 vc[2], vr[8], tmp[2];
  int tmp_id = (bx << 5) + (row_c >> 2);
  vc[0] = reinterpret_cast<const float4 *>(d_vc)[tmp_id];
  vc[1] = reinterpret_cast<const float4 *>(d_vc)[tmp_id + 1];

  tmp_id = (by << 5) + (col_c >> 2);
  tmp[0] = reinterpret_cast<const float4 *>(d_vr)[tmp_id];
  tmp[1] = reinterpret_cast<const float4 *>(d_vr)[tmp_id + 1];
  vr[0] = make_float4(tmp[0].x, tmp[0].x, tmp[0].x, tmp[0].x);
  vr[1] = make_float4(tmp[0].y, tmp[0].y, tmp[0].y, tmp[0].y);
  vr[2] = make_float4(tmp[0].z, tmp[0].z, tmp[0].z, tmp[0].z);
  vr[3] = make_float4(tmp[0].w, tmp[0].w, tmp[0].w, tmp[0].w);
  vr[4] = make_float4(tmp[1].x, tmp[1].x, tmp[1].x, tmp[1].x);
  vr[5] = make_float4(tmp[1].y, tmp[1].y, tmp[1].y, tmp[1].y);
  vr[6] = make_float4(tmp[1].z, tmp[1].z, tmp[1].z, tmp[1].z);
  vr[7] = make_float4(tmp[1].w, tmp[1].w, tmp[1].w, tmp[1].w);

  A = &A((bx << 7), 0);
  B = &B((by << 7), 0);
  C = &C((bx << 7), (by << 7));

  __shared__ alignas(alignof(float4)) float sa11[2][1024];
  __shared__ alignas(alignof(float4)) float sb11[2][1024];
  __shared__ uint32_t sary[LOOP_LEN];

  float4 Av1[2], Av2[2], Bv1[2], Bv2[2], Cres[16];
  memset(Cres, 0, sizeof(Cres));
  int loop_offset = 0;
  int remaining = min(nsamples, LOOP_LEN);

  do
  {
    if (tx <= 1)
    {
      __pipeline_memcpy_async(sary + 4 * tx, d_ary + 4 * tx, sizeof(float4));
      __pipeline_commit();
      __pipeline_wait_prior(0);
    }
    __syncthreads();

    for (int i = 2 + tx; 4 * i < remaining; i += 256)
    {
      __pipeline_memcpy_async(sary + 4 * i, d_ary + 4 * i, sizeof(float4));
    }
    __pipeline_commit();

    int sel_id = col_a;
    float *ptr_sa11 = (float *)sa11;
    float *ptr_sb11 = (float *)sb11;

    const int index = sary[sel_id];
    float pref_scale = d_weight[index];
    float pref_alpha = d_alpha[index];
    float pref_beta = d_beta[index];

    __pipeline_memcpy_async(ptr_sa11 + 4 * tx, &A(row_a, index), sizeof(float4));
    __pipeline_memcpy_async(ptr_sb11 + 4 * tx, &B(row_a, index), sizeof(float4));
    __pipeline_commit();
    __pipeline_wait_prior(0);

    ((float4 *)ptr_sa11)[tx] = (((float4 *)ptr_sa11)[tx] - pref_alpha) * pref_scale;
    ((float4 *)ptr_sb11)[tx] -= pref_beta;
    __syncthreads();

    // load data
    vload(Av1[0], &ptr_sa11(row_c, 0));
    vload(Av2[0], &ptr_sa11(row_c + 4, 0));
    vload(Bv1[0], &ptr_sb11(col_c, 0));
    vload(Bv2[0], &ptr_sb11(col_c + 4, 0));

    int offset = 0;
    for (remaining -= 8; remaining > 0; remaining -= 8)
    {
      // prefetch A and B into shared memory
      sel_id += 8;
      offset = 1024 - offset;

      const int index = sary[sel_id];

      __pipeline_memcpy_async((float *)sa11 + offset + 4 * tx, &A(row_a, index), sizeof(float4));
      __pipeline_memcpy_async((float *)sb11 + offset + 4 * tx, &B(row_a, index), sizeof(float4));
      __pipeline_commit();

      float pref_scale = d_weight[index];
      float pref_alpha = d_alpha[index];
      float pref_beta = d_beta[index];

#pragma unroll
      for (int inner_k_count = 0; inner_k_count < KS_11 - 1; inner_k_count++)
      {
        int next_inner_k_count = inner_k_count + 1;

        // double buffering
        vload(Av1[(inner_k_count + 1) & 1], &ptr_sa11(row_c, next_inner_k_count));
        vload(Av2[(inner_k_count + 1) & 1], &ptr_sa11(row_c + 4, next_inner_k_count));
        vload(Bv1[(inner_k_count + 1) & 1], &ptr_sb11(col_c, next_inner_k_count));
        vload(Bv2[(inner_k_count + 1) & 1], &ptr_sb11(col_c + 4, next_inner_k_count));

        Cres[0x0] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].x;
        Cres[0x1] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].x;
        Cres[0x2] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].y;
        Cres[0x3] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].y;
        Cres[0x4] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].z;
        Cres[0x5] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].z;
        Cres[0x6] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].w;
        Cres[0x7] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].w;
        Cres[0x8] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].x;
        Cres[0x9] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].x;
        Cres[0xA] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].y;
        Cres[0xB] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].y;
        Cres[0xC] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].z;
        Cres[0xD] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].z;
        Cres[0xE] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].w;
        Cres[0xF] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].w;
      }

      {
        // inner_k_count = 7
        Cres[0x0] += Av1[1] * Bv1[1].x;
        Cres[0x1] += Av2[1] * Bv1[1].x;
        Cres[0x2] += Av1[1] * Bv1[1].y;
        Cres[0x3] += Av2[1] * Bv1[1].y;
        Cres[0x4] += Av1[1] * Bv1[1].z;
        Cres[0x5] += Av2[1] * Bv1[1].z;
        Cres[0x6] += Av1[1] * Bv1[1].w;
        Cres[0x7] += Av2[1] * Bv1[1].w;
        Cres[0x8] += Av1[1] * Bv2[1].x;
        Cres[0x9] += Av2[1] * Bv2[1].x;
        Cres[0xA] += Av1[1] * Bv2[1].y;
        Cres[0xB] += Av2[1] * Bv2[1].y;
        Cres[0xC] += Av1[1] * Bv2[1].z;
        Cres[0xD] += Av2[1] * Bv2[1].z;
        Cres[0xE] += Av1[1] * Bv2[1].w;
        Cres[0xF] += Av2[1] * Bv2[1].w;
      }

      ptr_sa11 = (float *)sa11 + offset;
      ptr_sb11 = (float *)sb11 + offset;
      __pipeline_wait_prior(0);

      ((float4 *)ptr_sa11)[tx] = (((float4 *)ptr_sa11)[tx] - pref_alpha) * pref_scale;
      ((float4 *)ptr_sb11)[tx] -= pref_beta;
      __syncthreads();

      vload(Av1[0], &ptr_sa11(row_c, 0));
      vload(Av2[0], &ptr_sa11(row_c + 4, 0));
      vload(Bv1[0], &ptr_sb11(col_c, 0));
      vload(Bv2[0], &ptr_sb11(col_c + 4, 0));
    }

    {
#pragma unroll
      for (int inner_k_count = 0; inner_k_count < KS_11 - 1; inner_k_count++)
      {
        int next_inner_k_count = inner_k_count + 1;

        // double buffering
        vload(Av1[(inner_k_count + 1) & 1], &ptr_sa11(row_c, next_inner_k_count));
        vload(Av2[(inner_k_count + 1) & 1], &ptr_sa11(row_c + 4, next_inner_k_count));
        vload(Bv1[(inner_k_count + 1) & 1], &ptr_sb11(col_c, next_inner_k_count));
        vload(Bv2[(inner_k_count + 1) & 1], &ptr_sb11(col_c + 4, next_inner_k_count));

        Cres[0x0] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].x;
        Cres[0x1] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].x;
        Cres[0x2] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].y;
        Cres[0x3] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].y;
        Cres[0x4] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].z;
        Cres[0x5] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].z;
        Cres[0x6] += Av1[(inner_k_count)&1] * Bv1[(inner_k_count)&1].w;
        Cres[0x7] += Av2[(inner_k_count)&1] * Bv1[(inner_k_count)&1].w;
        Cres[0x8] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].x;
        Cres[0x9] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].x;
        Cres[0xA] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].y;
        Cres[0xB] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].y;
        Cres[0xC] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].z;
        Cres[0xD] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].z;
        Cres[0xE] += Av1[(inner_k_count)&1] * Bv2[(inner_k_count)&1].w;
        Cres[0xF] += Av2[(inner_k_count)&1] * Bv2[(inner_k_count)&1].w;
      }

      {
        // inner_k_count = 7
        Cres[0x0] += Av1[1] * Bv1[1].x;
        Cres[0x1] += Av2[1] * Bv1[1].x;
        Cres[0x2] += Av1[1] * Bv1[1].y;
        Cres[0x3] += Av2[1] * Bv1[1].y;
        Cres[0x4] += Av1[1] * Bv1[1].z;
        Cres[0x5] += Av2[1] * Bv1[1].z;
        Cres[0x6] += Av1[1] * Bv1[1].w;
        Cres[0x7] += Av2[1] * Bv1[1].w;
        Cres[0x8] += Av1[1] * Bv2[1].x;
        Cres[0x9] += Av2[1] * Bv2[1].x;
        Cres[0xA] += Av1[1] * Bv2[1].y;
        Cres[0xB] += Av2[1] * Bv2[1].y;
        Cres[0xC] += Av1[1] * Bv2[1].z;
        Cres[0xD] += Av2[1] * Bv2[1].z;
        Cres[0xE] += Av1[1] * Bv2[1].w;
        Cres[0xF] += Av2[1] * Bv2[1].w;
      }
    }

    loop_offset += LOOP_LEN;
    d_ary += LOOP_LEN;
    remaining = min(nsamples - loop_offset, LOOP_LEN);
  } while (remaining > 0);

  const float4 vc0_vec_w = vc[0] - make_float4(w, w, w, w);
  const float4 vc1_vec_w = vc[1] - make_float4(w, w, w, w);

  vstore(&C(row_c + 0, col_c + 0), Cres[0x0] + vc0_vec_w + vr[0]);
  vstore(&C(row_c + 4, col_c + 0), Cres[0x1] + vc1_vec_w + vr[0]);
  vstore(&C(row_c + 0, col_c + 1), Cres[0x2] + vc0_vec_w + vr[1]);
  vstore(&C(row_c + 4, col_c + 1), Cres[0x3] + vc1_vec_w + vr[1]);
  vstore(&C(row_c + 0, col_c + 2), Cres[0x4] + vc0_vec_w + vr[2]);
  vstore(&C(row_c + 4, col_c + 2), Cres[0x5] + vc1_vec_w + vr[2]);
  vstore(&C(row_c + 0, col_c + 3), Cres[0x6] + vc0_vec_w + vr[3]);
  vstore(&C(row_c + 4, col_c + 3), Cres[0x7] + vc1_vec_w + vr[3]);
  vstore(&C(row_c + 0, col_c + 4), Cres[0x8] + vc0_vec_w + vr[4]);
  vstore(&C(row_c + 4, col_c + 4), Cres[0x9] + vc1_vec_w + vr[4]);
  vstore(&C(row_c + 0, col_c + 5), Cres[0xA] + vc0_vec_w + vr[5]);
  vstore(&C(row_c + 4, col_c + 5), Cres[0xB] + vc1_vec_w + vr[5]);
  vstore(&C(row_c + 0, col_c + 6), Cres[0xC] + vc0_vec_w + vr[6]);
  vstore(&C(row_c + 4, col_c + 6), Cres[0xD] + vc1_vec_w + vr[6]);
  vstore(&C(row_c + 0, col_c + 7), Cres[0xE] + vc0_vec_w + vr[7]);
  vstore(&C(row_c + 4, col_c + 7), Cres[0xF] + vc1_vec_w + vr[7]);
}

#endif
