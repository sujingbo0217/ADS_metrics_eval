// The tangent space at r_i is the top-d_t principal subspace of the local set
// L_i = {r_i} u N_{k_t}(r_i). Everything is done in the m-dimensional span of
// L_i, never in the ambient space, which reaches 40704 dimensions for pixels:
//
//   X_c  (m, dim)  the local set centered on its own mean
//   G_c = X_c X_c^T = V diag(lam) V^T            (m x m, Jacobi below)
//   u_l = X_c^T V_l / sqrt(lam_l)                 the l-th tangent direction
//   ||U^T v||^2 = sum_{l <= d_t} (V_l^T X_c v)^2 / lam_l
//
// so a displacement v enters only through its m inner products with the local
// points, X v, centered (the local mean drops out of X_c v). G_c comes from the
// global Gram matrix K = R_c R_c^T of the globally centered clean set, one SGEMM
// per call, gathered and re-centered per point in double.
//
// The reference nu_i^0 leaves the neighbor r_j out of L_i and projects
// v = r_j - r_i, whose inner products with L_i are columns of K as well.

#include "geometry_common.cuh"
#include "cuda_check.h"

#include <cublas_v2.h>

#include <cmath>
#include <cstdio>

static constexpr int kMaxLocal = 64;      // m = k_t + 1 <= 64; A is 32 KB of shared
static constexpr int kJacobiThreads = 64;

// Column means in double, subtracted in place into a float copy. One thread per
// column, so consecutive threads read consecutive addresses of each row.
__global__ void center_columns_kernel(const float* __restrict__ X, int n, int dim,
                                      float* __restrict__ Xc) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  double s = 0.0;
  for (int i = 0; i < n; i++) s += (double)X[(size_t)i * dim + d];
  const double mu = s / n;
  for (int i = 0; i < n; i++) {
    Xc[(size_t)i * dim + d] = (float)((double)X[(size_t)i * dim + d] - mu);
  }
}

// K = R_c R_c^T. Global centering first, so that the local re-centering in
// double subtracts quantities of the size of the data's spread, not of its
// norm, from float entries.
static void clean_gram(const float* d_R, int n, int dim, DeviceArray<float>& K) {
  DeviceArray<float> Rc((size_t)n * dim);
  center_columns_kernel<<<(dim + kBlock - 1) / kBlock, kBlock>>>(d_R, n, dim, Rc.get());
  CUDA_CHECK_KERNEL();
  K.alloc((size_t)n * n);
  // Row-major R_c is column-major R_c^T (dim, n); K = (R_c^T)^T R_c^T.
  const float one = 1.0f, zero = 0.0f;
  CUBLAS_CHECK(cublasSgemm(knn_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, n, n, dim,
                           &one, Rc.get(), dim, Rc.get(), dim, &zero, K.get(), n));
}

__device__ __forceinline__ double warp_sum(double v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// One block per point: U[i, a] = <r_{L_a}, delta_i> for the m local points, and
// dd[i] = ||delta_i||^2, in double. One warp per inner product.
__global__ void local_dots_kernel(const float* __restrict__ R, const float* __restrict__ G,
                                  int dim, const int* __restrict__ nbr, int kt,
                                  double* __restrict__ U, double* __restrict__ dd) {
  const int i = blockIdx.x, m = kt + 1;
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
  const float* gi = G + (size_t)i * dim;
  const float* ri = R + (size_t)i * dim;
  for (int a = warp; a <= m; a += nw) {
    double s = 0.0;
    if (a < m) {
      const int l = a == 0 ? i : nbr[(size_t)i * kt + a - 1];
      const float* rl = R + (size_t)l * dim;
      for (int d = lane; d < dim; d += 32) {
        s += (double)rl[d] * ((double)gi[d] - (double)ri[d]);
      }
    } else {
      for (int d = lane; d < dim; d += 32) {
        double v = (double)gi[d] - (double)ri[d];
        s += v * v;
      }
    }
    s = warp_sum(s);
    if (lane == 0) {
      if (a < m) U[(size_t)i * m + a] = s;
      else dd[i] = s;
    }
  }
}

// One block per task. Plain task: point i, the full local set, v = delta_i.
// Reference task (kRef): point i = task / k_ref, neighbor p = task % k_ref left
// out, v = r_j - r_i.
//
// Cyclic Jacobi on the centered local Gram matrix in shared memory. The
// eigenvectors are never stored: each rotation J is applied to y = V^T w
// directly (V <- V J gives y <- J^T y), which is all the projection needs.
template <bool kRef>
__global__ void tangent_residual_kernel(const float* __restrict__ K, int n,
                                        const int* __restrict__ nbr, int kt, int dt,
                                        int k_ref, const double* __restrict__ U,
                                        const double* __restrict__ dd,
                                        float* __restrict__ out) {
  __shared__ double A[kMaxLocal * kMaxLocal];
  __shared__ double y[kMaxLocal];
  __shared__ double rm[kMaxLocal];
  __shared__ int L[kMaxLocal];
  __shared__ double s_grand, s_ymean, s_vv, s_off, s_diag;

  const int tid = threadIdx.x, bd = blockDim.x;
  const int task = blockIdx.x;
  const int i = kRef ? task / k_ref : task;
  const int p = kRef ? task % k_ref : -1;
  const int m = kRef ? kt : kt + 1;
  const int j = kRef ? nbr[(size_t)i * kt + p] : -1;

  if (tid == 0) {
    L[0] = i;
    int c = 1;
    for (int q = 0; q < kt; q++) {
      if (q != p) L[c++] = nbr[(size_t)i * kt + q];
    }
  }
  __syncthreads();

  for (int e = tid; e < m * m; e += bd) {
    A[e] = (double)K[(size_t)L[e / m] * n + L[e % m]];
  }
  for (int a = tid; a < m; a += bd) {
    y[a] = kRef ? (double)K[(size_t)L[a] * n + j] - (double)K[(size_t)L[a] * n + i]
                : U[(size_t)i * m + a];
  }
  __syncthreads();

  // Double centering: G_c = G - row means - column means + grand mean.
  for (int a = tid; a < m; a += bd) {
    double s = 0.0;
    for (int b = 0; b < m; b++) s += A[a * m + b];
    rm[a] = s / m;
  }
  __syncthreads();
  if (tid == 0) {
    double g = 0.0, ym = 0.0;
    for (int a = 0; a < m; a++) {
      g += rm[a];
      ym += y[a];
    }
    s_grand = g / m;
    s_ymean = ym / m;
    s_vv = kRef ? (double)K[(size_t)j * n + j] + (double)K[(size_t)i * n + i] -
                      2.0 * (double)K[(size_t)i * n + j]
                : dd[i];
  }
  __syncthreads();
  for (int e = tid; e < m * m; e += bd) A[e] += s_grand - rm[e / m] - rm[e % m];
  for (int a = tid; a < m; a += bd) y[a] -= s_ymean;
  __syncthreads();

  for (int sweep = 0; sweep < 50; sweep++) {
    if (tid == 0) {
      double off = 0.0, dg = 0.0;
      for (int a = 0; a < m; a++) {
        dg += A[a * m + a] * A[a * m + a];
        for (int b = a + 1; b < m; b++) off += A[a * m + b] * A[a * m + b];
      }
      s_off = off;
      s_diag = dg;
    }
    __syncthreads();
    if (s_off <= 1e-26 * s_diag) break;   // uniform: every thread reads the same

    for (int pp = 0; pp < m - 1; pp++) {
      for (int qq = pp + 1; qq < m; qq++) {
        const double apq = A[pp * m + qq];
        if (apq == 0.0) continue;         // uniform, as above
        const double app = A[pp * m + pp], aqq = A[qq * m + qq];
        // Numerical Recipes' rotation: the smaller root of t^2 + 2 theta t - 1.
        // For theta beyond ~1e154 theta^2 overflows, t becomes 0, and the
        // rotation reduces to zeroing a negligible a_pq.
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                         (fabs(theta) + sqrt(theta * theta + 1.0));
        const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
        __syncthreads();                  // all reads of A done before any write
        for (int x = tid; x < m; x += bd) {
          if (x == pp || x == qq) continue;
          const double axp = A[x * m + pp], axq = A[x * m + qq];
          const double nxp = c * axp - s * axq, nxq = s * axp + c * axq;
          A[x * m + pp] = nxp;
          A[pp * m + x] = nxp;
          A[x * m + qq] = nxq;
          A[qq * m + x] = nxq;
        }
        if (tid == 0) {
          A[pp * m + pp] = app - t * apq;
          A[qq * m + qq] = aqq + t * apq;
          A[pp * m + qq] = 0.0;
          A[qq * m + pp] = 0.0;
          const double yp = y[pp], yq = y[qq];
          y[pp] = c * yp - s * yq;
          y[qq] = s * yp + c * yq;
        }
        __syncthreads();
      }
    }
  }

  if (tid == 0) {
    // The centered local set has rank at most m - 1.
    const int d = dt < m - 1 ? dt : m - 1;
    double lam[kMaxLocal];
    bool taken[kMaxLocal];
    double lmax = 0.0;
    for (int a = 0; a < m; a++) {
      lam[a] = A[a * m + a];
      taken[a] = false;
      lmax = fmax(lmax, lam[a]);
    }
    double proj = 0.0;
    for (int r = 0; r < d; r++) {
      int best = -1;
      for (int a = 0; a < m; a++) {
        if (!taken[a] && (best < 0 || lam[a] > lam[best])) best = a;
      }
      taken[best] = true;
      if (lam[best] > 1e-12 * lmax) proj += y[best] * y[best] / lam[best];
    }
    const double vv = s_vv;
    out[task] = vv > 1e-30 ? (float)sqrt(fmax(0.0, 1.0 - proj / vv)) : nanf("");
  }
}

static void check_local(int n, int k_t) {
  if (k_t < 1 || k_t + 1 > kMaxLocal || k_t > n - 1) {
    std::fprintf(stderr, "[knn_ext] k_t = %d out of range: need 1 <= k_t <= %d and < n\n",
                 k_t, kMaxLocal - 1);
    std::abort();
  }
}

void off_manifold(const float* R, const float* G, int n, int dim, int k_t, int d_t,
                  DistanceType dist_type, float* nu_pt) {
  check_local(n, k_t);
  DeviceArray<float> d_R, d_G;
  d_R.from_host(R, (size_t)n * dim);
  d_G.from_host(G, (size_t)n * dim);

  DeviceArray<float> K;
  clean_gram(d_R.get(), n, dim, K);
  knnResult nb = mixed_knn(d_R.get(), n, dim, k_t, dist_type);

  DeviceArray<double> U((size_t)n * (k_t + 1)), dd(n);
  local_dots_kernel<<<n, kBlock>>>(d_R.get(), d_G.get(), dim, nb.indices, k_t, U.get(),
                                   dd.get());
  CUDA_CHECK_KERNEL();
  tangent_residual_kernel<false><<<n, kJacobiThreads>>>(K.get(), n, nb.indices, k_t, d_t,
                                                        1, U.get(), dd.get(), nu_pt);
  CUDA_CHECK_KERNEL();
  free_knn_result(nb);
}

void off_manifold_reference(const float* R, int n, int dim, int k, int k_t, int d_t,
                            DistanceType dist_type, float* nu0) {
  check_local(n, k_t);
  if (k < 1 || k > k_t) {
    std::fprintf(stderr, "[knn_ext] reference k = %d must lie in [1, k_t = %d]\n", k, k_t);
    std::abort();
  }
  DeviceArray<float> d_R;
  d_R.from_host(R, (size_t)n * dim);
  DeviceArray<float> K;
  clean_gram(d_R.get(), n, dim, K);
  knnResult nb = mixed_knn(d_R.get(), n, dim, k_t, dist_type);
  tangent_residual_kernel<true><<<n * k, kJacobiThreads>>>(K.get(), n, nb.indices, k_t,
                                                           d_t, k, nullptr, nullptr, nu0);
  CUDA_CHECK_KERNEL();
  free_knn_result(nb);
}
