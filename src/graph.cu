//   effective_rank              GP1 spectral shape
//   lid_profile, compute_lid    GP2 local scale and dimension
//   relative_contrast           GP3 concentration (hubness: neighborhood.cu)
//   cross_set_mixing, cross_lid GP5 mixing and support
//   neighborhood_preservation   GP6 trustworthiness and continuity
//   neighborhood_drift          GP6 drift

#include "geometry_common.cuh"
#include "cuda_check.h"

#include <cstddef>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <cmath>
#include <cstdio>
#include <vector>

#define CUSOLVER_CHECK(expr)                                                  \
  do {                                                                        \
    cusolverStatus_t _s = (expr);                                             \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "[knn_ext] cuSOLVER error at %s:%d in `%s`: %d\n", \
                   __FILE__, __LINE__, #expr, (int)_s);                       \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

static cusolverDnHandle_t cusolver_handle() {
  static cusolverDnHandle_t h = nullptr;
  if (!h) CUSOLVER_CHECK(cusolverDnCreate(&h));
  return h;
}

__global__ void fill_nan_kernel(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = nanf("");
}

// ───────────────────────────────────────────────────────────────────────────
// GP1 spectral shape
// ───────────────────────────────────────────────────────────────────────────

__global__ void col_sum_kernel(const float* __restrict__ X, int n, int dim,
                               double* __restrict__ partial) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  double s = 0.0;
  for (int i = blockIdx.y; i < n; i += gridDim.y) {
    s += (double)X[(size_t)i * dim + d];
  }
  partial[(size_t)blockIdx.y * dim + d] = s;
}

__global__ void reduce_partial_kernel(const double* __restrict__ partial, int chunks,
                                      int dim, double scale, double* __restrict__ out) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= dim) return;
  double s = 0.0;
  for (int c = 0; c < chunks; c++) {
    s += partial[(size_t)c * dim + d];
  }
  out[d] = s * scale;
}

// Center a chunk of consecutive ROWS into double. Row-major (c, dim) is the
// same linear layout as column-major (dim, c), so this block is A[:, i0:i0+c]
// with no transpose.
// grid = (ceil(dim / kBlock), yblocks), block = kBlock
__global__ void stage_rows_kernel(const float* __restrict__ X, const double* __restrict__ mu,
                                  int i0, int rows, int dim, double* __restrict__ out) {
  const size_t dd = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const float* __restrict__ tmp = X + (size_t)i0 * dim;
  for (size_t d = dd; d < dim; d += (size_t)gridDim.x * blockDim.x) {
    const double m = mu[d];
    for (int r = blockIdx.y; r < rows; r += gridDim.y) {
      const size_t offset = (size_t)r * dim + d;
      out[offset] = (double)tmp[offset] - m;
    }
  }
}

// Center a chunk of consecutive COLUMNS into double, transposed to (cols, n)
// column-major so that a single Dsyrk accumulates X_c^T X_c over it.
// grid = (ceil(cols / kBlock), yblocks), block = kBlock
__global__ void stage_cols_kernel(const float* __restrict__ X, const double* __restrict__ mu,
                                  int d0, int cols, int n, int dim,
                                  double* __restrict__ out) {
  const size_t cc = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t c = cc; c < cols; c += (size_t)gridDim.x * blockDim.x) {
    const double m = mu[d0 + c];
    for (int i = blockIdx.y; i < n; i += gridDim.y) {
      out[(size_t)i * cols + c] = (double)X[(size_t)i * dim + d0 + c] - m;
    }
  }
}

static dim3 grid_2d(int fast_dim, int slow_dim, int block, int target_blocks = 2048) {
  int gx = (fast_dim + block - 1) / block;
  int gy = target_blocks / (gx > 0 ? gx : 1);
  if (gy < 1) gy = 1;
  if (gy > slow_dim) gy = slow_dim;
  if (gy > 65535) gy = 65535;
  return dim3(gx, gy);
}

// The spectrum is taken from the SMALLER of the two Gram matrices: X_c^T X_c is
// (dim, dim) and X_c X_c^T is (n, n), they share their non-zero eigenvalues,
// and there are min(n, dim) of them either way -- the same count numpy's SVD
// of the centered block returns. Accumulated in float64 because the tail of
// the spectrum is what the entropy is sensitive to, and a float32 GEMM leaves
// nothing below ~1e-6 of the leading eigenvalue that is not rounding noise.
static double effective_rank_device(const float* d_X, int n, int dim, const double* d_mu) {
  const int m = n < dim ? n : dim;
  DeviceArray<double> M((size_t)m * m);
  M.zero();

  cublasHandle_t bh = knn_cublas_handle();
  const double one = 1.0;
  // Cap the float64 staging buffer at 128 MB however wide the space is.
  const size_t kStage = 16u << 20;

  if (dim <= n) {
    // M = A A^T over row chunks, A column-major (dim, n).
    int chunk = (int)(kStage / (size_t)dim);
    if (chunk < 1) chunk = 1;
    if (chunk > n) chunk = n;
    DeviceArray<double> stage((size_t)chunk * dim);
    for (int i0 = 0; i0 < n; i0 += chunk) {
      int rows = (i0 + chunk <= n) ? chunk : n - i0;
      stage_rows_kernel<<<grid_2d(dim, rows, kBlock), kBlock>>>(
          d_X, d_mu, i0, rows, dim, stage.get());
      CUDA_CHECK_KERNEL();
      CUBLAS_CHECK(cublasDsyrk(bh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, dim, rows,
                               &one, stage.get(), dim, &one, M.get(), dim));
    }
  } else {
    // M = A^T A over column chunks.
    int chunk = (int)(kStage / (size_t)n);
    if (chunk < 1) chunk = 1;
    if (chunk > dim) chunk = dim;
    DeviceArray<double> stage((size_t)chunk * n);
    for (int d0 = 0; d0 < dim; d0 += chunk) {
      int cols = (d0 + chunk <= dim) ? chunk : dim - d0;
      stage_cols_kernel<<<grid_2d(cols, n, kBlock), kBlock>>>(
          d_X, d_mu, d0, cols, n, dim, stage.get());
      CUDA_CHECK_KERNEL();
      CUBLAS_CHECK(cublasDsyrk(bh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, cols,
                               &one, stage.get(), cols, &one, M.get(), n));
    }
  }

  // Eigenvalues only: the entropy does not need the eigenvectors, and NOVECTOR
  // is what keeps this cheaper than the SVD it replaces.
  cusolverDnHandle_t sh = cusolver_handle();
  DeviceArray<double> w(m);
  int lwork = 0;
  CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(sh, CUSOLVER_EIG_MODE_NOVECTOR,
                                             CUBLAS_FILL_MODE_LOWER, m, M.get(), m,
                                             w.get(), &lwork));
  DeviceArray<double> work((size_t)lwork);
  DeviceArray<int> info(1);
  CUSOLVER_CHECK(cusolverDnDsyevd(sh, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                                  m, M.get(), m, w.get(), work.get(), lwork, info.get()));
  int h_info = 0;
  info.to_host(&h_info);
  if (h_info != 0) {
    std::fprintf(stderr, "[knn_ext] cusolverDnDsyevd did not converge (info = %d)\n", h_info);
    return NAN;
  }

  std::vector<double> lam(m);
  w.to_host(lam.data());
  double sum = 0.0;
  for (int i = 0; i < m; i++) {
    if (lam[i] < 0.0) lam[i] = 0.0;
    lam[i] /= (n > 1 ? n - 1 : 1);
    sum += lam[i];
  }
  if (sum < 1e-30) sum = 1e-30;
  double h = 0.0;
  for (int i = 0; i < m; i++) {
    double p = lam[i] / sum;
    if (p > 1e-12) h -= p * std::log(p);
  }
  return std::exp(h);
}

double effective_rank(const float* X, int n, int dim) {
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);

  const int chunks = n < 64 ? n : 64;
  DeviceArray<double> partial((size_t)chunks * dim);
  DeviceArray<double> d_mu(dim);
  dim3 grid((dim + kBlock - 1) / kBlock, chunks);
  col_sum_kernel<<<grid, kBlock>>>(d_X.get(), n, dim, partial.get());
  CUDA_CHECK_KERNEL();
  reduce_partial_kernel<<<(dim + kBlock - 1) / kBlock, kBlock>>>(
      partial.get(), chunks, dim, 1.0 / n, d_mu.get());
  CUDA_CHECK_KERNEL();
  return effective_rank_device(d_X.get(), n, dim, d_mu.get());
}

// ───────────────────────────────────────────────────────────────────────────
// GP2 local scale and dimension
// ───────────────────────────────────────────────────────────────────────────

// mu_i = d2/d1, or -1 where the point is unusable. Facco's estimator keeps
// only ratios above 1; a coincident pair gives d1 = 0 and no ratio at all.
__global__ void twonn_ratio_kernel(const float* __restrict__ dist, int n, int ld,
                                   float* __restrict__ mu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float r1 = dist[(size_t)i * ld], r2 = dist[(size_t)i * ld + 1];
  float m = (r1 > 1e-12f) ? r2 / r1 : -1.0f;
  mu[i] = (isfinite(m) && m > 1.0f) ? m : -1.0f;
}

struct not_usable {
  __device__ bool operator()(float v) const { return !(v > 1.0f); }
};

// The straight-line fit through the origin of Facco et al. (2017):
// slope = sum(x*y) / sum(x*x), x = log mu_(j), y = -log(1 - j/m).
struct twonn_term {
  const float* mu;
  int m;
  bool xy;
  __device__ double operator()(int j) const {
    double x = log((double)mu[j]);
    if (!xy) return x * x;
    double y = -log1p(-((double)(j + 1) / (double)m));
    return x * y;
  }
};

static double twonn_from_distances(const float* d_dist, int n, int ld) {
  if (ld < 2) return NAN;
  DeviceArray<float> mu(n);
  twonn_ratio_kernel<<<grid_for(n), kBlock>>>(d_dist, n, ld, mu.get());
  CUDA_CHECK_KERNEL();

  thrust::device_ptr<float> p(mu.get());
  auto end = thrust::remove_if(thrust::device, p, p + n, not_usable());
  int m = (int)(end - p);
  if (m < 10) return NAN;
  thrust::sort(thrust::device, p, p + m);

  // Facco's prescription: drop the top 10% of the ratio distribution, where
  // the fit bends away from the line.
  int keep = (int)(m * 0.9);
  if (keep < 1) return NAN;
  thrust::counting_iterator<int> first(0);
  double sxy = thrust::transform_reduce(thrust::device, first, first + keep,
                                        twonn_term{mu.get(), m, true}, 0.0,
                                        thrust::plus<double>());
  double sxx = thrust::transform_reduce(thrust::device, first, first + keep,
                                        twonn_term{mu.get(), m, false}, 0.0,
                                        thrust::plus<double>());
  return sxx > 0.0 ? sxy / sxx : NAN;
}

// Local intrinsic dimensionality, Levina-Bickel / Hill MLE.
//
//   LID = -(k-2) / sum_{i=1..k-1} ln(d_i / d_k)
//
// IMPORTANT: `dist` must hold TRUE distances, not the squared ones that
// DistanceType::L2 produces, and not the negative dot products that
// DistanceType::InnerProduct produces. Squared input halves the estimate,
// since ln(d_i^2/d_k^2) = 2 ln(d_i/d_k); negative input makes the logs NaN.
// to_true_distance performs that conversion on the device.
__global__ void compute_lid_kernel(const float *dist, int n_q, int k, int ld, float *output) {
  // dist sorted ascending, row stride ld >= k
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= n_q) return;

  const float* row = dist + (size_t)qi * ld;
  float dk = row[k - 1];
  constexpr float min_d = 1e-10f;
  constexpr float eps = 1e-6f;
  if (dk < min_d) {
    output[qi] = 0.0f;
    return;
  }

  // Numerator is k-2, not k-1 and certainly not k. The sum runs over k-1
  // terms; Levina & Bickel's own paper notes, and MacKay & Ghahramani (2005)
  // show, that dividing by k-2 rather than k-1 makes the per-point estimator
  // asymptotically unbiased. The original code divided by k, inflating every
  // estimate by k/(k-1) on top of that -- 11% at k=10, 2% at k=50, a bias that
  // VARIES WITH k and would therefore have surfaced as spurious structure in
  // the k ablation.
  //
  // All of these are constant rescalings at fixed k, so they cancel in the
  // reported log-ratio and leave Spearman on the per-point values unchanged.
  // Corrected anyway: the per-point LID is reported in its own units.
  float sum_log = 0.0f;
  for (int i = 0; i < k - 1; i++) {
    float di = fmaxf(row[i], min_d);
    sum_log += logf(di / dk);
  }
  // Every term is <= 0, so sum_log <= 0 and the guard must exclude a sum that
  // is merely CLOSE TO zero (all k neighbors equidistant, LID undefined).
  // The old test `sum_log < eps` with eps = +1e-6 was true for sum_log = 0 and
  // returned -inf there, and was false only in the impossible positive case --
  // so it fired backwards and never caught what it was written to catch.
  output[qi] = (k > 2 && sum_log < -eps) ? (-(float)(k - 2) / sum_log) : 0.0f;
}

void compute_lid(const float *dist, int n_q, int k, int ld, float *output) {
  int block_size = 256;
  int grid_size = (n_q + block_size - 1) / block_size;
  compute_lid_kernel<<<grid_size, block_size>>>(dist, n_q, k, ld, output);
  CUDA_CHECK_KERNEL();
}

__global__ void column_kernel(const float* __restrict__ m, int n, int ld, int col,
                              float* __restrict__ out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = m[(size_t)i * ld + col];
}

double lid_profile(const float* X, int n, int dim, int k, DistanceType dist_type,
                   float* lid_pt, float* rk_pt) {
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);

  // TwoNN needs two neighbors and the MLE needs k; one search covers both.
  int kk = k > 3 ? k : 3;
  if (kk > n - 1) kk = n - 1;
  knnResult res = mixed_knn(d_X.get(), n, dim, kk, dist_type);
  to_true_distance(res.distances, (size_t)n * kk, dist_type);

  if (lid_pt) {
    // The bias-corrected MLE sums k-1 log-ratios and divides by k-2, so it is
    // undefined below k = 3. The ablation grid includes k = 1.
    if (k < 3 || k > kk) {
      fill_nan_kernel<<<grid_for(n), kBlock>>>(lid_pt, n);
      CUDA_CHECK_KERNEL();
    } else {
      compute_lid(res.distances, n, k, kk, lid_pt);
    }
  }
  if (rk_pt) {
    column_kernel<<<grid_for(n), kBlock>>>(res.distances, n, kk, k - 1, rk_pt);
    CUDA_CHECK_KERNEL();
  }
  double twonn = twonn_from_distances(res.distances, n, kk);
  free_knn_result(res);
  return twonn;
}

// ───────────────────────────────────────────────────────────────────────────
// GP3 concentration
// ───────────────────────────────────────────────────────────────────────────

// One block per row of a masked true-distance matrix.
__global__ void contrast_kernel(const float* __restrict__ D, int n,
                                float* __restrict__ mean_pt, float* __restrict__ nn_pt) {
  const float* row = D + (size_t)blockIdx.x * n;
  double s = 0.0;
  float mn = INFINITY;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    float v = row[j];
    if (v < 1e30f) {
      s += v;
      mn = fminf(mn, v);
    }
  }
  typedef cub::BlockReduce<double, kBlock> RD;
  typedef cub::BlockReduce<float, kBlock> RF;
  __shared__ typename RD::TempStorage t1;
  __shared__ typename RF::TempStorage t2;
  double tot = RD(t1).Sum(s);
  float m = RF(t2).Reduce(mn, cub::Min());
  if (threadIdx.x == 0) {
    mean_pt[blockIdx.x] = (float)(tot / (n - 1));
    nn_pt[blockIdx.x] = m;
  }
}

void relative_contrast(const float* X, int n, int dim, DistanceType dist_type,
                       float* mean_pt, float* nn_pt) {
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);
  Neighborhood primary;
  primary.dist = dist_type;
  DeviceArray<float> D;
  construction_matrix(d_X.get(), n, dim, primary, 0, D);
  contrast_kernel<<<n, kBlock>>>(D.get(), n, mean_pt, nn_pt);
  CUDA_CHECK_KERNEL();
}

// ───────────────────────────────────────────────────────────────────────────
// GP5 cross-set mixing and support
// ───────────────────────────────────────────────────────────────────────────

__global__ void mixing_kernel(const int* __restrict__ neighbors, int n_r, int n_g, int k,
                              int excl, float* __restrict__ pt_real,
                              float* __restrict__ pt_gen) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int N = n_r + n_g;
  if (i >= N) return;
  int from_gen = 0;
  for (int j = 0; j < k; j++) from_gen += (neighbors[(size_t)i * k + j] >= n_r);
  double frac = (double)from_gen / k;
  // Divided by what random mixing would give: each candidate is equally
  // likely, and the excluded counterpart is one fewer candidate of the other
  // set, so the expected share is (n_other - excl) / (N - 1 - excl).
  const double cand = (double)(N - 1 - excl);
  if (i < n_r) pt_real[i] = (float)(frac / ((double)(n_g - excl) / cand));
  else pt_gen[i - n_r] = (float)((1.0 - frac) / ((double)(n_r - excl) / cand));
}

void cross_set_mixing(const float* R, const float* G, int n_r, int n_g, int dim, int k,
                      const Neighborhood& nb, bool exclude_counterpart,
                      float* pt_real, float* pt_gen) {
  const int N = n_r + n_g;
  // Stacked on the device. The Python side used to np.vstack, which copies both
  // sets on the host before the same bytes cross the bus anyway.
  DeviceArray<float> Z((size_t)N * dim);
  CUDA_CHECK(cudaMemcpy(Z.get(), R, (size_t)n_r * dim * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(Z.get() + (size_t)n_r * dim, G,
                        (size_t)n_g * dim * sizeof(float), cudaMemcpyHostToDevice));

  const int excl = exclude_counterpart ? 1 : 0;
  DeviceArray<float> D;
  construction_matrix(Z.get(), N, dim, nb, excl ? n_r : 0, D);
  knnResult res = knn_from_matrix(D.get(), N, N, k, -1);
  D.free();
  mixing_kernel<<<grid_for(N), kBlock>>>(res.indices, n_r, n_g, k, excl, pt_real, pt_gen);
  CUDA_CHECK_KERNEL();
  free_knn_result(res);
}

__global__ void mask_rows_self_kernel(float* D, int rows, int ld) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows) D[(size_t)i * ld + i] = INFINITY;
}

void cross_lid(const float* R, const float* G, int n_r, int n_g, int dim, int k,
               DistanceType dist_type, bool exclude_counterpart, float* lid_g) {
  if (k < 3) {
    fill_nan_kernel<<<grid_for(n_g), kBlock>>>(lid_g, n_g);
    CUDA_CHECK_KERNEL();
    return;
  }
  DeviceArray<float> d_R, d_G;
  d_R.from_host(R, (size_t)n_r * dim);
  d_G.from_host(G, (size_t)n_g * dim);
  DeviceArray<float> D((size_t)n_g * n_r);
  pairwise_distances(d_R.get(), d_G.get(), n_r, n_g, dim, dist_type, D.get());
  to_true_distance(D.get(), (size_t)n_g * n_r, dist_type);
  if (exclude_counterpart) {
    mask_rows_self_kernel<<<grid_for(n_g), kBlock>>>(D.get(), n_g, n_r);
    CUDA_CHECK_KERNEL();
  }
  knnResult res = knn_from_matrix(D.get(), n_g, n_r, k, -1);
  compute_lid(res.distances, n_g, k, k, lid_g);
  free_knn_result(res);
}

// ───────────────────────────────────────────────────────────────────────────
// GP6 neighborhood preservation and drift
// ───────────────────────────────────────────────────────────────────────────

__global__ void row_iota_kernel(int* v, int n) {
  const size_t total = (size_t)n * n;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < total; i += (size_t)gridDim.x * blockDim.x) {
    v[i] = (int)(i % n);
  }
}

__global__ void row_offsets_kernel(int* off, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= n) off[i] = i * n;
}

// rank[i, order[i, j]] = j + 1, so 1 is the nearest. Self sits at rank n,
// pushed there by the masked diagonal.
__global__ void scatter_rank_kernel(const int* __restrict__ order, int* __restrict__ rank,
                                    int n) {
  const size_t total = (size_t)n * n;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < total; i += (size_t)gridDim.x * blockDim.x) {
    size_t row = i / n;
    int j = (int)(i % n);
    rank[row * n + order[i]] = j + 1;
  }
}

// One block per point. Both statistics are sums of a per-point penalty, so the
// per-point form is the summand carrying its share of the normalizer -- the
// same baseline 1, and an image-level answer to "was this neighborhood
// preserved", which is what the tier-2 regression needs.
__global__ void trust_cont_kernel(const int* __restrict__ rank_R,
                                  const int* __restrict__ rank_G, int n, int k,
                                  double norm, float* __restrict__ t_pt,
                                  float* __restrict__ c_pt) {
  int i = blockIdx.x;
  long long ts = 0, cs = 0;
  for (int j = threadIdx.x; j < n; j += blockDim.x) {
    int rr = rank_R[(size_t)i * n + j];
    int rg = rank_G[(size_t)i * n + j];
    bool in_r = rr <= k, in_g = rg <= k;
    if (in_g && !in_r) ts += rr - k;   // false friend: near in G, far in R
    if (in_r && !in_g) cs += rg - k;   // lost friend: near in R, far in G
  }
  typedef cub::BlockReduce<long long, kBlock> Reduce;
  __shared__ typename Reduce::TempStorage tmp;
  long long t_tot = Reduce(tmp).Sum(ts);
  __syncthreads();
  long long c_tot = Reduce(tmp).Sum(cs);
  if (threadIdx.x == 0) {
    t_pt[i] = (float)(1.0 - norm * (double)t_tot);
    c_pt[i] = (float)(1.0 - norm * (double)c_tot);
  }
}

// Rows of the construction matrix -> ranks. The matrix is consumed, the ranks
// are kept.
static void ranks_from(const float* d_X, int n, int dim, const Neighborhood& nb,
                       DeviceArray<float>& dist_out, DeviceArray<int>& idx_in,
                       DeviceArray<int>& idx_out, DeviceArray<char>& tmp,
                       size_t tmp_bytes, const int* offsets, int* rank) {
  DeviceArray<float> D;
  construction_matrix(d_X, n, dim, nb, 0, D);
  row_iota_kernel<<<grid_for((size_t)n * n), kBlock>>>(idx_in.get(), n);
  CUDA_CHECK_KERNEL();
  // Radix sort is stable, which is what makes these ranks reproduce numpy's
  // argsort(kind="stable") tie for tie.
  size_t bytes = tmp_bytes;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
      tmp.get(), bytes, D.get(), dist_out.get(), idx_in.get(), idx_out.get(),
      n * n, n, offsets, offsets + 1));
  scatter_rank_kernel<<<grid_for((size_t)n * n), kBlock>>>(idx_out.get(), rank, n);
  CUDA_CHECK_KERNEL();
}

void neighborhood_preservation(const float* R, const float* G, int n, int dim, int k,
                               const Neighborhood& nb, float* t_pt, float* c_pt) {
  DeviceArray<float> d_R, d_G;
  d_R.from_host(R, (size_t)n * dim);
  d_G.from_host(G, (size_t)n * dim);

  const size_t nn = (size_t)n * n;
  DeviceArray<float> dist_out(nn);
  DeviceArray<int> idx_in(nn), idx_out(nn);
  DeviceArray<int> offsets(n + 1);
  row_offsets_kernel<<<grid_for(n + 1), kBlock>>>(offsets.get(), n);
  CUDA_CHECK_KERNEL();

  size_t tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, tmp_bytes, (const float*)nullptr, dist_out.get(), idx_in.get(),
      idx_out.get(), n * n, n, offsets.get(), offsets.get() + 1));
  DeviceArray<char> tmp(tmp_bytes);

  DeviceArray<int> rank_R(nn), rank_G(nn);
  ranks_from(d_R.get(), n, dim, nb, dist_out, idx_in, idx_out, tmp, tmp_bytes,
             offsets.get(), rank_R.get());
  ranks_from(d_G.get(), n, dim, nb, dist_out, idx_in, idx_out, tmp, tmp_bytes,
             offsets.get(), rank_G.get());
  dist_out.free();
  idx_in.free();
  idx_out.free();
  tmp.free();

  // Venna & Kaski's normalizer is the worst-case error sum and has TWO branches
  // (Lee & Verleysen 2009, eq. 4):
  //     G_K = N K (2N - 3K - 1)    K <  N/2
  //           N (N-K)(N-K-1)       K >= N/2
  // Only the first is reachable on our grid (largest k = 100 against smallest
  // n = 200), but using it unconditionally would silently return values outside
  // [0, 1] for anyone who widened k.
  double gk = (k < n / 2.0) ? (double)n * k * (2.0 * n - 3.0 * k - 1.0)
                            : (double)n * (n - k) * (n - k - 1.0);
  double norm = 2.0 * n / gk;
  trust_cont_kernel<<<n, kBlock>>>(rank_R.get(), rank_G.get(), n, k, norm, t_pt, c_pt);
  CUDA_CHECK_KERNEL();
}

// One block per point: || sum_{A_i} r - sum_{B_i} r || / (k r_k(r_i)). The
// shared members of A_i and B_i cancel, but summing both in full keeps the
// kernel branch-free and costs 2k reads per coordinate.
__global__ void drift_kernel(const float* __restrict__ R, int n, int dim, int k,
                             const int* __restrict__ A, const int* __restrict__ B,
                             const float* __restrict__ rk, float* __restrict__ b_pt) {
  extern __shared__ int lists[];   // 2k
  const int i = blockIdx.x;
  for (int t = threadIdx.x; t < k; t += blockDim.x) {
    lists[t] = A[(size_t)i * k + t];
    lists[k + t] = B[(size_t)i * k + t];
  }
  __syncthreads();
  double ss = 0.0;
  for (int d = threadIdx.x; d < dim; d += blockDim.x) {
    double s = 0.0;
    for (int t = 0; t < k; t++) {
      s += (double)R[(size_t)lists[t] * dim + d];
      s -= (double)R[(size_t)lists[k + t] * dim + d];
    }
    ss += s * s;
  }
  typedef cub::BlockReduce<double, kBlock> Reduce;
  __shared__ typename Reduce::TempStorage tmp;
  double tot = Reduce(tmp).Sum(ss);
  if (threadIdx.x == 0) {
    float r = rk[i];
    b_pt[i] = r > 0.0f ? (float)(sqrt(tot) / k / r) : nanf("");
  }
}

void neighborhood_drift(const float* R, const float* G, int n, int dim, int k,
                        const Neighborhood& nb, float* b_pt) {
  DeviceArray<float> d_R, d_G;
  d_R.from_host(R, (size_t)n * dim);
  d_G.from_host(G, (size_t)n * dim);

  knnResult a, b;
  {
    DeviceArray<float> D;
    construction_matrix(d_R.get(), n, dim, nb, 0, D);
    a = knn_from_matrix(D.get(), n, n, k, -1);
    construction_matrix(d_G.get(), n, dim, nb, 0, D);
    b = knn_from_matrix(D.get(), n, n, k, -1);
  }
  // The radius is r_k under the primary distance, whatever nb is.
  knnResult p = mixed_knn(d_R.get(), n, dim, k, nb.dist);
  to_true_distance(p.distances, (size_t)n * k, nb.dist);
  DeviceArray<float> rk(n);
  column_kernel<<<grid_for(n), kBlock>>>(p.distances, n, k, k - 1, rk.get());
  CUDA_CHECK_KERNEL();

  drift_kernel<<<n, kBlock, 2 * k * sizeof(int)>>>(d_R.get(), n, dim, k, a.indices,
                                                    b.indices, rk.get(), b_pt);
  CUDA_CHECK_KERNEL();
  free_knn_result(a);
  free_knn_result(b);
  free_knn_result(p);
}

// ═══════════════════════════════════════════════════════════════════════════
// Superseded: the first property design.
//
// None of the three is simply an older form of a present property:
//
//   barycenter_shift  per-point distance from a query to the centroid of its k
//                     nearest REAL neighbors
//                     -> the second design replaced it with a global centroid
//                        displacement, since dropped as the mean term of the
//                        Frechet distance. GP6's drift reinstates a local
//                        barycenter, but compares two neighborhoods of the same
//                        clean point rather than a point with one neighborhood.
//
//   neighbor_overlap  unweighted fraction of a real point's k-NN that survive
//                     in the pooled graph
//                     -> GP6 trustworthiness and continuity, which weight each
//                        lost or intruding neighbor by how far out of place it
//                        ended up, and split the two error types that this
//                        number conflates.
//
//   compute_indegree  in-degree of each node in the pooled k-NN graph, reported
//                     as real-vs-synthetic means
//                     -> GP5 mixing, which asks the same separability question
//                        directly and normalizes by what random mixing would
//                        give, and GP3 hubness, which reads the in-degree
//                        distribution against the clean one.
//
// compute_lid is NOT here: it remains live above, called by lid_profile.
// ═══════════════════════════════════════════════════════════════════════════

// // cross-set
// __global__ void barycenter_shift_kernel(const float *base, const float *query, const int *neighbors, int n_q, int k, int dim, float* output) {
//   int qi = blockIdx.x * blockDim.x + threadIdx.x;
//   if (qi >= n_q) return;
//
//   // compute centroid in base
//   float shift_sq = 0.0f;
//   for (int d = 0; d < dim; d++) {
//     float centroid_d = 0.0f;
//     for (int i = 0; i < k; i++) {
//       int nbh = neighbors[qi * k + i];
//       centroid_d += base[nbh * dim + d];
//     }
//     centroid_d /= k;
//     float diff = query[qi * dim + d] - centroid_d;
//     shift_sq += diff * diff;
//   }
//   output[qi] = sqrtf(shift_sq);
// }
//
// void barycenter_shift(const float *base, const float *query, const int *neighbors, int n_q, int k, int dim, float* output) {
//   int block_size = 128;
//   int grid_size = (n_q + block_size - 1) / block_size;
//   barycenter_shift_kernel<<<grid_size, block_size>>>(base, query, neighbors, n_q, k, dim, output);
//   CUDA_CHECK_KERNEL();
// }
//
// // pooled
// __global__ void neighbor_overlap_kernel(const int *knn_real, const int *knn_pooled, int n_b, int k, float* output) {
//   int bi = blockIdx.x * blockDim.x + threadIdx.x;
//   if (bi >= n_b) return;
//
//   // count intersection of k-nn real and pooled
//   int cnt = 0;
//   for (int i = 0; i < k; i++) {
//     int real_id = knn_real[bi * k + i];
//     for (int j = 0; j < k; j++) {
//       if (knn_pooled[bi * k + j] == real_id) {
//         cnt += 1;
//         break;
//       }
//     }
//   }
//   output[bi] = (float)cnt / k;
// }
//
// void neighbor_overlap(const int* knn_real, const int* knn_pooled, int n_b, int k, float* output) {
//   int block_size = 256;
//   int grid_size = (n_b + block_size - 1) / block_size;
//   neighbor_overlap_kernel<<<grid_size, block_size>>>(knn_real, knn_pooled, n_b, k, output);
//   CUDA_CHECK_KERNEL();
// }
//
// // pooled
// __global__ void compute_indegree_kernel(const int *neighbors, int n, int k, int *output) {
//   int i = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i >= n) return;
//
//   for (int j = 0; j < k; j++) {
//     int nbh = neighbors[i * k + j];
//     if (0 <= nbh && nbh < n) {
//       atomicAdd(&output[nbh], 1);
//     }
//   }
// }
//
// void compute_indegree(const int *neighbors, int n, int k, int *output) {
//   cudaMemset(output, 0, n * sizeof(int));
//   int block_size = 256;
//   int grid_size = (n + block_size - 1) / block_size;
//   compute_indegree_kernel<<<grid_size, block_size>>>(neighbors, n, k, output);
//   CUDA_CHECK_KERNEL();
// }
