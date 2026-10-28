#include "knn.h"
#include "cuda_check.h"
#include "device_buffer.h"
#include "utils.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdlib>

// Created on first use and kept. See the note in device_buffer.h.
cublasHandle_t knn_cublas_handle() {
  static cublasHandle_t h = nullptr;
  if (!h) CUBLAS_CHECK(cublasCreate(&h));
  return h;
}

// ─── Precompute squared norms ───
__global__ void compute_norms_kernel(const float* X, float* norms, int n, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    float v = X[i * dim + d];
    sum += v * v;
  }
  norms[i] = sum;
}

// ─── Fuse norms into GEMM result to get L2 distances ───
// dist[i][j] = norms_b[i] + norms_q[j] - 2 * gemm[i][j] (alpha = -2)
__global__ void fuse_l2_kernel(
  float* dist,            // (n_q, n_b) - in-place, initially holds -2 * Q @ B^T
  const float* norms_b,
  const float* norms_q,
  int n_b, int n_q
) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_q || j >= n_b) return;
  dist[i * n_b + j] += norms_q[i] + norms_b[j];
}

// ─── Full pairwise L2 distance matrix ───
// Output: dist (n_q, n_b) on device
void pairwise_l2_dist(
  cublasHandle_t handle, const float* base, const float* query,
  int n_b, int n_q, int dim, float* dist
) {
  // compute norms
  float *norms_b, *norms_q;
  cudaMalloc(&norms_b, n_b * sizeof(float));
  cudaMalloc(&norms_q, n_q * sizeof(float));
  int block_size = 256;
  // Each launch must be sized from ITS OWN row count. Both used (n_b + ...)
  // before, which silently under-covered the query side whenever n_q > n_b --
  // harmless for mixed_knn where n_q == n_b, wrong for cross_set_knn.
  compute_norms_kernel<<<(n_b + block_size - 1) / block_size, block_size>>>(
      base, norms_b, n_b, dim);
  CUDA_CHECK_KERNEL();
  compute_norms_kernel<<<(n_q + block_size - 1) / block_size, block_size>>>(
      query, norms_q, n_q, dim);
  CUDA_CHECK_KERNEL();

  // GEMM -> dist
  float alpha = -2.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(
    handle, CUBLAS_OP_T, CUBLAS_OP_N, // B^T, Q
    n_b, n_q, dim,                     // m, n, k
    &alpha,
    base, dim,                        // lda = dim
    query, dim,                       // ldb = dim
    &beta,
    dist, n_b                         // ldc = n_b
  ));

  // fuse norms
  dim3 block2(16, 16);
  dim3 grid2((n_b + 15) / 16, (n_q + 15) / 16);
  fuse_l2_kernel<<<grid2, block2>>>(dist, norms_b, norms_q, n_b, n_q);
  CUDA_CHECK_KERNEL();

  CUDA_CHECK(cudaFree(norms_b));
  CUDA_CHECK(cudaFree(norms_q));
}

// ─── Inner-product distance (cosine after normalization) ───
// dist[i][j] = -dot(B[i], Q[i]) - smaller = more similar
void pairwise_ip_dist(
  cublasHandle_t handle, const float* base, const float* query,
  int n_b, int n_q, int dim, float* dist
) {
  float alpha = -1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n_b, n_q, dim, &alpha,
                           base, dim, query, dim, &beta, dist, n_b));
}

// ─── Row-wise L2 normalization (in-place) ───
__global__ void l2_normalize_rows_kernel(float* X, int n, int dim) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float sum = 0.0f;
  for (int d = 0; d < dim; d++) {
    float v = X[i * dim + d];
    sum += v * v;
  }
  float inv_norm = rsqrtf(fmaxf(sum, 1e-12f));
  for (int d = 0; d < dim; d++) {
    X[i * dim + d] *= inv_norm;
  }
}

void l2_normalize_rows(float* X, int n, int dim) {
  int block_size = 64;
  int grid_size = (n + block_size - 1) / block_size;
  l2_normalize_rows_kernel<<<grid_size, block_size>>>(X, n, dim);
  CUDA_CHECK_KERNEL();
}

__global__ void pairwise_mahalanobis_kernel(const float* base, const float* query, const float* inv_cov, float* dist, int n_b, int n_q, int dim) {
  int qi = blockIdx.y * blockDim.y + threadIdx.y;
  int bi = blockIdx.x * blockDim.x + threadIdx.x;
  if (bi >= n_b || qi >= n_q) return;

  // compute diff = query[qi] - base[bi], then diff^T @ inv_cov @ diff
  float res = 0.0f;
  for (int d1 = 0; d1 < dim; d1++) {
    float diff_d1 = query[qi * dim + d1] - base[bi * dim + d1];
    float inner = 0.0f;
    for (int d2 = 0; d2 < dim; d2++) {
      float diff_d2 = query[qi * dim + d2] - base[bi * dim + d2];
      inner += inv_cov[d1 * dim + d2] * diff_d2;
    }
    res += diff_d1 * inner;
  }
  dist[qi * n_b + bi] = res;
}

// Full pairwise distance matrix
// Output: dist (n_q, n_b), row-major, device memory. L2 returns SQUARED
// Euclidean and InnerProduct returns NEGATIVE dot product, matching
// cross_set_knn and mixed_knn. Both are monotone in true distance, so ranks
// are unaffected; true_distance_kernel converts where a magnitude is needed.
void pairwise_distances(
  const float* base, const float* query, int n_b, int n_q, int dim,
  DistanceType dist_type, float* dist
) {
  cublasHandle_t handle = knn_cublas_handle();
  if (dist_type == DistanceType::L2) {
    pairwise_l2_dist(handle, base, query, n_b, n_q, dim, dist);
  } else if (dist_type == DistanceType::InnerProduct) {
    pairwise_ip_dist(handle, base, query, n_b, n_q, dim, dist);
  } else {
    std::fprintf(stderr, "[knn_ext] pairwise_distances: Mahalanobis needs inv_cov, "
                         "use cross_set_knn or mixed_knn\n");
    abort();
  }
}
