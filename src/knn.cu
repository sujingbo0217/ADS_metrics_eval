#include "knn.h"
#include "cuda_check.h"
#include "device_buffer.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

// declared in pairwise_dist.cu
extern void pairwise_l2_dist(cublasHandle_t, const float*, const float*, int, int, int, float*);
extern void pairwise_ip_dist(cublasHandle_t, const float*, const float*, int, int, int, float*);
extern __global__ void pairwise_mahalanobis_kernel(const float*, const float*, const float*, float*, int, int, int);

// declared in search.cu
extern __global__ void topk_search_kernel(const float*, int*, float*, int, int, int, int);

// The distance dispatch was copied verbatim between cross_set_knn and
// mixed_knn, which differ only in self-exclusion. One copy, two callers.
static void fill_distance_matrix(
  const float* base, const float* query, int n_b, int n_q, int dim,
  DistanceType dist_type, const float* inv_cov, float* dist
) {
  if (dist_type == DistanceType::Mahalanobis) {
    if (inv_cov == nullptr) {
      std::fprintf(stderr, "[knn_ext] Mahalanobis requires inv_cov\n");
      std::abort();
    }
    const int n_block = 16;
    dim3 block(n_block, n_block);
    dim3 grid((n_b + n_block - 1) / n_block, (n_q + n_block - 1) / n_block);
    pairwise_mahalanobis_kernel<<<grid, block>>>(base, query, inv_cov, dist, n_b, n_q, dim);
    CUDA_CHECK_KERNEL();
    return;
  }
  cublasHandle_t handle = knn_cublas_handle();
  if (dist_type == DistanceType::L2) {
    pairwise_l2_dist(handle, base, query, n_b, n_q, dim, dist);
  } else if (dist_type == DistanceType::InnerProduct) {
    pairwise_ip_dist(handle, base, query, n_b, n_q, dim, dist);
  } else {
    std::fprintf(stderr, "[knn_ext] unknown DistanceType %d\n", (int)dist_type);
    std::abort();
  }
}

// offset = -1 for a cross-set search, 0 for a self-query that must skip itself.
static knnResult topk(
  const float* base, const float* query, int n_b, int n_q, int dim, int k,
  DistanceType dist_type, const float* inv_cov, int offset
) {
  // A row has n_b candidates, one fewer when the query is its own base row.
  // Above that, topk_search_kernel leaves the tail of the row at (-1, +inf)
  // and every consumer reads the padding as a neighbor.
  const int avail = (offset >= 0) ? n_b - 1 : n_b;
  if (k < 1 || k > avail) {
    std::fprintf(stderr, "[knn_ext] k = %d out of range: %d candidate(s) per row\n",
                 k, avail);
    std::abort();
  }

  DeviceArray<float> dist((size_t)n_q * n_b);
  fill_distance_matrix(base, query, n_b, n_q, dim, dist_type, inv_cov, dist.get());

  knnResult res;
  res.n_q = n_q;
  res.k = k;
  CUDA_CHECK(cudaMalloc(&res.indices, (size_t)n_q * k * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&res.distances, (size_t)n_q * k * sizeof(float)));

  const int block_size = 128;
  const int grid_size = (n_q + block_size - 1) / block_size;
  topk_search_kernel<<<grid_size, block_size>>>(
      dist.get(), res.indices, res.distances, n_b, n_q, k, offset);
  CUDA_CHECK_KERNEL();
  return res;
}

knnResult cross_set_knn(
  const float *base, const float *query,
  int n_b, int n_q, int dim, int k,
  DistanceType dist_type, const float* inv_cov
) {
  return topk(base, query, n_b, n_q, dim, k, dist_type, inv_cov, -1);
}

knnResult mixed_knn(
  const float* mixed, int n, int dim, int k,
  DistanceType dist_type, const float* inv_cov
) {
  return topk(mixed, mixed, n, n, dim, k, dist_type, inv_cov, 0);
}

void free_knn_result(knnResult &res) {
  if (res.indices) cudaFree(res.indices);
  if (res.distances) cudaFree(res.distances);
  res.indices = nullptr;
  res.distances = nullptr;
}
