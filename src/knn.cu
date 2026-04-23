#include "knn.h"

#include <cstddef>
#include <cstdlib>
#include <cublas_v2.h>

// declared in pairwise_dist.cu
extern void pairwise_l2_dist(cublasHandle_t, const float*, const float*, int, int, int, float*);
extern void pairwise_ip_dist(cublasHandle_t, const float*, const float*, int, int, int, float*);
extern __global__ void pairwise_mahalanobis_kernel(const float*, const float*, const float*, float*, int, int, int);

// declared in search.cu
extern void topk_search_kernel(const float*, int*, float*, int, int, int, int);

knnResult cross_set_knn(
  const float *base, const float *query, 
  int n_b, int n_q, int dim, int k,
  DistanceType dist_type, const float* inv_cov
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float* dist_matrix;
  cudaMalloc(&dist_matrix, (size_t)n_q * n_b * sizeof(float));
  
  // compute pairwise distances
  if (dist_type == DistanceType::L2) {
    pairwise_l2_dist(handle, base, query, n_b, n_q, dim, dist_matrix);
  } else
  if (dist_type == DistanceType::InnerProduct) {
    pairwise_ip_dist(handle, base, query, n_b, n_q, dim, dist_matrix);
  } else
  if (dist_type == DistanceType::Mahalanobis) {
    if (inv_cov == nullptr) {
      abort();
    }
    int n_block = 16;
    dim3 block(n_block, n_block);
    dim3 grid((n_b + n_block - 1) / n_block, (n_q + n_block - 1) / n_block);
    pairwise_mahalanobis_kernel<<<grid, block>>>(base, query, inv_cov, dist_matrix, n_b, n_q, dim);
  } else {
    abort();
  }

  knnResult res;
  res.n_q = n_q;
  res.k = k;
  cudaMalloc(&res.indices, (size_t)n_q * k * sizeof(int));
  cudaMalloc(&res.distances, (size_t)n_q * k * sizeof(float));

  // top-k selection
  int block_size = 128;
  int grid_size = (n_q + block_size - 1) / block_size;
  topk_search_kernel<<<grid_size, block_size>>>(dist_matrix, res.indices, res.distances, n_b, n_q, k, -1);

  cudaFree(dist_matrix);
  cublasDestroy(handle);
  return res;
}

knnResult pooled_knn(
  const float* mixed, int n, int dim, int k,
  DistanceType dist_type, const float* inv_cov
) {
  cublasHandle_t handle;
  cublasCreate(&handle);
  float* dist_matrix;
  cudaMalloc(&dist_matrix, (size_t)n * n * sizeof(float));

  // compute self-distance matrix
  if (dist_type == DistanceType::L2) {
    pairwise_l2_dist(handle, mixed, mixed, n, n, dim, dist_matrix);
  } else
  if (dist_type == DistanceType::InnerProduct) {
    pairwise_ip_dist(handle, mixed, mixed, n, n, dim, dist_matrix);
  } else
  if (dist_type == DistanceType::Mahalanobis) {
    if (inv_cov == nullptr) {
      abort();
    }
    int n_block = 16;
    dim3 block(n_block, n_block);
    dim3 grid((n + n_block - 1) / n_block, (n + n_block - 1) / n_block);
    pairwise_mahalanobis_kernel<<<grid, block>>>(mixed, mixed, inv_cov, dist_matrix, n, n, dim);
  } else {
    abort();
  }

  knnResult res;
  res.n_q = n;
  res.k = k;
  cudaMalloc(&res.indices, (size_t)n * k * sizeof(int));
  cudaMalloc(&res.distances, (size_t)n * k * sizeof(float));

  int block_size = 128;
  int grid_size = (n + block_size - 1) / block_size;
  topk_search_kernel<<<grid_size, block_size>>>(dist_matrix, res.indices, res.distances, n, n, k, 0);

  cudaFree(dist_matrix);
  cublasDestroy(handle);
  return res;
}

void free_knn_result(knnResult &res) {
  if (res.indices) cudaFree(res.indices);
  if (res.distances) cudaFree(res.distances);
  res.indices = nullptr;
  res.distances = nullptr;
}
