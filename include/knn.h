#pragma once
#include <stdint.h>

enum class DistanceType {
  L2,             // squared Euclidean: ||a-b||^2
  InnerProduct,   // negative dot product (for cosine after normalization)
  Mahalanobis     // (a-b)^T Σ^{-1} (a-b), requires inv_cov matrix
};

struct knnResult {
  int* indices;       // (n_q, k) - neighbor indices, device memory
  float* distances;   // (n_q, k) - neighbor distances, device memory
  int n_q;
  int k;
};

// ─── k-NN index types ───

// Cross-set k-NN: for each query in Q, find k nearest neighbors in base
// Q: (n_q, dim), R: (n_b, dim), both device pointers, row-major float32
knnResult cross_set_knn (
  const float* base, const float* query, int n_b, int n_q, 
  int dim, int k, DistanceType dist_type = DistanceType::L2, 
  const float* inv_cov = nullptr // (dim, dim) for Mahalanobis
);

// Mixed k-NN: for each point in X, find k nearest neighbors in X (excl. self)
// X: (n_b + n_q, dim), device pointer, row-major float32
knnResult mixed_knn(
  const float* mixed, int n, int dim, int k,
  DistanceType dist_type = DistanceType::L2, const float* inv_cov = nullptr
);

// Free device memory
void free_knn_result(knnResult &res);

// Full pairwise distance matrix (n_q, n_b), device pointers.
void pairwise_distances(
  const float* base, const float* query, int n_b, int n_q, int dim,
  DistanceType dist_type, float* dist
);

// The five geometric properties live in geometry.h / graph.cu. The previous
// design's per-point kernels -- barycenter_shift, neighbor_overlap,
// compute_indegree -- and the mutual-k-NN CSR builder were declared here; the
// first three are kept commented at the end of graph.cu, the CSR builder is
// gone with mutual_graph.cu.
