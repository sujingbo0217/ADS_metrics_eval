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

// Pooled k-NN: for each point in X, find k nearest neighbors in X (excl. self)
// X: (n_b + n_q, dim), device pointer, row-major float32
knnResult pooled_knn(
  const float* mixed, int n, int dim, int k,
  DistanceType dist_type = DistanceType::L2, const float* inv_cov = nullptr
);

// Free device memory
void free_knn_result(knnResult &res);

// ─── Graph property functions ───

// Barycenter shift: for each query, distance from query to centroid of its k-NN in base
// Q: (n_q, dim), B: (n_b, dim), knn_indices: (n_q, k) from cross_set_knn
void barycenter_shift(
  const float* base, const float* query, const int* neighbors, 
  int n_q, int k, int dim, float* output // (n_q, ) output, device memory
);

// Neighborhood overlap: for real points, compare real-only k-NN vs pooled k-NN
// knn_real: (n_b, k) from base (real-image) set
// knn_pooled: (n, k) from pooled k-NN (only first n_b rows used)
void neighbor_overlap(
  const int* knn_real, const int* knn_pooled, // first n_b rows are real
  int n_b, int k, float* output // (n_b, ) output, device memory
);

// Local Intrinsic Dimensionality (LID) fromi cross-set distances
// dist: (n_q, k) sorted ascending, from cross_set_knn
void compute_lid(
  const float* dist, int n_q, int k,
  float* output // (n_q, ) output, device memory
);

// In-degree distribution from directed k-NN graph
// neighbors: (n, k)
void compute_indegree(
  const int* neighbors, int n, int k,
  int* output // (n, ) output, device memory
);

// Build mutual k-NN graph in CSR format
struct CSRGraph {
  int* row_ptr; // (row + 1, ) device memory
  int* col_idx; // (nnz, ) device memory
  int n;
  int nnz;      // number of non-zero entries (total edges)
};

CSRGraph build_mutual_knn(const int* neighbors, int n, int k);

void free_csr_graph(CSRGraph &g);
