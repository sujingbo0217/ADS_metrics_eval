#include "knn.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ─── Small test helpers ───────────────────────────────────────────────

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
              cudaGetErrorString(_err));                                      \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

static inline bool approx_eq(float a, float b, float atol = 1e-3f,
                             float rtol = 1e-3f) {
  float diff = std::fabs(a - b);
  float tol = atol + rtol * std::max(std::fabs(a), std::fabs(b));
  return diff <= tol;
}

static float* upload(const std::vector<float>& h) {
  float* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

static int* upload(const std::vector<int>& h) {
  int* d = nullptr;
  CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  return d;
}

static std::vector<float> rand_vec(int n, unsigned seed) {
  std::srand(seed);
  std::vector<float> v(n);
  for (int i = 0; i < n; i++) {
    v[i] = static_cast<float>(std::rand() % 1000) / 1000.0f;
  }
  return v;
}

// CPU reference: squared L2 distance matrix (row-major, n_q × n_b)
static std::vector<float> cpu_l2_dist_matrix(const std::vector<float>& base,
                                             const std::vector<float>& query,
                                             int n_b, int n_q, int dim) {
  std::vector<float> D((size_t)n_q * n_b);
  for (int q = 0; q < n_q; q++) {
    for (int b = 0; b < n_b; b++) {
      float s = 0.0f;
      for (int d = 0; d < dim; d++) {
        float diff = query[q * dim + d] - base[b * dim + d];
        s += diff * diff;
      }
      D[q * n_b + b] = s;
    }
  }
  return D;
}

// CPU reference: pick top-k smallest per row, optionally excluding self index.
// Returns (indices, dists) with indices sorted by ascending distance.
static void cpu_topk(const std::vector<float>& D, int n_q, int n_b, int k,
                     int self_offset /* -1 if none */,
                     std::vector<int>& out_idx,
                     std::vector<float>& out_dist) {
  out_idx.assign((size_t)n_q * k, -1);
  out_dist.assign((size_t)n_q * k, 0.0f);
  for (int q = 0; q < n_q; q++) {
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(n_b);
    int self = (self_offset >= 0) ? q + self_offset : -1;
    for (int b = 0; b < n_b; b++) {
      if (b == self) continue;
      pairs.emplace_back(D[q * n_b + b], b);
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<float, int>& a,
                 const std::pair<float, int>& b) {
                if (a.first != b.first) return a.first < b.first;
                return a.second < b.second;
              });
    for (int j = 0; j < k; j++) {
      out_idx[q * k + j] = pairs[j].second;
      out_dist[q * k + j] = pairs[j].first;
    }
  }
}

// Verify top-k result distances match CPU reference within tolerance.
// Index positions can differ on ties, but the distance value at each rank
// must match, and each returned index must actually correspond to that
// distance in the full distance matrix.
static void verify_topk(const knnResult& res, const std::vector<float>& D,
                        const std::vector<int>& cpu_idx,
                        const std::vector<float>& cpu_dist, int n_q, int n_b,
                        int k, float atol = 1e-3f) {
  std::vector<int> gpu_idx((size_t)n_q * k);
  std::vector<float> gpu_dist((size_t)n_q * k);
  CUDA_CHECK(cudaMemcpy(gpu_idx.data(), res.indices,
                        gpu_idx.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_dist.data(), res.distances,
                        gpu_dist.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  for (int q = 0; q < n_q; q++) {
    for (int j = 0; j < k; j++) {
      float gd = gpu_dist[q * k + j];
      float cd = cpu_dist[q * k + j];
      assert(approx_eq(gd, cd, atol, 1e-3f) &&
             "top-k distance rank mismatch with CPU reference");

      int gi = gpu_idx[q * k + j];
      assert(gi >= 0 && gi < n_b && "index out of range");
      float actual = D[q * n_b + gi];
      assert(approx_eq(actual, gd, atol, 1e-3f) &&
             "returned index does not correspond to returned distance");
    }
  }
  (void)cpu_idx;
}

// ─── Tests ─────────────────────────────────────────────────────────────

void test_cross_set_knn_l2() {
  const int n_b = 12, n_q = 7, dim = 4, k = 3;
  auto hB = rand_vec(n_b * dim, 1);
  auto hQ = rand_vec(n_q * dim, 2);

  float* dB = upload(hB);
  float* dQ = upload(hQ);
  knnResult res = cross_set_knn(dB, dQ, n_b, n_q, dim, k, DistanceType::L2);

  auto D = cpu_l2_dist_matrix(hB, hQ, n_b, n_q, dim);
  std::vector<int> ci;
  std::vector<float> cd;
  cpu_topk(D, n_q, n_b, k, /*self_offset=*/-1, ci, cd);

  verify_topk(res, D, ci, cd, n_q, n_b, k);

  free_knn_result(res);
  cudaFree(dB);
  cudaFree(dQ);
  printf("test_cross_set_knn_l2 PASSED\n");
}

void test_cross_set_knn_inner_product() {
  const int n_b = 10, n_q = 5, dim = 6, k = 4;
  auto hB = rand_vec(n_b * dim, 11);
  auto hQ = rand_vec(n_q * dim, 22);

  float* dB = upload(hB);
  float* dQ = upload(hQ);
  knnResult res =
      cross_set_knn(dB, dQ, n_b, n_q, dim, k, DistanceType::InnerProduct);

  // Reference: negative dot product (smaller = more similar).
  std::vector<float> D((size_t)n_q * n_b);
  for (int q = 0; q < n_q; q++) {
    for (int b = 0; b < n_b; b++) {
      float s = 0.0f;
      for (int d = 0; d < dim; d++) {
        s += hQ[q * dim + d] * hB[b * dim + d];
      }
      D[q * n_b + b] = -s;
    }
  }
  std::vector<int> ci;
  std::vector<float> cd;
  cpu_topk(D, n_q, n_b, k, -1, ci, cd);

  verify_topk(res, D, ci, cd, n_q, n_b, k);

  free_knn_result(res);
  cudaFree(dB);
  cudaFree(dQ);
  printf("test_cross_set_knn_inner_product PASSED\n");
}

void test_cross_set_knn_mahalanobis_identity() {
  // With inv_cov = I, Mahalanobis distance equals squared L2.
  const int n_b = 8, n_q = 6, dim = 3, k = 2;
  auto hB = rand_vec(n_b * dim, 7);
  auto hQ = rand_vec(n_q * dim, 9);
  std::vector<float> hI(dim * dim, 0.0f);
  for (int i = 0; i < dim; i++) hI[i * dim + i] = 1.0f;

  float* dB = upload(hB);
  float* dQ = upload(hQ);
  float* dI = upload(hI);

  knnResult res = cross_set_knn(dB, dQ, n_b, n_q, dim, k,
                                DistanceType::Mahalanobis, dI);

  auto D = cpu_l2_dist_matrix(hB, hQ, n_b, n_q, dim);
  std::vector<int> ci;
  std::vector<float> cd;
  cpu_topk(D, n_q, n_b, k, -1, ci, cd);

  verify_topk(res, D, ci, cd, n_q, n_b, k);

  free_knn_result(res);
  cudaFree(dB);
  cudaFree(dQ);
  cudaFree(dI);
  printf("test_cross_set_knn_mahalanobis_identity PASSED\n");
}

void test_pooled_knn_excludes_self() {
  const int n = 15, dim = 4, k = 3;
  auto hX = rand_vec(n * dim, 33);
  float* dX = upload(hX);

  knnResult res = pooled_knn(dX, n, dim, k, DistanceType::L2);

  auto D = cpu_l2_dist_matrix(hX, hX, n, n, dim);
  std::vector<int> ci;
  std::vector<float> cd;
  cpu_topk(D, n, n, k, /*self_offset=*/0, ci, cd);

  verify_topk(res, D, ci, cd, n, n, k);

  // Additionally confirm no row contains self as a neighbor.
  std::vector<int> gpu_idx((size_t)n * k);
  CUDA_CHECK(cudaMemcpy(gpu_idx.data(), res.indices,
                        gpu_idx.size() * sizeof(int),
                        cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < k; j++) {
      assert(gpu_idx[i * k + j] != i && "pooled_knn must exclude self");
    }
  }

  free_knn_result(res);
  cudaFree(dX);
  printf("test_pooled_knn_excludes_self PASSED\n");
}

void test_free_knn_result_nullifies() {
  knnResult res{};
  CUDA_CHECK(cudaMalloc(&res.indices, 4 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&res.distances, 4 * sizeof(float)));
  res.n_q = 2;
  res.k = 2;
  free_knn_result(res);
  assert(res.indices == nullptr && res.distances == nullptr);
  printf("test_free_knn_result_nullifies PASSED\n");
}

// ─── Graph property functions ───

void test_barycenter_shift() {
  // 3 queries, 4 base points, dim=2, k=2
  const int n_q = 3, dim = 2, k = 2;
  std::vector<float> hB = {
      0.0f, 0.0f,
      2.0f, 0.0f,
      0.0f, 2.0f,
      2.0f, 2.0f,
  };
  std::vector<float> hQ = {
      1.0f, 1.0f,   // centroid of any 2 of B; shift depends on neighbors
      0.0f, 0.0f,
      3.0f, 3.0f,
  };
  // Pick neighbors deterministically for each query.
  std::vector<int> hN = {
      0, 3,   // centroid = (1,1); shift from (1,1) = 0
      0, 1,   // centroid = (1,0); shift from (0,0) = 1
      2, 3,   // centroid = (1,2); shift from (3,3) = sqrt(4+1)
  };

  float* dB = upload(hB);
  float* dQ = upload(hQ);
  int* dN = upload(hN);
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, n_q * sizeof(float)));

  barycenter_shift(dB, dQ, dN, n_q, k, dim, dOut);

  std::vector<float> hOut(n_q);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, n_q * sizeof(float),
                        cudaMemcpyDeviceToHost));

  assert(approx_eq(hOut[0], 0.0f));
  assert(approx_eq(hOut[1], 1.0f));
  assert(approx_eq(hOut[2], std::sqrt(5.0f)));

  cudaFree(dB);
  cudaFree(dQ);
  cudaFree(dN);
  cudaFree(dOut);
  printf("test_barycenter_shift PASSED\n");
}

void test_neighbor_overlap() {
  const int n_b = 3, k = 3;
  std::vector<int> real = {
      1, 2, 3,
      0, 4, 5,
      7, 8, 9,
  };
  std::vector<int> pooled = {
      1, 2, 9,   // overlap {1,2} -> 2/3
      0, 4, 5,   // overlap {0,4,5} -> 3/3
      0, 1, 2,   // overlap {} -> 0/3
  };
  int* dR = upload(real);
  int* dP = upload(pooled);
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, n_b * sizeof(float)));

  neighbor_overlap(dR, dP, n_b, k, dOut);

  std::vector<float> hOut(n_b);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, n_b * sizeof(float),
                        cudaMemcpyDeviceToHost));

  assert(approx_eq(hOut[0], 2.0f / 3.0f));
  assert(approx_eq(hOut[1], 1.0f));
  assert(approx_eq(hOut[2], 0.0f));

  cudaFree(dR);
  cudaFree(dP);
  cudaFree(dOut);
  printf("test_neighbor_overlap PASSED\n");
}

void test_compute_lid() {
  // Hand-computed LID: formula LID = -k / Σ_{i=0..k-2} ln(d_i / d_{k-1})
  // (Matches the kernel, which iterates i in [0, k-1).)
  const int n_q = 2, k = 4;
  std::vector<float> hD = {
      // row 0: all equal distances -> sum_log = 0 -> branch outputs 0
      0.5f, 0.5f, 0.5f, 0.5f,
      // row 1: varying
      0.1f, 0.2f, 0.3f, 0.4f,
  };
  float* dD = upload(hD);
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, n_q * sizeof(float)));

  compute_lid(dD, n_q, k, dOut);
  std::vector<float> hOut(n_q);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, n_q * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Row 0: sum_log = 0, which is NOT < eps (1e-6). Kernel sets output = 0.
  assert(approx_eq(hOut[0], 0.0f));

  // Row 1: reference computation matching the kernel's loop range [0, k-1).
  float dk = 0.4f;
  float sum_log = 0.0f;
  for (int i = 0; i < k - 1; i++) {
    sum_log += std::log(hD[1 * k + i] / dk);
  }
  float expected = (sum_log < 1e-6f) ? (-(float)k / sum_log) : 0.0f;
  assert(approx_eq(hOut[1], expected, 1e-3f, 1e-3f));

  cudaFree(dD);
  cudaFree(dOut);
  printf("test_compute_lid PASSED\n");
}

void test_compute_indegree() {
  const int n = 5, k = 2;
  std::vector<int> hN = {
      1, 2,
      0, 2,
      3, 4,
      0, 1,
      2, 0,
  };
  // In-degree: count of occurrences in neighbor table.
  // 0: rows 1,3,4 -> 3
  // 1: rows 0,3   -> 2
  // 2: rows 0,1,4 -> 3
  // 3: row 2      -> 1
  // 4: row 2      -> 1
  int* dN = upload(hN);
  int* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, n * sizeof(int)));

  compute_indegree(dN, n, k, dOut);

  std::vector<int> hOut(n);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, n * sizeof(int),
                        cudaMemcpyDeviceToHost));

  std::vector<int> expected = {3, 2, 3, 1, 1};
  for (int i = 0; i < n; i++) {
    assert(hOut[i] == expected[i]);
  }

  cudaFree(dN);
  cudaFree(dOut);
  printf("test_compute_indegree PASSED\n");
}

void test_build_mutual_knn() {
  // n = 4, k = 2
  // Edges in k-NN (directed):
  //   0 -> {1, 2}
  //   1 -> {0, 2}
  //   2 -> {1, 3}
  //   3 -> {2, 0}
  // Mutual pairs (both directions):
  //   (0,1), (1,0)
  //   (1,2), (2,1)
  //   (2,3), (3,2)
  //   NOT (0,2) because 2 does not list 0, (0,3) not in 0's list, etc.
  const int n = 4, k = 2;
  std::vector<int> hN = {
      1, 2,
      0, 2,
      1, 3,
      2, 0,
  };
  int* dN = upload(hN);

  CSRGraph g = build_mutual_knn(dN, n, k);
  assert(g.n == n);

  std::vector<int> row_ptr(n + 1);
  CUDA_CHECK(cudaMemcpy(row_ptr.data(), g.row_ptr, (n + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost));
  std::vector<int> col_idx(g.nnz);
  if (g.nnz > 0) {
    CUDA_CHECK(cudaMemcpy(col_idx.data(), g.col_idx, g.nnz * sizeof(int),
                          cudaMemcpyDeviceToHost));
  }

  // Expected adjacency as sorted per-row sets.
  std::vector<std::vector<int>> expected = {
      {1},     // 0 <-> 1 only
      {0, 2},  // 1 <-> 0, 1 <-> 2
      {1, 3},  // 2 <-> 1, 2 <-> 3
      {2},     // 3 <-> 2
  };
  int expected_nnz = 0;
  for (auto& r : expected) expected_nnz += (int)r.size();
  assert(g.nnz == expected_nnz);

  assert(row_ptr[0] == 0);
  assert(row_ptr[n] == g.nnz);

  for (int u = 0; u < n; u++) {
    int lo = row_ptr[u], hi = row_ptr[u + 1];
    std::vector<int> row(col_idx.begin() + lo, col_idx.begin() + hi);
    std::sort(row.begin(), row.end());
    assert(row == expected[u] && "mutual knn row mismatch");
  }

  free_csr_graph(g);
  assert(g.row_ptr == nullptr && g.col_idx == nullptr);
  cudaFree(dN);
  printf("test_build_mutual_knn PASSED\n");
}

int main() {
  test_cross_set_knn_l2();
  test_cross_set_knn_inner_product();
  test_cross_set_knn_mahalanobis_identity();
  test_pooled_knn_excludes_self();
  test_free_knn_result_nullifies();

  test_barycenter_shift();
  test_neighbor_overlap();
  test_compute_lid();
  test_compute_indegree();
  test_build_mutual_knn();

  printf("\nAll tests passed.\n");
  return 0;
}
