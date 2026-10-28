#include "knn.h"
#include "geometry.h"

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
      if (!approx_eq(gd, cd, atol, 1e-3f)) {
        fprintf(stderr, "  row %d rank %d: gpu %.6f (idx %d) vs cpu %.6f (idx %d)\n", q,
                j, gd, gpu_idx[q * k + j], cd, cpu_idx[q * k + j]);
      }
      assert(approx_eq(gd, cd, atol, 1e-3f) &&
             "top-k distance rank mismatch with CPU reference");

      int gi = gpu_idx[q * k + j];
      assert(gi >= 0 && gi < n_b && "index out of range");
      float actual = D[q * n_b + gi];
      assert(approx_eq(actual, gd, atol, 1e-3f) &&
             "returned index does not correspond to returned distance");
    }
  }
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

void test_mixed_knn_excludes_self() {
  const int n = 15, dim = 4, k = 3;
  auto hX = rand_vec(n * dim, 33);
  float* dX = upload(hX);

  knnResult res = mixed_knn(dX, n, dim, k, DistanceType::L2);

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
      assert(gpu_idx[i * k + j] != i && "mixed_knn must exclude self");
    }
  }

  free_knn_result(res);
  cudaFree(dX);
  printf("test_mixed_knn_excludes_self PASSED\n");
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

void test_compute_lid() {
  // Levina-Bickel with MacKay & Ghahramani's bias correction, which is what the
  // kernel implements:  LID = -(k-2) / Σ_{i=0..k-2} ln(d_i / d_{k-1}).
  // The numerator is k-2, not k: this test asserted -k before the kernel was
  // corrected, and would have gone on asserting an 11%-inflated estimate at
  // k = 10.
  const int n_q = 2, k = 4;
  const int ld = 6;   // wider than k: P3 searches three neighbors for TwoNN
                      // whatever k is, and reads the first k of that row.
  std::vector<float> hD = {
      // row 0: all equal distances -> sum_log = 0 -> branch outputs 0
      0.5f, 0.5f, 0.5f, 0.5f, 0.6f, 0.7f,
      // row 1: varying
      0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
  };
  float* dD = upload(hD);
  float* dOut = nullptr;
  CUDA_CHECK(cudaMalloc(&dOut, n_q * sizeof(float)));

  compute_lid(dD, n_q, k, ld, dOut);
  std::vector<float> hOut(n_q);
  CUDA_CHECK(cudaMemcpy(hOut.data(), dOut, n_q * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Row 0: all k neighbors equidistant, LID undefined. Kernel outputs 0.
  assert(approx_eq(hOut[0], 0.0f));

  // Row 1: the columns beyond k must not have been read.
  float dk = hD[1 * ld + (k - 1)];
  float sum_log = 0.0f;
  for (int i = 0; i < k - 1; i++) {
    sum_log += std::log(hD[1 * ld + i] / dk);
  }
  float expected = -(float)(k - 2) / sum_log;
  assert(approx_eq(hOut[1], expected, 1e-3f, 1e-3f));

  cudaFree(dD);
  cudaFree(dOut);
  printf("test_compute_lid PASSED\n");
}

// Tests for barycenter_shift, neighbor_overlap, compute_indegree and
// build_mutual_knn were here. The first three cover the previous
// five-property design and are now commented out at the end of src/graph.cu;
// build_mutual_knn went with mutual_graph.cu. Nothing calls any of them.
// Superseded tests are removed rather than skipped so a green run means every
// kernel that ships is covered.
int main() {
  test_cross_set_knn_l2();
  test_cross_set_knn_inner_product();
  test_cross_set_knn_mahalanobis_identity();
  test_mixed_knn_excludes_self();
  test_free_knn_result_nullifies();

  test_compute_lid();

  printf("\nAll tests passed.\n");
  return 0;
}
