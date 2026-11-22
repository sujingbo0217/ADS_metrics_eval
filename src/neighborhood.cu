//   construction_matrix     Primary, SNN, local scaling, geodesic
//   neighbor_lists          k-NN under a construction
//   neighborhood_quality    Q1 hubness, Q2 components, Q3 LID coherence,
//                           Q4 subsampling stability
//   k_occurrence            GP3 hubness h_c
//   component_counts        GP4 connectivity tau_c

#include "geometry_common.cuh"
#include "cuda_check.h"

#include <cub/cub.cuh>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// ───────────────────────────────────────────────────────────────────────────
// Primary distance
// ───────────────────────────────────────────────────────────────────────────

__global__ void true_distance_kernel(float* d, size_t total, int is_ip) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = idx; i < total; i += (size_t)gridDim.x * blockDim.x) {
    float v = d[i];
    d[i] = is_ip ? sqrtf(fmaxf(2.0f + 2.0f * v, 0.0f)) : sqrtf(fmaxf(v, 0.0f));
  }
}

void to_true_distance(float* d, size_t total, DistanceType dist_type) {
  true_distance_kernel<<<grid_for(total), kBlock>>>(
      d, total, dist_type == DistanceType::InnerProduct);
  CUDA_CHECK_KERNEL();
}

// The GEMM-based distance is not exactly symmetric: the norms are added to
// -2 <a,b> in a different order for (i, j) and (j, i). The geodesic graph is
// built from both directions, so the two must agree.
__global__ void symmetrize_min_kernel(float* D, int N) {
  const size_t total = (size_t)N * N;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t e = idx; e < total; e += (size_t)gridDim.x * blockDim.x) {
    size_t i = e / N, j = e % N;
    if (i < j) {
      float m = fminf(D[e], D[j * N + i]);
      D[e] = m;
      D[j * N + i] = m;
    }
  }
}

__global__ void mask_pairs_kernel(float* D, int N, int pair) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  D[(size_t)i * N + i] = INFINITY;
  if (pair > 0 && i < 2 * pair) {
    int j = i < pair ? i + pair : i - pair;
    D[(size_t)i * N + j] = INFINITY;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Local scaling
// ───────────────────────────────────────────────────────────────────────────

__global__ void local_scaling_kernel(float* D, int N, const float* __restrict__ knn_d,
                                     int p) {
  const size_t total = (size_t)N * N;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t e = idx; e < total; e += (size_t)gridDim.x * blockDim.x) {
    float v = D[e];
    if (!(v < 1e30f)) continue;
    float si = knn_d[(e / N) * p + p - 1], sj = knn_d[(e % N) * p + p - 1];
    D[e] = v / sqrtf(fmaxf(si * sj, 1e-30f));
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Shared nearest neighbors
// ───────────────────────────────────────────────────────────────────────────

__global__ void adjacency_kernel(const int* __restrict__ idx, int N, int p, float* A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  for (int t = 0; t < p; t++) {
    int j = idx[(size_t)i * p + t];
    if (j >= 0) A[(size_t)i * N + j] = 1.0f;
  }
}

// Largest finite entry of each row. One block per row.
__global__ void row_max_finite_kernel(const float* __restrict__ D, int N, float* out) {
  const float* row = D + (size_t)blockIdx.x * N;
  float m = 0.0f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    float v = row[j];
    if (v < 1e30f) m = fmaxf(m, v);
  }
  typedef cub::BlockReduce<float, kBlock> Reduce;
  __shared__ typename Reduce::TempStorage tmp;
  float tot = Reduce(tmp).Reduce(m, cub::Max());
  if (threadIdx.x == 0) out[blockIdx.x] = tot;
}

// key = (p - shared) + d / (rowmax (1 + 1e-6)). The shared count is an integer
// and the tie-break lies in [0, 1), so the SNN order is exact and the primary
// distance only orders points with the same count -- which, with integer
// counts in 0..p, is most of them.
__global__ void snn_key_kernel(float* D, const float* __restrict__ C, int N, int p,
                               const float* __restrict__ rowmax) {
  const size_t total = (size_t)N * N;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t e = idx; e < total; e += (size_t)gridDim.x * blockDim.x) {
    float v = D[e];
    if (!(v < 1e30f)) continue;
    float scale = rowmax[e / N] * (1.0f + 1e-6f) + 1e-30f;
    D[e] = ((float)p - C[e]) + v / scale;
  }
}

// ───────────────────────────────────────────────────────────────────────────
// Geodesic: blocked Floyd-Warshall on the symmetrized k_g-NN graph
// ───────────────────────────────────────────────────────────────────────────

static constexpr int kTile = 32;

__global__ void fill_kernel(float* x, size_t total, float v) {
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t e = idx; e < total; e += (size_t)gridDim.x * blockDim.x) x[e] = v;
}

__global__ void zero_diagonal_kernel(float* W, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < P) W[(size_t)i * P + i] = 0.0f;
}

// Both directions get the same weight because D was symmetrized, so the race
// between rows i and j writing the same entry is benign.
__global__ void edge_kernel(const int* __restrict__ idx, const float* __restrict__ d,
                            int N, int p, float* W, int P) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  for (int t = 0; t < p; t++) {
    int j = idx[(size_t)i * p + t];
    if (j < 0) continue;
    float w = d[(size_t)i * p + t];
    W[(size_t)i * P + j] = w;
    W[(size_t)j * P + i] = w;
  }
}

// Phase 1: the pivot tile on its own. Entry (i, k) and (k, j) do not change in
// step k because the diagonal is 0, so one barrier per step suffices.
__global__ void fw_pivot_kernel(float* W, int P, int b) {
  __shared__ float t[kTile][kTile];
  const int i = threadIdx.y, j = threadIdx.x;
  const size_t g = (size_t)(b * kTile + i) * P + b * kTile + j;
  t[i][j] = W[g];
  __syncthreads();
  for (int k = 0; k < kTile; k++) {
    t[i][j] = fminf(t[i][j], t[i][k] + t[k][j]);
    __syncthreads();
  }
  W[g] = t[i][j];
}

// Phase 2: the pivot row (blockIdx.y = 0) and pivot column (1) tiles.
__global__ void fw_cross_kernel(float* W, int P, int b) {
  const int tile = blockIdx.x;
  if (tile == b) return;
  __shared__ float piv[kTile][kTile];
  __shared__ float cur[kTile][kTile];
  const int i = threadIdx.y, j = threadIdx.x;
  piv[i][j] = W[(size_t)(b * kTile + i) * P + b * kTile + j];
  const bool row = blockIdx.y == 0;
  const size_t g = row ? (size_t)(b * kTile + i) * P + tile * kTile + j
                       : (size_t)(tile * kTile + i) * P + b * kTile + j;
  cur[i][j] = W[g];
  __syncthreads();
  for (int k = 0; k < kTile; k++) {
    float v = row ? piv[i][k] + cur[k][j] : cur[i][k] + piv[k][j];
    cur[i][j] = fminf(cur[i][j], v);
    __syncthreads();
  }
  W[g] = cur[i][j];
}

// Phase 3: every other tile, from its row and column partners, no dependency.
__global__ void fw_rest_kernel(float* W, int P, int b) {
  const int bx = blockIdx.x, by = blockIdx.y;
  if (bx == b || by == b) return;
  __shared__ float rt[kTile][kTile];
  __shared__ float ct[kTile][kTile];
  const int i = threadIdx.y, j = threadIdx.x;
  rt[i][j] = W[(size_t)(by * kTile + i) * P + b * kTile + j];
  ct[i][j] = W[(size_t)(b * kTile + i) * P + bx * kTile + j];
  __syncthreads();
  const size_t g = (size_t)(by * kTile + i) * P + bx * kTile + j;
  float best = W[g];
  #pragma unroll
  for (int k = 0; k < kTile; k++) best = fminf(best, rt[i][k] + ct[k][j]);
  W[g] = best;
}

__global__ void unpad_kernel(const float* __restrict__ W, int P, float* D, int N) {
  const size_t total = (size_t)N * N;
  const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t e = idx; e < total; e += (size_t)gridDim.x * blockDim.x) {
    float v = W[(e / N) * P + e % N];
    D[e] = isinf(v) ? kUnreachable : v;
  }
}

static void geodesic(DeviceArray<float>& D, int N, int p) {
  knnResult g = knn_from_matrix(D.get(), N, N, p, -1);
  const int tiles = (N + kTile - 1) / kTile;
  const int P = tiles * kTile;
  DeviceArray<float> W((size_t)P * P);
  fill_kernel<<<grid_for((size_t)P * P), kBlock>>>(W.get(), (size_t)P * P, INFINITY);
  CUDA_CHECK_KERNEL();
  zero_diagonal_kernel<<<grid_for(P), kBlock>>>(W.get(), P);
  CUDA_CHECK_KERNEL();
  edge_kernel<<<grid_for(N), kBlock>>>(g.indices, g.distances, N, p, W.get(), P);
  CUDA_CHECK_KERNEL();
  free_knn_result(g);

  dim3 block(kTile, kTile);
  for (int b = 0; b < tiles; b++) {
    fw_pivot_kernel<<<1, block>>>(W.get(), P, b);
    CUDA_CHECK(cudaGetLastError());
    fw_cross_kernel<<<dim3(tiles, 2), block>>>(W.get(), P, b);
    CUDA_CHECK(cudaGetLastError());
    fw_rest_kernel<<<dim3(tiles, tiles), block>>>(W.get(), P, b);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK_KERNEL();
  unpad_kernel<<<grid_for((size_t)N * N), kBlock>>>(W.get(), P, D.get(), N);
  CUDA_CHECK_KERNEL();
}

// ───────────────────────────────────────────────────────────────────────────
// Dispatch
// ───────────────────────────────────────────────────────────────────────────

void construction_matrix(const float* d_X, int N, int dim, const Neighborhood& nb,
                         int pair, DeviceArray<float>& D) {
  const size_t NN = (size_t)N * N;
  D.alloc(NN);
  pairwise_distances(d_X, d_X, N, N, dim, nb.dist, D.get());
  to_true_distance(D.get(), NN, nb.dist);
  symmetrize_min_kernel<<<grid_for(NN), kBlock>>>(D.get(), N);
  CUDA_CHECK_KERNEL();
  mask_pairs_kernel<<<grid_for(N), kBlock>>>(D.get(), N, pair);
  CUDA_CHECK_KERNEL();

  const int p = nb.param;
  if (nb.kind != Construction::Primary && (p < 1 || p > N - 2)) {
    std::fprintf(stderr, "[knn_ext] construction parameter %d out of range for %d points\n",
                 p, N);
    std::abort();
  }

  switch (nb.kind) {
    case Construction::Primary:
      break;

    case Construction::LocalScaling: {
      knnResult s = knn_from_matrix(D.get(), N, N, p, -1);
      local_scaling_kernel<<<grid_for(NN), kBlock>>>(D.get(), N, s.distances, p);
      CUDA_CHECK_KERNEL();
      free_knn_result(s);
      break;
    }

    case Construction::SharedNeighbors: {
      knnResult s = knn_from_matrix(D.get(), N, N, p, -1);
      DeviceArray<float> A(NN), C(NN), rowmax(N);
      A.zero();
      adjacency_kernel<<<grid_for(N), kBlock>>>(s.indices, N, p, A.get());
      CUDA_CHECK_KERNEL();
      free_knn_result(s);
      // Row-major A is column-major A^T; C = A A^T is symmetric, so either
      // layout reads the same counts. Exact in float: they are at most p.
      const float one = 1.0f, zero = 0.0f;
      CUBLAS_CHECK(cublasSgemm(knn_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, N, N, N,
                               &one, A.get(), N, A.get(), N, &zero, C.get(), N));
      row_max_finite_kernel<<<N, kBlock>>>(D.get(), N, rowmax.get());
      CUDA_CHECK_KERNEL();
      snn_key_kernel<<<grid_for(NN), kBlock>>>(D.get(), C.get(), N, p, rowmax.get());
      CUDA_CHECK_KERNEL();
      break;
    }

    case Construction::Geodesic:
      geodesic(D, N, p);
      // The counterpart may be reachable through the graph; mask it again.
      mask_pairs_kernel<<<grid_for(N), kBlock>>>(D.get(), N, pair);
      CUDA_CHECK_KERNEL();
      break;
  }
}

void neighbor_lists_host(const float* d_X, int N, int dim, int k, const Neighborhood& nb,
                         int pair, int* idx_host) {
  DeviceArray<float> D;
  construction_matrix(d_X, N, dim, nb, pair, D);
  knnResult r = knn_from_matrix(D.get(), N, N, k, -1);
  CUDA_CHECK(cudaMemcpy(idx_host, r.indices, (size_t)N * k * sizeof(int),
                        cudaMemcpyDeviceToHost));
  free_knn_result(r);
}

void neighbor_lists(const float* X, int n, int dim, int k, const Neighborhood& nb,
                    int* idx) {
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);
  DeviceArray<float> D;
  construction_matrix(d_X.get(), n, dim, nb, 0, D);
  knnResult r = knn_from_matrix(D.get(), n, n, k, -1);
  CUDA_CHECK(cudaMemcpy(idx, r.indices, (size_t)n * k * sizeof(int),
                        cudaMemcpyDeviceToDevice));
  free_knn_result(r);
}

// ───────────────────────────────────────────────────────────────────────────
// GP3 k-occurrence, GP4 components
// ───────────────────────────────────────────────────────────────────────────

static std::vector<int> occurrences(const std::vector<int>& idx, int n, int k) {
  std::vector<int> occ(n, 0);
  for (size_t e = 0; e < (size_t)n * k; e++) {
    if (idx[e] >= 0) occ[idx[e]]++;
  }
  return occ;
}

void k_occurrence(const float* X, int n, int dim, int k, const Neighborhood& nb,
                  int* occ) {
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);
  std::vector<int> idx((size_t)n * k);
  neighbor_lists_host(d_X.get(), n, dim, k, nb, 0, idx.data());
  std::vector<int> h = occurrences(idx, n, k);
  CUDA_CHECK(cudaMemcpy(occ, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
}

// Components of the graph with an undirected edge i - idx[i][t], t < k, read
// from lists of width ld >= k. Union-find with path halving.
static int count_components(const std::vector<int>& idx, int n, int ld, int k) {
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  int comps = n;
  for (int i = 0; i < n; i++) {
    for (int t = 0; t < k; t++) {
      int j = idx[(size_t)i * ld + t];
      if (j < 0) continue;
      int a = find(i), b = find(j);
      if (a != b) {
        parent[a] = b;
        comps--;
      }
    }
  }
  return comps;
}

void component_counts(const float* X, int n, int dim, const Neighborhood& nb,
                      const int* ks, int n_ks, int* beta0) {
  int kmax = 0;
  for (int t = 0; t < n_ks; t++) kmax = std::max(kmax, ks[t]);
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);
  // The lists are sorted, so the k-NN graph for every k <= kmax is a prefix of
  // one search.
  std::vector<int> idx((size_t)n * kmax);
  neighbor_lists_host(d_X.get(), n, dim, kmax, nb, 0, idx.data());
  for (int t = 0; t < n_ks; t++) beta0[t] = count_components(idx, n, kmax, ks[t]);
}

// ───────────────────────────────────────────────────────────────────────────
// Q1--Q4
// ───────────────────────────────────────────────────────────────────────────

// Population skewness. The mean of O_k is k by construction.
static double skewness(const std::vector<int>& x) {
  const double n = (double)x.size();
  double m = 0.0;
  for (int v : x) m += v;
  m /= n;
  double m2 = 0.0, m3 = 0.0;
  for (int v : x) {
    double d = v - m;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= n;
  m3 /= n;
  return m2 > 0.0 ? m3 / std::pow(m2, 1.5) : 0.0;
}

NeighborhoodQuality neighborhood_quality(const float* X, int n, int dim, int k,
                                         const Neighborhood& nb, int n_draws,
                                         double drop, unsigned seed) {
  NeighborhoodQuality q{};
  DeviceArray<float> d_X;
  d_X.from_host(X, (size_t)n * dim);
  std::vector<int> idx((size_t)n * k);
  neighbor_lists_host(d_X.get(), n, dim, k, nb, 0, idx.data());

  // Q1, Q2
  q.hub_skew = skewness(occurrences(idx, n, k));
  q.components = count_components(idx, n, k, k);

  // Q3: the pointwise LID is always the primary-distance estimate; only the
  // neighborhood over which its spread is taken depends on the construction.
  {
    DeviceArray<float> lid(n);
    lid_profile(X, n, dim, k, nb.dist, lid.get(), nullptr);
    std::vector<float> h(n);
    lid.to_host(h.data());
    double acc = 0.0;
    int used = 0;
    for (int i = 0; i < n; i++) {
      double s = 0.0, s2 = 0.0;
      int c = 0;
      auto add = [&](int j) {
        float v = h[j];
        if (std::isfinite(v) && v > 0.0f) {
          s += v;
          s2 += (double)v * v;
          c++;
        }
      };
      add(i);
      for (int t = 0; t < k; t++) {
        if (idx[(size_t)i * k + t] >= 0) add(idx[(size_t)i * k + t]);
      }
      if (c < 2) continue;
      double mean = s / c;
      double var = std::max(s2 / c - mean * mean, 0.0);
      if (mean > 0.0) {
        acc += std::sqrt(var) / mean;
        used++;
      }
    }
    q.lid_cv = used ? acc / used : NAN;
  }

  // Q4
  {
    std::mt19937 rng(seed);
    const int keep = n - (int)std::lround(drop * n);
    if (keep <= k + 1) {
      q.stability = NAN;
      return q;
    }
    std::vector<int> perm(n), pos(n);
    std::vector<float> sub((size_t)keep * dim);
    std::vector<int> sidx((size_t)keep * k);
    double acc = 0.0;
    long long used = 0;
    for (int d = 0; d < n_draws; d++) {
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), rng);
      std::sort(perm.begin(), perm.begin() + keep);
      std::fill(pos.begin(), pos.end(), -1);
      for (int s = 0; s < keep; s++) {
        pos[perm[s]] = s;
        std::copy(X + (size_t)perm[s] * dim, X + (size_t)(perm[s] + 1) * dim,
                  sub.begin() + (size_t)s * dim);
      }
      DeviceArray<float> d_sub;
      d_sub.from_host(sub.data(), sub.size());
      neighbor_lists_host(d_sub.get(), keep, dim, k, nb, 0, sidx.data());
      for (int s = 0; s < keep; s++) {
        const int i = perm[s];
        int surv = 0, kept = 0;
        for (int t = 0; t < k; t++) {
          int j = idx[(size_t)i * k + t];
          if (j < 0 || pos[j] < 0) continue;
          surv++;
          for (int u = 0; u < k; u++) {
            if (sidx[(size_t)s * k + u] == pos[j]) {
              kept++;
              break;
            }
          }
        }
        if (surv > 0) {
          acc += (double)kept / surv;
          used++;
        }
      }
    }
    q.stability = used ? acc / used : NAN;
  }
  return q;
}
