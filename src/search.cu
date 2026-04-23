#include "knn.h"

constexpr float inf = 1e30f;

// ─── Per-row top-k via partial insertion sort ───
// Simple and correct for k <= 128. Each thread processes one row.
__global__ void topk_search_kernel(
  const float* dist,    // (n_q, n_b) distance matrix
  int* neighbors,       // (n_q, k) output neighbors
  float* distances,     // (n_q, k) output distances
  int n_b, int n_q, int k,
  int offset            // -1 for cross-set (no self-exclusion)
) {
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= n_q) return;

  // Maintain a max-heap of size k
  // For simplicity, use arrays in local memory
  // This is efficient for scale of k <= 128
  const float* row = dist + qi * n_b;
  int* nbhs = neighbors + qi * k;
  float* out_dist = distances + qi * k;
  
  for (int i = 0; i < k; i++) {
    nbhs[i] = -1;
    out_dist[i] = inf;
  }
  float max_d = inf;
  int max_pos = 0;
  int self_idx = (offset >= 0) ? qi + offset : -1;

  for (int i = 0; i < n_b; i++) {
    if (i == self_idx) continue;
    float d = row[i];
    if (d < max_d) {
      nbhs[max_pos] = i;
      out_dist[max_pos] = d;
      max_d = -1.0f;
      for (int j = 0; j < k; j++) {
        if (out_dist[j] > max_d) {
          max_d = out_dist[j];
          max_pos = j;
        }
      }
    }
  }

  // Sort the k results by distance
  for (int i = 1; i < k; i++) {
    int idx = nbhs[i];
    float d = out_dist[i];
    int j = i - 1;
    while (j >= 0 && out_dist[j] > d) {
      nbhs[j + 1] = nbhs[j];
      out_dist[j + 1] = out_dist[j];
      --j;
    }
    nbhs[j + 1] = idx;
    out_dist[j + 1] = d;
  }
}
