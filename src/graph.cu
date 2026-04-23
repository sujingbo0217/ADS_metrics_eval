#include "knn.h"
#include "utils.h"

#include <cmath>

// cross-set
__global__ void barycenter_shift_kernel(const float *base, const float *query, const int *neighbors, int n_q, int k, int dim, float* output) {
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= n_q) return;

  // compute centroid in base
  float shift_sq = 0.0f;
  for (int d = 0; d < dim; d++) {
    float centroid_d = 0.0f;
    for (int i = 0; i < k; i++) {
      int nbh = neighbors[qi * k + i];
      centroid_d += base[nbh * dim + d];
    }
    centroid_d /= k;
    float diff = query[qi * dim + d] - centroid_d;
    shift_sq += diff * diff;
  }
  output[qi] = sqrtf(shift_sq);
}

void barycenter_shift(const float *base, const float *query, const int *neighbors, int n_q, int k, int dim, float* output) {
  int block_size = 128;
  int grid_size = (n_q + block_size - 1) / block_size;
  barycenter_shift_kernel<<<grid_size, block_size>>>(base, query, neighbors, n_q, k, dim, output);
}

// pooled
__global__ void neighbor_overlap_kernel(const int *knn_real, const int *knn_pooled, int n_b, int k, float* output) {
  int bi = blockIdx.x * blockDim.x + threadIdx.x;
  if (bi >= n_b) return;

  // count intersection of k-nn real and pooled
  int cnt = 0;
  for (int i = 0; i < k; i++) {
    int real_id = knn_real[bi * k + i];
    for (int j = 0; j < k; j++) {
      if (knn_pooled[bi * k + j] == real_id) {
        cnt += 1;
        break;
      }
    }
  }
  output[bi] = (float)cnt / k;
}

void neighbor_overlap(const int* knn_real, const int* knn_pooled, int n_b, int k, float* output) {
  int block_size = 256;
  int grid_size = (n_b + block_size - 1) / block_size;
  neighbor_overlap_kernel<<<grid_size, block_size>>>(knn_real, knn_pooled, n_b, k, output);
}

// cross-set
__global__ void compute_lid_kernel(const float *dist, int n_q, int k, float *output) {
  // dist sorted ascending
  int qi = blockIdx.x * blockDim.x + threadIdx.x;
  if (qi >= n_q) return;

  float dk = dist[qi * k + (k - 1)];  // furthest distance
  constexpr float min_d = 1e-10f;
  constexpr float eps = 1e-6f;
  if (dk < min_d) {
    output[qi] = 0.0f;
    return;
  }
  
  // LID = -k / Σ ln(d_i / d_k)
  float sum_log = 0.0f;
  for (int i = 0; i < k - 1; i++) {
    float di = fmaxf(dist[qi * k + i], min_d);
    sum_log += logf(di / dk);
  }
  output[qi] = (sum_log < eps) ? (-(float)k / sum_log) : 0.0f;
}

void compute_lid(const float *dist, int n_q, int k, float *output) {
  int block_size = 256;
  int grid_size = (n_q + block_size - 1) / block_size;
  compute_lid_kernel<<<grid_size, block_size>>>(dist, n_q, k, output);
}

// pooled
__global__ void compute_indegree_kernel(const int *neighbors, int n, int k, int *output) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  for (int j = 0; j < k; j++) {
    int nbh = neighbors[i * k + j];
    if (0 <= nbh && nbh < n) {
      atomicAdd(&output[nbh], 1);
    }
  }
}

void compute_indegree(const int *neighbors, int n, int k, int *output) {
  cudaMemset(output, 0, n * sizeof(int));
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;
  compute_indegree_kernel<<<grid_size, block_size>>>(neighbors, n, k, output);
}
