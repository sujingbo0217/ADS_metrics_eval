#include "knn.h"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

// count mutual edges per node
__global__ void count_mutual_edges_kernel(const int* neighbors, int* edge_counts, int n, int k) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;

  int cnt = 0;
  for (int i = 0; i < k; i++) {
    int v = neighbors[u * k + i];
    if (v < 0 || v >= n || v == u) continue;
    bool is_mutual = false;
    for (int j = 0; j < k; j++) {
      if (neighbors[v * k + j] == u) {
        is_mutual = true;
        break;
      }
    }
    if (is_mutual) {
      cnt += 1;
    }
  }
  edge_counts[u] = cnt;
}

// fill CSR col_idx
__global__ void fill_mutual_edges_kernel(
  const int* neighbors,
  const int* row_ptr,     // (n + 1, ) - prefix sum of edge_counts
  int* col_idx,           // output
  int n, int k
) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n) return;

  // col_idx[row_ptr[u] : row_ptr[u+1]]
  int offset = row_ptr[u];
  for (int i = 0; i < k; i++) {
    int v = neighbors[u * k + i];
    if (v < 0 || v >= n || v == u) continue;
    bool is_mutual = false;
    for (int j = 0; j < k; j++) {
      if (neighbors[v * k + j] == u) {
        is_mutual = true;
        break;
      }
    }
    if (is_mutual) {
      col_idx[offset] = v;
      ++offset;
    }
  }
}

CSRGraph build_mutual_knn(const int *neighbors, int n, int k) {
  CSRGraph g;
  g.n = n;
  g.nnz = 0;
  g.row_ptr = nullptr;
  g.col_idx = nullptr;

  cudaMalloc(&g.row_ptr, (n + 1) * sizeof(int));

  if (n == 0) {
    cudaMemset(g.row_ptr, 0, sizeof(int));
    return g;
  }

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  // count mutual edges per node
  int* edge_counts;
  cudaMalloc(&edge_counts, n * sizeof(int));
  count_mutual_edges_kernel<<<grid_size, block_size>>>(neighbors, edge_counts, n, k);

  // exclusive prefix sum -> row_ptr (row_ptr[0] = 0, row_ptr[i] = sum(counts[0..i)))
  cudaMemset(g.row_ptr, 0, sizeof(int));
  thrust::inclusive_scan(thrust::device, edge_counts, edge_counts + n, g.row_ptr + 1);

  cudaMemcpy(&g.nnz, g.row_ptr + n, sizeof(int), cudaMemcpyDeviceToHost);

  if (g.nnz > 0) {
    cudaMalloc(&g.col_idx, g.nnz * sizeof(int));
    fill_mutual_edges_kernel<<<grid_size, block_size>>>(neighbors, g.row_ptr, g.col_idx, n, k);
  }

  cudaFree(edge_counts);
  return g;
}

void free_csr_graph(CSRGraph &g) {
  if (g.row_ptr) cudaFree(g.row_ptr);
  if (g.col_idx) cudaFree(g.col_idx);
  g.row_ptr = nullptr;
  g.col_idx = nullptr;
}
