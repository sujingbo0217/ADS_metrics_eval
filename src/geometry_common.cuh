#pragma once

#include "geometry.h"
#include "device_buffer.h"

#include <cstddef>

static constexpr int kBlock = 256;
static inline int grid_for(size_t n, int block = kBlock) {
  size_t g = (n + block - 1) / block;
  return (int)(g > 65535 ? 65535 : (g ? g : 1));
}

static constexpr float kUnreachable = 1e29f;

void to_true_distance(float* d, size_t total, DistanceType dist_type);

void construction_matrix(const float* d_X, int N, int dim, const Neighborhood& nb,
                         int pair, DeviceArray<float>& D);

void neighbor_lists_host(const float* d_X, int N, int dim, int k,
                         const Neighborhood& nb, int pair, int* idx_host);
