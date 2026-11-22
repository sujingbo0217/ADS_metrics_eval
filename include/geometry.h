#pragma once

#include "knn.h"

enum class Construction {
  Primary, 
  SharedNeighbors,
  LocalScaling,
  Geodesic
};

struct Neighborhood {
  DistanceType dist = DistanceType::L2;
  Construction kind = Construction::Primary;
  int param = 0;
};

void neighbor_lists(const float* X, int n, int dim, int k, const Neighborhood& nb, int* idx);

struct NeighborhoodQuality {
  double hub_skew;
  int components;
  double lid_cv;
  double stability;
};
NeighborhoodQuality neighborhood_quality(const float* X, int n, int dim, int k,
                                         const Neighborhood& nb, int n_draws,
                                         double drop, unsigned seed);

double effective_rank(const float* X, int n, int dim);

double lid_profile(const float* X, int n, int dim, int k, DistanceType dist_type,
                   float* lid_pt, float* rk_pt);

void compute_lid(const float* dist, int n_q, int k, int ld, float* output);

void k_occurrence(const float* X, int n, int dim, int k, const Neighborhood& nb,
                  int* occ);

void relative_contrast(const float* X, int n, int dim, DistanceType dist_type,
                       float* mean_pt, float* nn_pt);

void component_counts(const float* X, int n, int dim, const Neighborhood& nb,
                      const int* ks, int n_ks, int* beta0);

void cross_set_mixing(const float* R, const float* G, int n_r, int n_g, int dim,
                      int k, const Neighborhood& nb, bool exclude_counterpart,
                      float* pt_real, float* pt_gen);

void cross_lid(const float* R, const float* G, int n_r, int n_g, int dim, int k,
               DistanceType dist_type, bool exclude_counterpart, float* lid_g);

void neighborhood_preservation(const float* R, const float* G, int n, int dim,
                               int k, const Neighborhood& nb,
                               float* t_pt, float* c_pt);

void neighborhood_drift(const float* R, const float* G, int n, int dim, int k,
                        const Neighborhood& nb, float* b_pt);

void off_manifold(const float* R, const float* G, int n, int dim, int k_t, int d_t,
                  DistanceType dist_type, float* nu_pt);

void off_manifold_reference(const float* R, int n, int dim, int k, int k_t, int d_t,
                            DistanceType dist_type, float* nu0);
