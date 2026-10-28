#pragma once
#include "knn.h"

// Device-side forms of the five geometric properties of scripts/geometry_lib.py.
//
// Each entry point takes host pointers to row-major float32 embeddings, does
// everything on the GPU, and returns only the small result -- a scalar or an
// (n,) vector. That is the whole point of them: the quantities below are all
// defined over an n x n relation, and the previous Python implementation built
// those matrices on the host. P4 copied two n x n distance matrices back and
// argsorted them in numpy; P2 ran a float64 SVD of the full (n, dim) block.
// At the largest pool (n = 3769, dim = 2048) those two accounted for 5.4 of the
// 6.1 seconds of a single (condition, space, k) evaluation, of which the grid
// runs 1350 per pool.
//
// DISTANCE CONVENTION. DistanceType::L2 is SQUARED Euclidean and
// DistanceType::InnerProduct is NEGATIVE dot product, as everywhere else in
// this library. Both are monotone in true distance, so P4 and P5 -- which use
// ordering only -- consume them raw. P3 needs magnitudes and converts on the
// device (true_distance_inplace), which is the one place the old Python
// _true_dist() did the same job in float64 on the host.

// ─── P1 position, P2 shape ───
// Both are per-SET quantities, and both need the centred second moment, so one
// pass serves both.
//   mu       (dim,) column mean, accumulated in double
//   sigma    per-axis rms dispersion, sqrt(mean_i ||x_i - mu||^2 / dim)
//   eff_rank exp of the Shannon entropy of the normalised covariance spectrum
//            (Roy & Vetterli 2007)
struct SetStats {
  double sigma;
  double eff_rank;
};
SetStats set_stats(const float* X, int n, int dim, double* mu /* (dim,) host */);

// ─── P3 local intrinsic dimension ───
// One pooled k-NN serves both estimators.
//   lid_pt (n,) device, per-point Levina-Bickel MLE at width k; filled with NaN
//          when k < 3, where the bias-corrected estimator is undefined. May be
//          null if only the global figure is wanted.
//   return global TwoNN estimate (Facco et al. 2017) from the first two
//          neighbors, NaN when fewer than 10 usable ratios survive.
double lid_profile(const float* X, int n, int dim, int k, DistanceType dist_type,
                   float* lid_pt);

// ─── P4 neighborhood preservation ───
// Venna & Kaski trustworthiness and continuity, per point, baseline 1. R and G
// must be row-aligned: row i of both is the same image.
// Defined over the FULL rank of every point from every other, not over a
// k-truncated list, so this sorts complete rows rather than going through
// topk_search_kernel.
void neighborhood_preservation(
  const float* R, const float* G, int n, int dim, int k,
  DistanceType dist_type,
  float* t_pt, float* c_pt   // (n,) device
);

// ─── P3 helper, exported ───
// Levina-Bickel MLE with the MacKay-Ghahramani k-2 bias correction, over the
// first k columns of an (n_q, ld) row-major distance matrix sorted ascending.
// Called by lid_profile; separate because the Hill sum is also useful on a
// distance matrix produced elsewhere. `dist` must hold TRUE distances, not the
// squared ones DistanceType::L2 produces.
void compute_lid(const float* dist, int n_q, int k, int ld, float* output);

// ─── P5 cross-set mixing ───
// Per-point share of the other set among the k nearest neighbors in R u G,
// divided by the share random mixing would give. Baseline 1, 0 = separable.
void cross_set_mixing(
  const float* R, const float* G, int n_r, int n_g, int dim, int k,
  DistanceType dist_type,
  float* pt_real, float* pt_gen   // (n_r,) and (n_g,) device
);
