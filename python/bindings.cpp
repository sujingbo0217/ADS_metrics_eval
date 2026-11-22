#include "cuda_check.h"
#include "device_buffer.h"
#include "geometry.h"
#include "knn.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

using Farray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using Iarray = py::array_t<int, py::array::c_style | py::array::forcecast>;

// ─── Argument checking ───

static void check_matrix(const py::buffer_info& b, const char* name) {
  if (b.ndim != 2)
    throw py::value_error(std::string(name) + " must be 2-D, got " +
                          std::to_string(b.ndim) + "-D");
  if (b.shape[0] < 1 || b.shape[1] < 1)
    throw py::value_error(std::string(name) + " is empty");
}

static void check_matrix_fits(long long n_q, long long n_b) {
  if (n_q * n_b > 2147483647LL)
    throw py::value_error("the " + std::to_string(n_q) + " x " + std::to_string(n_b) +
                          " distance matrix this needs does not fit in a signed int; "
                          "split the set");
}

static void check_paired(const py::buffer_info& a, const py::buffer_info& b) {
  if (a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1])
    throw py::value_error("R and G must have the same shape: row i of both is the "
                          "same image. Got (" + std::to_string(a.shape[0]) + ", " +
                          std::to_string(a.shape[1]) + ") and (" +
                          std::to_string(b.shape[0]) + ", " +
                          std::to_string(b.shape[1]) + ")");
}

static void check_k(int k, int avail) {
  if (k < 1 || k > avail)
    throw py::value_error("k = " + std::to_string(k) + " out of range: " +
                          std::to_string(avail) + " candidate neighbor(s) per row");
}

// ─── Host <-> device helpers ───

static DeviceArray<float> upload(const Farray& a) {
  auto b = a.request();
  DeviceArray<float> d;
  d.from_host(static_cast<const float*>(b.ptr), (size_t)b.size);
  return d;
}

static DeviceArray<int> upload(const Iarray& a) {
  auto b = a.request();
  DeviceArray<int> d;
  d.from_host(static_cast<const int*>(b.ptr), (size_t)b.size);
  return d;
}

// Optional (dim, dim) inv_cov for Mahalanobis; empty when absent.
static DeviceArray<float> upload_optional(const py::object& obj) {
  if (obj.is_none()) return DeviceArray<float>();
  return upload(obj.cast<Farray>());
}

template <typename T>
static py::array_t<T> download(const DeviceArray<T>& d, std::vector<py::ssize_t> shape) {
  py::array_t<T> out(shape);
  d.to_host(out.mutable_data());
  return out;
}

// ─── Cross-set k-NN ───
// For each query in `query`, find k nearest neighbors in `base`.
// Returns (neighbors, distances) of shape (n_q, k).
static py::tuple py_cross_set_knn(Farray base, Farray query, int k,
                                  DistanceType dist_type, py::object inv_cov) {
  auto b = base.request(), q = query.request();
  check_matrix(b, "base");
  check_matrix(q, "query");
  if (b.shape[1] != q.shape[1])
    throw py::value_error("base and query must have the same dimension");
  const int n_b = (int)b.shape[0], n_q = (int)q.shape[0], dim = (int)b.shape[1];
  check_matrix_fits(n_q, n_b);
  check_k(k, n_b);

  auto d_base = upload(base);
  auto d_query = upload(query);
  auto d_cov = upload_optional(inv_cov);

  knnResult res = cross_set_knn(d_base.get(), d_query.get(), n_b, n_q, dim, k,
                                dist_type, d_cov.get());
  auto neighbors = py::array_t<int>({n_q, k});
  auto distances = py::array_t<float>({n_q, k});
  CUDA_CHECK(cudaMemcpy(neighbors.mutable_data(), res.indices,
                        (size_t)n_q * k * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(distances.mutable_data(), res.distances,
                        (size_t)n_q * k * sizeof(float), cudaMemcpyDeviceToHost));
  free_knn_result(res);
  return py::make_tuple(neighbors, distances);
}

// ─── Mixed k-NN ───
// For each point in `X`, find k nearest neighbors in `X` (excluding self).
static py::tuple py_mixed_knn(Farray X, int k, DistanceType dist_type,
                               py::object inv_cov) {
  auto b = X.request();
  check_matrix(b, "X");
  const int n = (int)b.shape[0], dim = (int)b.shape[1];
  check_matrix_fits(n, n);
  check_k(k, n - 1);

  auto d_X = upload(X);
  auto d_cov = upload_optional(inv_cov);

  knnResult res = mixed_knn(d_X.get(), n, dim, k, dist_type, d_cov.get());
  auto neighbors = py::array_t<int>({n, k});
  auto distances = py::array_t<float>({n, k});
  CUDA_CHECK(cudaMemcpy(neighbors.mutable_data(), res.indices,
                        (size_t)n * k * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(distances.mutable_data(), res.distances,
                        (size_t)n * k * sizeof(float), cudaMemcpyDeviceToHost));
  free_knn_result(res);
  return py::make_tuple(neighbors, distances);
}

// LID
static py::array_t<float> py_compute_lid(Farray dist, int k) {
  auto b = dist.request();
  check_matrix(b, "dist");
  const int n_q = (int)b.shape[0], ld = (int)b.shape[1];
  if (k < 1 || k > ld)
    throw py::value_error("k = " + std::to_string(k) + " exceeds the " +
                          std::to_string(ld) + " neighbor columns of dist");

  auto d_dist = upload(dist);
  DeviceArray<float> out(n_q);
  compute_lid(d_dist.get(), n_q, k, ld, out.get());
  return download(out, {n_q});
}

static py::array_t<float> py_l2_normalize_rows(Farray X) {
  auto b = X.request();
  check_matrix(b, "X");
  const int n = (int)b.shape[0], dim = (int)b.shape[1];

  auto d_X = upload(X);
  l2_normalize_rows(d_X.get(), n, dim);
  return download(d_X, {n, dim});
}

static py::array_t<float> py_pairwise_distances(Farray base, py::object query,
                                                DistanceType dist_type) {
  auto b = base.request();
  check_matrix(b, "base");
  const int n_b = (int)b.shape[0], dim = (int)b.shape[1];

  auto d_base = upload(base);
  int n_q = n_b;
  DeviceArray<float> d_query;
  const float* q_ptr = d_base.get();
  if (!query.is_none()) {
    Farray q = query.cast<Farray>();
    auto qb = q.request();
    check_matrix(qb, "query");
    if (qb.shape[1] != dim)
      throw py::value_error("base and query must have the same dimension");
    n_q = (int)qb.shape[0];
    d_query = upload(q);
    q_ptr = d_query.get();
  }
  check_matrix_fits(n_q, n_b);

  DeviceArray<float> out((size_t)n_q * n_b);
  pairwise_distances(d_base.get(), q_ptr, n_b, n_q, dim, dist_type, out.get());
  return download(out, {n_q, n_b});
}

// ─── Geometric properties ───

static std::pair<int, int> matrix_shape(const Farray& a, const char* name) {
  auto b = a.request();
  check_matrix(b, name);
  return {(int)b.shape[0], (int)b.shape[1]};
}

static const float* host_ptr(const Farray& a) { return static_cast<const float*>(a.data()); }

// A construction over N points. Each one materializes an (N, N) matrix, and
// the geodesic pads it to a multiple of 32.
static Neighborhood make_neighborhood(DistanceType dist_type, Construction kind, int param,
                                      int N) {
  if (dist_type == DistanceType::Mahalanobis)
    throw py::value_error("the geometric properties take L2 or InnerProduct");
  const long long P = ((long long)N + 31) / 32 * 32;
  check_matrix_fits(P, P);
  if (kind != Construction::Primary && (param < 1 || param > N - 2))
    throw py::value_error("construction parameter " + std::to_string(param) +
                          " out of range: need 1 <= param <= " + std::to_string(N - 2));
  Neighborhood nb;
  nb.dist = dist_type;
  nb.kind = kind;
  nb.param = param;
  return nb;
}

static void check_same_dim(int a, int b) {
  if (a != b) throw py::value_error("R and G must have the same dimension");
}

static void check_paired_shapes(std::pair<int, int> r, std::pair<int, int> g) {
  if (r != g)
    throw py::value_error("R and G must have the same shape: row i of both is the "
                          "same image. Got (" + std::to_string(r.first) + ", " +
                          std::to_string(r.second) + ") and (" +
                          std::to_string(g.first) + ", " + std::to_string(g.second) + ")");
}

static py::array_t<int> py_neighbor_lists(Farray X, int k, DistanceType dist_type,
                                          Construction kind, int param) {
  auto [n, dim] = matrix_shape(X, "X");
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  check_k(k, n - 1);
  DeviceArray<int> idx((size_t)n * k);
  neighbor_lists(host_ptr(X), n, dim, k, nb, idx.get());
  return download(idx, {n, k});
}

static py::dict py_neighborhood_quality(Farray X, int k, DistanceType dist_type,
                                        Construction kind, int param, int n_draws,
                                        double drop, unsigned seed) {
  auto [n, dim] = matrix_shape(X, "X");
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  check_k(k, n - 1);
  if (k < 3) throw py::value_error("Q3 needs the pointwise LID, defined for k >= 3");
  if (!(drop > 0.0 && drop < 1.0)) throw py::value_error("drop must lie in (0, 1)");
  NeighborhoodQuality q =
      neighborhood_quality(host_ptr(X), n, dim, k, nb, n_draws, drop, seed);
  py::dict d;
  d["hub_skew"] = q.hub_skew;
  d["components"] = q.components;
  d["lid_cv"] = q.lid_cv;
  d["stability"] = q.stability;
  return d;
}

// (*) GP1 spectral shape
static double py_effective_rank(Farray X) {
  auto [n, dim] = matrix_shape(X, "X");
  return effective_rank(host_ptr(X), n, dim);
}

// (*) GP2 local scale and dimension
static py::tuple py_lid_profile(Farray X, int k, DistanceType dist_type) {
  auto [n, dim] = matrix_shape(X, "X");
  check_matrix_fits(n, n);
  check_k(k, n - 1);
  DeviceArray<float> lid(n), rk(n);
  double twonn = lid_profile(host_ptr(X), n, dim, k, dist_type, lid.get(), rk.get());
  return py::make_tuple(download(lid, {n}), download(rk, {n}), twonn);
}

// GP3 hubness and concentration
static py::array_t<int> py_k_occurrence(Farray X, int k, DistanceType dist_type,
                                        Construction kind, int param) {
  auto [n, dim] = matrix_shape(X, "X");
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  check_k(k, n - 1);
  DeviceArray<int> occ(n);
  k_occurrence(host_ptr(X), n, dim, k, nb, occ.get());
  return download(occ, {n});
}

static py::tuple py_relative_contrast(Farray X, DistanceType dist_type) {
  auto [n, dim] = matrix_shape(X, "X");
  make_neighborhood(dist_type, Construction::Primary, 0, n);
  if (n < 2) throw py::value_error("relative contrast needs at least two points");
  DeviceArray<float> mean(n), nn(n);
  relative_contrast(host_ptr(X), n, dim, dist_type, mean.get(), nn.get());
  return py::make_tuple(download(mean, {n}), download(nn, {n}));
}

// GP4 connectivity
static py::array_t<int> py_component_counts(Farray X, std::vector<int> ks,
                                            DistanceType dist_type, Construction kind,
                                            int param) {
  auto [n, dim] = matrix_shape(X, "X");
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  if (ks.empty()) throw py::value_error("ks is empty");
  for (int k : ks) check_k(k, n - 1);
  py::array_t<int> out((py::ssize_t)ks.size());
  component_counts(host_ptr(X), n, dim, nb, ks.data(), (int)ks.size(), out.mutable_data());
  return out;
}

// (*) GP5 cross-set mixing
static py::tuple py_cross_set_mixing(Farray R, Farray G, int k, DistanceType dist_type,
                                     Construction kind, int param,
                                     bool exclude_counterpart) {
  auto r = matrix_shape(R, "R");
  auto g = matrix_shape(G, "G");
  check_same_dim(r.second, g.second);
  if (exclude_counterpart) check_paired_shapes(r, g);
  const int n_r = r.first, n_g = g.first, dim = r.second;
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n_r + n_g);
  check_k(k, n_r + n_g - 1 - (exclude_counterpart ? 1 : 0));
  DeviceArray<float> pr(n_r), pg(n_g);
  cross_set_mixing(host_ptr(R), host_ptr(G), n_r, n_g, dim, k, nb, exclude_counterpart, pr.get(), pg.get());
  return py::make_tuple(download(pr, {n_r}), download(pg, {n_g}));
}

static py::array_t<float> py_cross_lid(Farray R, Farray G, int k, DistanceType dist_type,
                                       bool exclude_counterpart) {
  auto r = matrix_shape(R, "R");
  auto g = matrix_shape(G, "G");
  check_same_dim(r.second, g.second);
  if (exclude_counterpart) check_paired_shapes(r, g);
  const int n_r = r.first, n_g = g.first, dim = r.second;
  make_neighborhood(dist_type, Construction::Primary, 0, 1);
  check_matrix_fits(n_g, n_r);
  check_k(k, n_r - (exclude_counterpart ? 1 : 0));
  DeviceArray<float> lid(n_g);
  cross_lid(host_ptr(R), host_ptr(G), n_r, n_g, dim, k, dist_type, exclude_counterpart, lid.get());
  return download(lid, {n_g});
}

// GP6 neighborhood preservation and drift
static py::tuple py_neighborhood_preservation(Farray R, Farray G, int k,
                                              DistanceType dist_type, Construction kind,
                                              int param) {
  auto r = matrix_shape(R, "R");
  check_paired_shapes(r, matrix_shape(G, "G"));
  const int n = r.first, dim = r.second;
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  check_k(k, n - 1);
  DeviceArray<float> t(n), c(n);
  neighborhood_preservation(host_ptr(R), host_ptr(G), n, dim, k, nb, t.get(), c.get());
  return py::make_tuple(download(t, {n}), download(c, {n}));
}

static py::array_t<float> py_neighborhood_drift(Farray R, Farray G, int k,
                                                DistanceType dist_type, Construction kind,
                                                int param) {
  auto r = matrix_shape(R, "R");
  check_paired_shapes(r, matrix_shape(G, "G"));
  const int n = r.first, dim = r.second;
  Neighborhood nb = make_neighborhood(dist_type, kind, param, n);
  check_k(k, n - 1);
  DeviceArray<float> b(n);
  neighborhood_drift(host_ptr(R), host_ptr(G), n, dim, k, nb, b.get());
  return download(b, {n});
}

// GP7 off-manifold displacement
static void check_tangent(int n, int k_t, int d_t) {
  if (k_t < 1 || k_t > 63 || k_t > n - 1)
    throw py::value_error("k_t = " + std::to_string(k_t) +
                          " out of range: need 1 <= k_t <= min(63, n - 1)");
  if (d_t < 1) throw py::value_error("d_t must be at least 1");
}

static py::array_t<float> py_off_manifold(Farray R, Farray G, int d_t, int k_t,
                                          DistanceType dist_type) {
  auto r = matrix_shape(R, "R");
  check_paired_shapes(r, matrix_shape(G, "G"));
  const int n = r.first, dim = r.second;
  make_neighborhood(dist_type, Construction::Primary, 0, n);
  check_tangent(n, k_t, d_t);
  DeviceArray<float> nu(n);
  off_manifold(host_ptr(R), host_ptr(G), n, dim, k_t, d_t, dist_type, nu.get());
  return download(nu, {n});
}

static py::array_t<float> py_off_manifold_reference(Farray R, int k, int d_t, int k_t,
                                                    DistanceType dist_type) {
  auto [n, dim] = matrix_shape(R, "R");
  make_neighborhood(dist_type, Construction::Primary, 0, n);
  check_tangent(n, k_t, d_t);
  if (k < 1 || k > k_t)
    throw py::value_error("k = " + std::to_string(k) + " must lie in [1, k_t]");
  DeviceArray<float> nu0((size_t)n * k);
  off_manifold_reference(host_ptr(R), n, dim, k, k_t, d_t, dist_type, nu0.get());
  return download(nu0, {n, k});
}

PYBIND11_MODULE(knn_ext, m) {
  m.doc() = "CUDA k-NN, neighborhood constructions, and the geometric properties GP1--GP7";

  py::enum_<DistanceType>(m, "DistanceType")
    .value("L2", DistanceType::L2)
    .value("InnerProduct", DistanceType::InnerProduct)
    .value("Mahalanobis", DistanceType::Mahalanobis);

  py::enum_<Construction>(m, "Construction")
    .value("Primary", Construction::Primary)
    .value("SharedNeighbors", Construction::SharedNeighbors)
    .value("LocalScaling", Construction::LocalScaling)
    .value("Geodesic", Construction::Geodesic);

  m.def("cross_set_knn", &py_cross_set_knn,
        "Cross-set exact k-NN: for each query, k nearest neighbors in base",
        py::arg("base"), py::arg("query"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("inv_cov") = py::none());

  m.def("mixed_knn", &py_mixed_knn,
        "Mixed k-NN (self-query, excludes self)",
        py::arg("X"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("inv_cov") = py::none());

  m.def("pairwise_distances", &py_pairwise_distances,
        "Full pairwise distance matrix (n_q, n_b); query=None gives self-distances. "
        "L2 returns SQUARED Euclidean, InnerProduct returns NEGATIVE dot product. "
        "Diagnostic: the properties derive their ranks on the device.",
        py::arg("base"), py::arg("query") = py::none(),
        py::arg("dist_type") = DistanceType::L2);

  m.def("l2_normalize_rows", &py_l2_normalize_rows,
        "Row-wise L2 normalization (returns a copy)",
        py::arg("X"));

  m.def("neighbor_lists", &py_neighbor_lists,
        "(n, k) neighbors of every row under a construction, self excluded, "
        "nearest first.",
        py::arg("X"), py::arg("k"), py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0);

  m.def("neighborhood_quality", &py_neighborhood_quality,
        "Task-free quality criteria of a construction on one cloud: dict with "
        "hub_skew (Q1), components (Q2), lid_cv (Q3), stability (Q4).",
        py::arg("X"), py::arg("k"), py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0,
        py::arg("n_draws") = 10, py::arg("drop") = 0.2, py::arg("seed") = 0u);

  m.def("effective_rank", &py_effective_rank,
        "GP1: exp of the entropy of the normalized covariance spectrum.",
        py::arg("X"));

  m.def("lid_profile", &py_lid_profile,
        "GP2: (per-point Levina-Bickel LID at width k, per-point true distance to "
        "the k-th neighbor, global TwoNN estimate). Primary distance; per-point LID "
        "is NaN for k < 3.",
        py::arg("X"), py::arg("k"), py::arg("dist_type") = DistanceType::L2);

  m.def("k_occurrence", &py_k_occurrence,
        "GP3 hubness: (n,) k-occurrence of every row under a construction.",
        py::arg("X"), py::arg("k"), py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0);

  m.def("relative_contrast", &py_relative_contrast,
        "GP3 concentration: per-point (mean distance to the other points, nearest "
        "distance), primary true distance. RC = sum(mean) / sum(nearest).",
        py::arg("X"), py::arg("dist_type") = DistanceType::L2);

  m.def("component_counts", &py_component_counts,
        "GP4: connected components of the symmetrized k-NN graph for each k in ks.",
        py::arg("X"), py::arg("ks"), py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0);

  m.def("cross_set_mixing", &py_cross_set_mixing,
        "GP5: per-point cross-set neighbor share over its random-mixing "
        "expectation, for the real rows and the generated rows. With "
        "exclude_counterpart, row i of R and row i of G never see each other.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0,
        py::arg("exclude_counterpart") = true);

  m.def("cross_lid", &py_cross_lid,
        "GP5: per-row LID of G from its k nearest rows of R (Barua et al. 2019), "
        "primary distance, own clean row excluded with exclude_counterpart.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("exclude_counterpart") = true);

  m.def("neighborhood_preservation", &py_neighborhood_preservation,
        "GP6: per-point (trustworthiness, continuity), Venna & Kaski, baseline 1, "
        "under a construction. R and G must be row-aligned.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0);

  m.def("neighborhood_drift", &py_neighborhood_drift,
        "GP6: per-point distance between the clean barycenters of the clean and "
        "the corrupted neighborhoods, in units of the clean k-NN radius.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("construction") = Construction::Primary, py::arg("param") = 0);

  m.def("off_manifold", &py_off_manifold,
        "GP7: per-point off-tangent fraction of g_i - r_i, tangent = top-d_t local "
        "PCA of r_i and its k_t primary neighbors.",
        py::arg("R"), py::arg("G"), py::arg("d_t"), py::arg("k_t") = 50,
        py::arg("dist_type") = DistanceType::L2);

  m.def("off_manifold_reference", &py_off_manifold_reference,
        "GP7 reference: (n, k) off-tangent fractions of r_j - r_i for the first k "
        "neighbors j, each left out of the tangent estimate. nu0 = row median.",
        py::arg("R"), py::arg("k"), py::arg("d_t"), py::arg("k_t") = 50,
        py::arg("dist_type") = DistanceType::L2);

  m.def("compute_lid", &py_compute_lid,
        "Levina-Bickel LID from TRUE k-NN distances sorted ascending. Reads the "
        "first k of the array's columns.",
        py::arg("dist"), py::arg("k"));
}
