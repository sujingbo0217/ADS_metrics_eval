#include "cuda_check.h"
#include "device_buffer.h"
#include "geometry.h"
#include "knn.h"
#include "geometry.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace py = pybind11;

// Every array crossing this boundary is declared c_style | forcecast, so numpy
// converts float64, Fortran-order and sliced views before we see a pointer.
// Without it a non-contiguous or float64 array was read as raw bytes and the
// caller got silent nonsense, which is why the Python side was littered with
// np.ascontiguousarray().
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

// Everything here goes through an (n_q, n_b) distance matrix, whose elements
// the kernels and CUB's segment count address with a signed int.
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

// ─── Local Intrinsic Dimensionality ───
// Levina-Bickel MLE from k-NN distances that are already TRUE distances.
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

// ─── L2 row normalization ───
static py::array_t<float> py_l2_normalize_rows(Farray X) {
  auto b = X.request();
  check_matrix(b, "X");
  const int n = (int)b.shape[0], dim = (int)b.shape[1];

  auto d_X = upload(X);
  l2_normalize_rows(d_X.get(), n, dim);
  return download(d_X, {n, dim});
}

// ─── Full pairwise distance matrix ───
// Diagnostic. The properties that need ranks derive them on the device; pulling
// an n x n matrix across the bus to sort it on the host is what this rewrite
// removed from P4.
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

// ─── P1 position, P2 shape ───
static py::tuple py_set_stats(Farray X) {
  auto b = X.request();
  check_matrix(b, "X");
  const int n = (int)b.shape[0], dim = (int)b.shape[1];

  auto mu = py::array_t<double>(dim);
  SetStats s = set_stats(static_cast<const float*>(b.ptr), n, dim, mu.mutable_data());
  return py::make_tuple(mu, s.sigma, s.eff_rank);
}

// ─── P3 local intrinsic dimension ───
static py::tuple py_lid_profile(Farray X, int k, DistanceType dist_type) {
  auto b = X.request();
  check_matrix(b, "X");
  const int n = (int)b.shape[0], dim = (int)b.shape[1];
  check_matrix_fits(n, n);
  check_k(k, n - 1);

  DeviceArray<float> lid(n);
  double twonn = lid_profile(static_cast<const float*>(b.ptr), n, dim, k, dist_type,
                             lid.get());
  return py::make_tuple(download(lid, {n}), twonn);
}

// ─── P4 neighborhood preservation ───
static py::tuple py_neighborhood_preservation(Farray R, Farray G, int k,
                                               DistanceType dist_type) {
  auto rb = R.request(), gb = G.request();
  check_matrix(rb, "R");
  check_matrix(gb, "G");
  check_paired(rb, gb);
  const int n = (int)rb.shape[0], dim = (int)rb.shape[1];
  check_matrix_fits(n, n);
  check_k(k, n - 1);

  DeviceArray<float> t(n), c(n);
  neighborhood_preservation(static_cast<const float*>(rb.ptr),
                             static_cast<const float*>(gb.ptr), n, dim, k, dist_type,
                             t.get(), c.get());
  return py::make_tuple(download(t, {n}), download(c, {n}));
}

// ─── P5 cross-set mixing ───
static py::tuple py_cross_set_mixing(Farray R, Farray G, int k, DistanceType dist_type) {
  auto rb = R.request(), gb = G.request();
  check_matrix(rb, "R");
  check_matrix(gb, "G");
  if (rb.shape[1] != gb.shape[1])
    throw py::value_error("R and G must have the same dimension");
  const int n_r = (int)rb.shape[0], n_g = (int)gb.shape[0], dim = (int)rb.shape[1];
  check_matrix_fits(n_r + n_g, n_r + n_g);
  check_k(k, n_r + n_g - 1);

  DeviceArray<float> pr(n_r), pg(n_g);
  cross_set_mixing(static_cast<const float*>(rb.ptr), static_cast<const float*>(gb.ptr),
                   n_r, n_g, dim, k, dist_type, pr.get(), pg.get());
  return py::make_tuple(download(pr, {n_r}), download(pg, {n_g}));
}

PYBIND11_MODULE(knn_ext, m) {
  m.doc() = "CUDA k-NN and geometric-property kernels";

  // barycenter_shift, neighbor_overlap and compute_indegree were bound here.
  // They implemented the previous five-property design and nothing in the
  // current pipeline calls them; leaving them exposed invited the reading that
  // they were part of it. The implementations are kept, commented, at the end
  // of src/graph.cu.

  py::enum_<DistanceType>(m, "DistanceType")
    .value("L2", DistanceType::L2)
    .value("InnerProduct", DistanceType::InnerProduct)
    .value("Mahalanobis", DistanceType::Mahalanobis);

  // ── k-NN search ──
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

  // ── The five geometric properties ──
  m.def("set_stats", &py_set_stats,
        "P1 and P2 in one pass over the set: (mu (dim,) float64, sigma, eff_rank). "
        "sigma is the per-axis rms dispersion; eff_rank is exp of the entropy of "
        "the normalised covariance spectrum.",
        py::arg("X"));

  m.def("lid_profile", &py_lid_profile,
        "P3: (per-point Levina-Bickel LID at width k, global TwoNN estimate). "
        "One mixed search serves both; per-point values are NaN for k < 3.",
        py::arg("X"), py::arg("k"), py::arg("dist_type") = DistanceType::L2);

  m.def("neighborhood_preservation", &py_neighborhood_preservation,
        "P4: per-point (trustworthiness, continuity), Venna & Kaski, baseline 1. "
        "R and G must be row-aligned.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2);

  m.def("cross_set_mixing", &py_cross_set_mixing,
        "P5: per-point cross-set neighbor share over its random-mixing "
        "expectation, for the real rows and the generated rows.",
        py::arg("R"), py::arg("G"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2);

  // ── Graph properties, used by the superseded python/pipeline.py ──
  m.def("compute_lid", &py_compute_lid,
        "Levina-Bickel LID from TRUE k-NN distances sorted ascending. Reads the "
        "first k of the array's columns.",
        py::arg("dist"), py::arg("k"));
}
