#include "knn.h"
#include "utils.h"

#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

// ─── Host ↔ device helpers ───

// Copy numpy float array to newly allocated device memory
static float* to_device_float(py::array_t<float> arr) {
  auto buf = arr.request();
  size_t bytes = (size_t)buf.size * sizeof(float);
  float* d_ptr;
  cudaMalloc(&d_ptr, bytes);
  cudaMemcpy(d_ptr, buf.ptr, bytes, cudaMemcpyHostToDevice);
  return d_ptr;
}

// Copy numpy int array to newly allocated device memory
static int* to_device_int(py::array_t<int> arr) {
  auto buf = arr.request();
  size_t bytes = (size_t)buf.size * sizeof(int);
  int* d_ptr;
  cudaMalloc(&d_ptr, bytes);
  cudaMemcpy(d_ptr, buf.ptr, bytes, cudaMemcpyHostToDevice);
  return d_ptr;
}

// Optional (dim, dim) inv_cov matrix for Mahalanobis; nullptr when absent
static float* optional_inv_cov_to_device(py::object obj) {
  if (obj.is_none()) return nullptr;
  return to_device_float(obj.cast<py::array_t<float>>());
}

// ─── Cross-set k-NN ───
// For each query in `query`, find k nearest neighbors in `base`.
// Returns (neighbors, distances) of shape (n_q, k).
py::tuple py_cross_set_knn(
  py::array_t<float> base,    // (n_b, dim)
  py::array_t<float> query,   // (n_q, dim)
  int k, DistanceType dist_type,
  py::object inv_cov          // (dim, dim) or None
) {
  auto b_buf = base.request();
  auto q_buf = query.request();
  int n_b = b_buf.shape[0], dim = b_buf.shape[1];
  int n_q = q_buf.shape[0];

  float* d_base = to_device_float(base);
  float* d_query = to_device_float(query);
  float* d_inv_cov = optional_inv_cov_to_device(inv_cov);

  knnResult res = cross_set_knn(d_base, d_query, n_b, n_q, dim, k, dist_type, d_inv_cov);

  auto neighbors = py::array_t<int>({n_q, k});
  auto distances = py::array_t<float>({n_q, k});
  cudaMemcpy(neighbors.mutable_data(), res.indices, (size_t)n_q * k * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(distances.mutable_data(), res.distances, (size_t)n_q * k * sizeof(float), cudaMemcpyDeviceToHost);

  free_knn_result(res);
  cudaFree(d_base);
  cudaFree(d_query);
  if (d_inv_cov) cudaFree(d_inv_cov);
  return py::make_tuple(neighbors, distances);
}

// ─── Pooled k-NN ───
// For each point in `mixed`, find k nearest neighbors in `mixed` (excluding self).
// Returns (neighbors, distances) of shape (n, k).
py::tuple py_pooled_knn(
  py::array_t<float> mixed,   // (n, dim)
  int k, DistanceType dist_type,
  py::object inv_cov          // (dim, dim) or None
) {
  auto buf = mixed.request();
  int n = buf.shape[0], dim = buf.shape[1];

  float* d_mixed = to_device_float(mixed);
  float* d_inv_cov = optional_inv_cov_to_device(inv_cov);

  knnResult res = pooled_knn(d_mixed, n, dim, k, dist_type, d_inv_cov);

  auto neighbors = py::array_t<int>({n, k});
  auto distances = py::array_t<float>({n, k});
  cudaMemcpy(neighbors.mutable_data(), res.indices, (size_t)n * k * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(distances.mutable_data(), res.distances, (size_t)n * k * sizeof(float), cudaMemcpyDeviceToHost);

  free_knn_result(res);
  cudaFree(d_mixed);
  if (d_inv_cov) cudaFree(d_inv_cov);
  return py::make_tuple(neighbors, distances);
}

// ─── Barycenter shift ───
// Distance from each query to the centroid of its k-NN in base.
py::array_t<float> py_barycenter_shift(
  py::array_t<float> base,       // (n_b, dim)
  py::array_t<float> query,      // (n_q, dim)
  py::array_t<int> neighbors,    // (n_q, k) from cross_set_knn
  int k
) {
  auto b_buf = base.request();
  auto q_buf = query.request();
  int dim = b_buf.shape[1];
  int n_q = q_buf.shape[0];

  float* d_base = to_device_float(base);
  float* d_query = to_device_float(query);
  int* d_neighbors = to_device_int(neighbors);

  float* d_output;
  cudaMalloc(&d_output, n_q * sizeof(float));
  barycenter_shift(d_base, d_query, d_neighbors, n_q, k, dim, d_output);

  auto output = py::array_t<float>(n_q);
  cudaMemcpy(output.mutable_data(), d_output, n_q * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_base);
  cudaFree(d_query);
  cudaFree(d_neighbors);
  cudaFree(d_output);
  return output;
}

// ─── Neighborhood overlap ───
// Fraction of real-only k-NN that also appear in the pooled k-NN, per real point.
py::array_t<float> py_neighbor_overlap(
  py::array_t<int> knn_real,     // (n_b, k)
  py::array_t<int> knn_pooled,   // (n, k) - first n_b rows used
  int k
) {
  int n_b = knn_real.request().shape[0];

  int* d_real = to_device_int(knn_real);
  int* d_pooled = to_device_int(knn_pooled);

  float* d_output;
  cudaMalloc(&d_output, n_b * sizeof(float));
  neighbor_overlap(d_real, d_pooled, n_b, k, d_output);

  auto output = py::array_t<float>(n_b);
  cudaMemcpy(output.mutable_data(), d_output, n_b * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_real);
  cudaFree(d_pooled);
  cudaFree(d_output);
  return output;
}

// ─── Local Intrinsic Dimensionality ───
// LID from cross-set distances sorted ascending along axis 1.
py::array_t<float> py_compute_lid(
  py::array_t<float> dist,       // (n_q, k)
  int k
) {
  int n_q = dist.request().shape[0];

  float* d_dist = to_device_float(dist);
  float* d_output;
  cudaMalloc(&d_output, n_q * sizeof(float));
  compute_lid(d_dist, n_q, k, d_output);

  auto output = py::array_t<float>(n_q);
  cudaMemcpy(output.mutable_data(), d_output, n_q * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_dist);
  cudaFree(d_output);
  return output;
}

// ─── In-degree distribution ───
// Per-node in-degree in a directed k-NN graph.
py::array_t<int> py_compute_indegree(
  py::array_t<int> neighbors,    // (n, k)
  int k
) {
  int n = neighbors.request().shape[0];

  int* d_neighbors = to_device_int(neighbors);
  int* d_output;
  cudaMalloc(&d_output, n * sizeof(int));
  compute_indegree(d_neighbors, n, k, d_output);

  auto output = py::array_t<int>(n);
  cudaMemcpy(output.mutable_data(), d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_neighbors);
  cudaFree(d_output);
  return output;
}

// ─── L2 row normalization ───
// Returns a copy of X with each row L2-normalized.
py::array_t<float> py_l2_normalize_rows(py::array_t<float> X) {
  auto buf = X.request();
  int n = buf.shape[0], dim = buf.shape[1];

  float* d_X = to_device_float(X);
  l2_normalize_rows(d_X, n, dim);

  auto output = py::array_t<float>({n, dim});
  cudaMemcpy(output.mutable_data(), d_X, (size_t)n * dim * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(d_X);
  return output;
}

PYBIND11_MODULE(knn_ext, m) {
  m.doc() = "CUDA k-NN library for V&V geometric validation";

  py::enum_<DistanceType>(m, "DistanceType")
    .value("L2", DistanceType::L2)
    .value("InnerProduct", DistanceType::InnerProduct)
    .value("Mahalanobis", DistanceType::Mahalanobis);

  m.def("cross_set_knn", &py_cross_set_knn,
        "Cross-set exact k-NN: for each query, k nearest neighbors in base",
        py::arg("base"), py::arg("query"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("inv_cov") = py::none());

  m.def("pooled_knn", &py_pooled_knn,
        "Pooled exact k-NN (self-query, excludes self)",
        py::arg("X"), py::arg("k"),
        py::arg("dist_type") = DistanceType::L2,
        py::arg("inv_cov") = py::none());

  m.def("barycenter_shift", &py_barycenter_shift,
        "Distance from each query to centroid of its k-NN in base",
        py::arg("base"), py::arg("query"),
        py::arg("neighbors"), py::arg("k"));

  m.def("neighbor_overlap", &py_neighbor_overlap,
        "Per-real-point overlap between real-only and pooled k-NN rows",
        py::arg("knn_real"), py::arg("knn_pooled"), py::arg("k"));

  m.def("compute_lid", &py_compute_lid,
        "Local Intrinsic Dimensionality from sorted cross-set distances",
        py::arg("dist"), py::arg("k"));

  m.def("compute_indegree", &py_compute_indegree,
        "In-degree of each node in a directed k-NN graph",
        py::arg("neighbors"), py::arg("k"));

  m.def("l2_normalize_rows", &py_l2_normalize_rows,
        "Row-wise L2 normalization (returns a copy)",
        py::arg("X"));
}
