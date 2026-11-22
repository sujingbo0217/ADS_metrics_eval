#pragma once

#include "cuda_check.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <utility>

template <typename T>
class DeviceArray {
 public:
  DeviceArray() = default;
  explicit DeviceArray(size_t n) { alloc(n); }
  ~DeviceArray() { free(); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
  DeviceArray& operator=(DeviceArray&& o) noexcept {
    if (this != &o) { free(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; }
    return *this;
  }

  void alloc(size_t n) {
    free();
    if (n) CUDA_CHECK(cudaMalloc(&p_, n * sizeof(T)));
    n_ = n;
  }
  void free() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    n_ = 0;
  }

  void from_host(const T* src, size_t n) {
    alloc(n);
    CUDA_CHECK(cudaMemcpy(p_, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void to_host(T* dst) const {
    CUDA_CHECK(cudaMemcpy(dst, p_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }
  void zero() { CUDA_CHECK(cudaMemset(p_, 0, n_ * sizeof(T))); }

  T* get() { return p_; }
  const T* get() const { return p_; }
  size_t size() const { return n_; }

 private:
  T* p_ = nullptr;
  size_t n_ = 0;
};

cublasHandle_t knn_cublas_handle();

#define CUBLAS_CHECK(expr)                                                    \
  do {                                                                        \
    cublasStatus_t _s = (expr);                                               \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "[knn_ext] cuBLAS error at %s:%d in `%s`: %d\n",   \
                   __FILE__, __LINE__, #expr, (int)_s);                       \
      std::abort();                                                           \
    }                                                                         \
  } while (0)
