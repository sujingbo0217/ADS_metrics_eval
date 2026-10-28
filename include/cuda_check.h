#pragma once
// Fail loudly. Nothing in this library checked a CUDA return code, and the
// consequence was not a crash but silent wrong answers: a module built for
// sm_89 and loaded on an sm_80 A100 fails EVERY hand-written kernel launch with
// cudaErrorNoKernelImageForDevice, leaves the output buffer as cudaMalloc
// returned it (zeros), and reports nothing. cuBLAS keeps working, because it
// ships a fat binary for every architecture -- so the GEMM-based results looked
// plausible while every kernel-based result was zero.
//
// CUDA_CHECK wraps an API call; CUDA_CHECK_KERNEL follows a <<<>>> launch and
// catches both the launch error and, after a synchronise, the execution error.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _e = (expr);                                                  \
    if (_e != cudaSuccess) {                                                  \
      std::fprintf(stderr, "[knn_ext] CUDA error at %s:%d in `%s`: %s (%d)\n",\
                   __FILE__, __LINE__, #expr, cudaGetErrorString(_e), (int)_e);\
      if (_e == cudaErrorNoKernelImageForDevice) {                            \
        int _dev = 0, _maj = 0, _min = 0;                                     \
        cudaGetDevice(&_dev);                                                 \
        cudaDeviceGetAttribute(&_maj, cudaDevAttrComputeCapabilityMajor, _dev);\
        cudaDeviceGetAttribute(&_min, cudaDevAttrComputeCapabilityMinor, _dev);\
        std::fprintf(stderr,                                                  \
          "[knn_ext] this GPU is sm_%d%d but the module was not compiled for " \
          "it.\n[knn_ext] rebuild with -DKNN_CUDA_ARCH=%d%d (see README 1b)\n",\
          _maj, _min, _maj, _min);                                            \
      }                                                                       \
      std::abort();                                                           \
    }                                                                         \
  } while (0)

// Use immediately after a kernel launch.
#define CUDA_CHECK_KERNEL()                                                   \
  do {                                                                        \
    CUDA_CHECK(cudaGetLastError());                                           \
    CUDA_CHECK(cudaDeviceSynchronize());                                      \
  } while (0)
