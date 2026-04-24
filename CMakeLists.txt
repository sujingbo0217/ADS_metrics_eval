cmake_minimum_required(VERSION 3.18)
project(knn_cuda LANGUAGES CXX CUDA)

# ─── CUDA arch ───
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
  message(WARNING
    "CUDA ${CMAKE_CUDA_COMPILER_VERSION} < 11.8: "
    "compiling PTX for compute_86 (JIT to sm_89 at runtime). "
    "Upgrade to CUDA 12.x for native sm_89 support.")
  set(CMAKE_CUDA_ARCHITECTURES 86)
else()
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ─── Dependencies ───
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED
  HINTS "${Python3_SITELIB}" "$ENV{HOME}/.local"
)

# ─── Static library: core CUDA kernels ───
add_library(knn_core STATIC
  src/pairwise_dist.cu
  src/knn.cu
  src/graph.cu
  src/mutual_graph.cu
  src/search.cu
)
target_include_directories(knn_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(knn_core PUBLIC
  CUDA::cublas
  CUDA::cudart
)
# Suppress noisy CUDA warnings (optional)
target_compile_options(knn_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# ─── Python module: knn_ext ───
pybind11_add_module(knn_ext python/bindings.cpp)
target_link_libraries(knn_ext PRIVATE knn_core)

# After build, knn_ext.cpython-*.so lands in ${CMAKE_BINARY_DIR}.
# We create a symlink in python/ so "import knn_ext" works from there.
add_custom_command(TARGET knn_ext POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    $<TARGET_FILE:knn_ext>
    ${CMAKE_SOURCE_DIR}/python/$<TARGET_FILE_NAME:knn_ext>
  COMMENT "Symlinking knn_ext into python/"
)

# ─── C++ test binary ───
add_executable(test_knn tests/test_knn.cu)
target_link_libraries(test_knn PRIVATE knn_core)
