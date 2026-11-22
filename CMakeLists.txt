cmake_minimum_required(VERSION 3.18)
project(knn_cuda LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CUDA arch
set(KNN_CUDA_ARCH "" CACHE STRING "CUDA architecture to compile for, e.g. 75")
if(NOT KNN_CUDA_ARCH)
  execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                  OUTPUT_VARIABLE _knn_cap RESULT_VARIABLE _knn_rc ERROR_QUIET)
  if(_knn_rc EQUAL 0 AND _knn_cap MATCHES "^([0-9]+)\\.([0-9]+)")
    set(KNN_CUDA_ARCH "${CMAKE_MATCH_1}${CMAKE_MATCH_2}" CACHE STRING
        "CUDA architecture to compile for, e.g. 75" FORCE)
    message(STATUS "knn_ext: compiling for the GPU of this machine, sm_${KNN_CUDA_ARCH}")
  endif()
endif()
if(KNN_CUDA_ARCH)
  set(CMAKE_CUDA_ARCHITECTURES ${KNN_CUDA_ARCH})
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.8)
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

# Dependencies
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED
  HINTS "${Python3_SITELIB}" "$ENV{HOME}/.local"
)

add_library(knn_core STATIC
  src/pairwise_dist.cu
  src/knn.cu
  src/graph.cu
  src/neighborhood.cu
  src/tangent.cu
  src/search.cu
)
target_include_directories(knn_core PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(knn_core PUBLIC
  CUDA::cublas
  CUDA::cusolver
  CUDA::cudart
)

target_compile_options(knn_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# knn_ext module
pybind11_add_module(knn_ext python/bindings.cpp)
target_link_libraries(knn_ext PRIVATE knn_core)

add_custom_command(TARGET knn_ext POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    $<TARGET_FILE:knn_ext>
    ${CMAKE_SOURCE_DIR}/python/$<TARGET_FILE_NAME:knn_ext>
  COMMENT "Symlinking knn_ext into python/"
)

# if(EXISTS "${CMAKE_SOURCE_DIR}/tests/test_knn.cu")
#   add_executable(test_knn tests/test_knn.cu)
#   target_link_libraries(test_knn PRIVATE knn_core)
# endif()
