cmake_minimum_required(VERSION 3.18)
project(ads_metrics_eval LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)

add_library(ads_metrics_eval SHARED
    src/pairwise_distance.cu
    src/topk_select.cu
    src/knn.cu
    src/graph_properties.cu
    src/mutual_graph.cu
)

target_include_directories(ads_metrics_eval PUBLIC include)
target_link_libraries(ads_metrics_eval PRIVATE CUDA::cublas)

pybind11_add_module(knn_ext python/bindings.cpp)
target_link_libraries(knn_ext PRIVATE ads_metrics_eval)
