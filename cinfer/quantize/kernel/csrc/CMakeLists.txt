project(FasterTransformer LANGUAGES CXX CUDA)

set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cutlass)

add_subdirectory(cutlass_kernels)
