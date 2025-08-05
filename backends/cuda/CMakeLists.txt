cmake_minimum_required(VERSION 3.20)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Create CUDA backend library
add_library(cuda_backend SHARED
    cuda_runner.cpp
    ../../src/kernels/saxpy.cu
    ../../src/kernels/triad.cu
)

# Set CUDA properties
set_property(TARGET cuda_backend PROPERTY CUDA_STANDARD 17)
set_property(TARGET cuda_backend PROPERTY CUDA_STANDARD_REQUIRED ON)

# Link CUDA libraries
target_link_libraries(cuda_backend 
    CUDA::cudart 
    CUDA::cublas
)

# Include directories
target_include_directories(cuda_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Set CUDA architecture (adjust based on target GPU)
set_property(TARGET cuda_backend PROPERTY CUDA_ARCHITECTURES "70;75;80;86;89;90")

# Compiler flags for optimization
target_compile_options(cuda_backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

# Install targets
install(TARGETS cuda_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)