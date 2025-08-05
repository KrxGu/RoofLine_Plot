cmake_minimum_required(VERSION 3.20)
project(gpu_roofline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for different backends
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_METAL "Enable Metal backend" OFF)
option(ENABLE_CPU "Enable CPU backend" ON)

# Auto-detect available backends
if(NOT ENABLE_CUDA AND NOT ENABLE_METAL AND NOT ENABLE_CPU)
    # Auto-detection logic
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(ENABLE_CUDA ON)
        message(STATUS "Auto-detected CUDA toolkit, enabling CUDA backend")
    endif()
    
    if(APPLE)
        set(ENABLE_METAL ON)
        message(STATUS "Apple platform detected, enabling Metal backend")
    endif()
    
    if(NOT ENABLE_CUDA AND NOT ENABLE_METAL)
        set(ENABLE_CPU ON)
        message(STATUS "Falling back to CPU backend")
    endif()
endif()

# Global include directory
include_directories(include)

# Add subdirectories based on enabled backends
if(ENABLE_CUDA)
    enable_language(CUDA)
    add_subdirectory(backends/cuda)
endif()

if(ENABLE_METAL)
    enable_language(OBJCXX)
    add_subdirectory(backends/metal)
endif()

if(ENABLE_CPU)
    # OpenMP is optional for CPU backend
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found - CPU backend will use parallel execution")
    else()
        message(WARNING "OpenMP not found - CPU backend will use serial execution")
                    # Help find Homebrew OpenMP on macOS
        if(APPLE AND EXISTS "/opt/homebrew/opt/libomp")
            message(STATUS "Attempting to use Homebrew OpenMP")
            set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
            set(OpenMP_CXX_LIB_NAMES "omp")
            set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
            set(OpenMP_CXX_FOUND TRUE)
        endif()
    endif()
    add_subdirectory(backends/cpu)
endif()

# Install headers
install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "GPU Roofline Configuration:")
message(STATUS "  CUDA backend:  ${ENABLE_CUDA}")
message(STATUS "  Metal backend: ${ENABLE_METAL}")
message(STATUS "  CPU backend:   ${ENABLE_CPU}")
message(STATUS "")