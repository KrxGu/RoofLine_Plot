cmake_minimum_required(VERSION 3.20)

# Create CPU backend library
add_library(cpu_backend SHARED
    cpu_runner.cpp
)

# Set C++ standard
set_target_properties(cpu_backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(cpu_backend OpenMP::OpenMP_CXX)
    else()
        # Manual linking for macOS Homebrew case
        target_compile_options(cpu_backend PRIVATE ${OpenMP_CXX_FLAGS})
        target_link_libraries(cpu_backend ${OpenMP_omp_LIBRARY})
        target_include_directories(cpu_backend PRIVATE "/opt/homebrew/opt/libomp/include")
    endif()
    target_compile_definitions(cpu_backend PRIVATE HAVE_OPENMP=1)
else()
    message(WARNING "CPU backend compiled without OpenMP - performance will be limited")
    target_compile_definitions(cpu_backend PRIVATE HAVE_OPENMP=0)
endif()

# Include directories
target_include_directories(cpu_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Compiler flags for optimization
target_compile_options(cpu_backend PRIVATE
    -O3 -march=native -ffast-math
)

# Additional optimizations for different architectures
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # Apple Silicon optimizations
    target_compile_options(cpu_backend PRIVATE -mcpu=apple-m1)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # x86_64 optimizations
    target_compile_options(cpu_backend PRIVATE -mavx2 -mfma)
endif()

# Install targets
install(TARGETS cpu_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)