cmake_minimum_required(VERSION 3.20)

# Metal backend is only available on Apple platforms
if(NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
endif()

# Create Metal backend library
add_library(metal_backend SHARED
    metal_runner.mm
)

# Set language standards
set_target_properties(metal_backend PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

# Find required frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Link frameworks
target_link_libraries(metal_backend
    ${METAL_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Include directories
target_include_directories(metal_backend PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

# Compiler flags for optimization
target_compile_options(metal_backend PRIVATE
    -O3 -march=native
    -fobjc-arc  # Enable ARC for Objective-C++
)

# Add custom command to compile Metal shaders
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    COMMAND xcrun -sdk macosx metal -O3 -c ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/saxpy.metal -o ${CMAKE_CURRENT_BINARY_DIR}/saxpy.air
    COMMAND xcrun -sdk macosx metal -O3 -c ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/triad.metal -o ${CMAKE_CURRENT_BINARY_DIR}/triad.air
    COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/saxpy.air ${CMAKE_CURRENT_BINARY_DIR}/triad.air -o ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DEPENDS 
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/saxpy.metal
        ${CMAKE_CURRENT_SOURCE_DIR}/../../src/kernels/triad.metal
    COMMENT "Compiling Metal shaders"
)

# Add custom target for shader compilation
add_custom_target(metal_shaders ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
)

# Make metal_backend depend on shader compilation
add_dependencies(metal_backend metal_shaders)

# Install targets
install(TARGETS metal_backend
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install compiled shaders
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/default.metallib
    DESTINATION share/metal_shaders
)