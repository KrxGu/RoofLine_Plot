#include <metal_stdlib>
using namespace metal;

// SAXPY kernel: Y = αX + Y
// Memory-bound kernel with operational intensity ≈ 0.25 (2 FLOPs / 8 bytes)

kernel void saxpy_float(constant float& alpha [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]],
                       uint index [[thread_position_in_grid]],
                       uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    y[index] = alpha * x[index] + y[index];
}

kernel void saxpy_double(constant double& alpha [[buffer(0)]],
                        device const double* x [[buffer(1)]],
                        device double* y [[buffer(2)]],
                        uint index [[thread_position_in_grid]],
                        uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    y[index] = alpha * x[index] + y[index];
}

kernel void saxpy_half(constant half& alpha [[buffer(0)]],
                      device const half* x [[buffer(1)]],
                      device half* y [[buffer(2)]],
                      uint index [[thread_position_in_grid]],
                      uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    y[index] = alpha * x[index] + y[index];
}

// Vectorized version for better memory throughput
kernel void saxpy_float4(constant float& alpha [[buffer(0)]],
                        device const float4* x [[buffer(1)]],
                        device float4* y [[buffer(2)]],
                        uint index [[thread_position_in_grid]],
                        uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    float4 x_vec = x[index];
    float4 y_vec = y[index];
    
    y[index] = alpha * x_vec + y_vec;
}

// SIMD group version for Apple Silicon optimization
kernel void saxpy_simd_float(constant float& alpha [[buffer(0)]],
                            device const float* x [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            uint index [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    if (index >= grid_size) return;
    
    // Use SIMD operations for better efficiency on Apple Silicon
    float x_val = x[index];
    float y_val = y[index];
    float result = alpha * x_val + y_val;
    
    y[index] = result;
}