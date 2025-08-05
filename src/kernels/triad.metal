#include <metal_stdlib>
using namespace metal;

// Triad kernel: A = B + α·C
// Memory-bound kernel with operational intensity ≈ 0.167 (2 FLOPs / 12 bytes)

kernel void triad_float(constant float& alpha [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device const float* c [[buffer(2)]],
                       device float* a [[buffer(3)]],
                       uint index [[thread_position_in_grid]],
                       uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    a[index] = b[index] + alpha * c[index];
}

kernel void triad_double(constant double& alpha [[buffer(0)]],
                        device const double* b [[buffer(1)]],
                        device const double* c [[buffer(2)]],
                        device double* a [[buffer(3)]],
                        uint index [[thread_position_in_grid]],
                        uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    a[index] = b[index] + alpha * c[index];
}

kernel void triad_half(constant half& alpha [[buffer(0)]],
                      device const half* b [[buffer(1)]],
                      device const half* c [[buffer(2)]],
                      device half* a [[buffer(3)]],
                      uint index [[thread_position_in_grid]],
                      uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    a[index] = b[index] + alpha * c[index];
}

// Vectorized version for better memory throughput
kernel void triad_float4(constant float& alpha [[buffer(0)]],
                        device const float4* b [[buffer(1)]],
                        device const float4* c [[buffer(2)]],
                        device float4* a [[buffer(3)]],
                        uint index [[thread_position_in_grid]],
                        uint grid_size [[threads_per_grid]]) {
    if (index >= grid_size) return;
    
    float4 b_vec = b[index];
    float4 c_vec = c[index];
    
    a[index] = b_vec + alpha * c_vec;
}

// SIMD group version for Apple Silicon optimization
kernel void triad_simd_float(constant float& alpha [[buffer(0)]],
                            device const float* b [[buffer(1)]],
                            device const float* c [[buffer(2)]],
                            device float* a [[buffer(3)]],
                            uint index [[thread_position_in_grid]],
                            uint grid_size [[threads_per_grid]],
                            uint simd_lane_id [[thread_index_in_simdgroup]],
                            uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
    if (index >= grid_size) return;
    
    // Use SIMD operations for better efficiency on Apple Silicon
    float b_val = b[index];
    float c_val = c[index];
    float result = b_val + alpha * c_val;
    
    a[index] = result;
}