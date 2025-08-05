// Triad kernel: A = B + α·C  
// Memory-bound kernel with operational intensity ≈ 0.167 (2 FLOPs / 12 bytes)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void triad_kernel_float(int n, float alpha, const float* b, const float* c, float* a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        a[idx] = b[idx] + alpha * c[idx];
    }
}

__global__ void triad_kernel_double(int n, double alpha, const double* b, const double* c, double* a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        a[idx] = b[idx] + alpha * c[idx];
    }
}

// Vectorized version
__global__ void triad_kernel_float4(int n, float alpha, const float4* b, const float4* c, float4* a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 4 < n) {
        float4 b_vec = b[idx];
        float4 c_vec = c[idx];
        float4 result;
        
        result.x = b_vec.x + alpha * c_vec.x;
        result.y = b_vec.y + alpha * c_vec.y;
        result.z = b_vec.z + alpha * c_vec.z;
        result.w = b_vec.w + alpha * c_vec.w;
        
        a[idx] = result;
    }
}

// Host interface functions
extern "C" {
    void launch_triad_float(int n, float alpha, const float* b, const float* c, float* a,
                          int block_size = 256) {
        int grid_size = (n + block_size - 1) / block_size;
        triad_kernel_float<<<grid_size, block_size>>>(n, alpha, b, c, a);
    }
    
    void launch_triad_double(int n, double alpha, const double* b, const double* c, double* a,
                           int block_size = 256) {
        int grid_size = (n + block_size - 1) / block_size;
        triad_kernel_double<<<grid_size, block_size>>>(n, alpha, b, c, a);
    }
    
    void launch_triad_float4(int n, float alpha, const float4* b, const float4* c, float4* a,
                           int block_size = 64) {
        int grid_size = ((n/4) + block_size - 1) / block_size;
        triad_kernel_float4<<<grid_size, block_size>>>(n, alpha, b, c, a);
    }
}