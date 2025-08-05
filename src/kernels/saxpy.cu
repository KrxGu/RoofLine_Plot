// SAXPY kernel: Y = αX + Y
// Memory-bound, low arithmetic intensity (~0.25)

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void saxpy_kernel_float(int n, float alpha, const float* x, float* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

__global__ void saxpy_kernel_double(int n, double alpha, const double* x, double* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// float4 version for better bandwidth
__global__ void saxpy_kernel_float4(int n, float alpha, const float4* x, float4* y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx * 4 < n) {
        float4 x_vec = x[idx];
        float4 y_vec = y[idx];
        
        y_vec.x = alpha * x_vec.x + y_vec.x;
        y_vec.y = alpha * x_vec.y + y_vec.y;
        y_vec.z = alpha * x_vec.z + y_vec.z;
        y_vec.w = alpha * x_vec.w + y_vec.w;
        
        y[idx] = y_vec;
    }
}

// host launch functions
extern "C" {
    // basic float version
    void launch_saxpy_float(int n, float alpha, const float* x, float* y, 
                          int block_size = 256) {
        int grid_size = (n + block_size - 1) / block_size;
        saxpy_kernel_float<<<grid_size, block_size>>>(n, alpha, x, y);
    }
    
    void launch_saxpy_double(int n, double alpha, const double* x, double* y,
                           int block_size = 256) {
        int grid_size = (n + block_size - 1) / block_size;
        saxpy_kernel_double<<<grid_size, block_size>>>(n, alpha, x, y);
    }
    
    // vectorized version for better bandwidth
    void launch_saxpy_float4(int n, float alpha, const float4* x, float4* y,
                           int block_size = 64) {
        int grid_size = ((n/4) + block_size - 1) / block_size;
        saxpy_kernel_float4<<<grid_size, block_size>>>(n, alpha, x, y);
    }
}