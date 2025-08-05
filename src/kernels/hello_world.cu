// Hello World CUDA kernel for toolchain verification
// Simple vector addition kernel to test compilation and execution

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void hello_world_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" {
    bool test_cuda_hello_world() {
        const int n = 1024;
        const size_t bytes = n * sizeof(float);
        
        // Host data
        float* h_a = new float[n];
        float* h_b = new float[n];
        float* h_c = new float[n];
        
        // Initialize
        for (int i = 0; i < n; i++) {
            h_a[i] = static_cast<float>(i);
            h_b[i] = static_cast<float>(i * 2);
        }
        
        // Device data
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        
        // Copy to device
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        hello_world_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
        
        // Check for errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(error) << std::endl;
            return false;
        }
        
        // Wait for completion
        cudaDeviceSynchronize();
        
        // Copy result back
        cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
        
        // Verify results
        bool success = true;
        for (int i = 0; i < n; i++) {
            float expected = h_a[i] + h_b[i];
            if (std::abs(h_c[i] - expected) > 1e-5) {
                std::cerr << "Verification failed at index " << i 
                          << ": expected " << expected 
                          << ", got " << h_c[i] << std::endl;
                success = false;
                break;
            }
        }
        
        if (success) {
            std::cout << "✓ CUDA hello world test passed!" << std::endl;
        }
        
        // Cleanup
        delete[] h_a;
        delete[] h_b;
        delete[] h_c;
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        return success;
    }
}

// Simple main function for testing
int main() {
    // Print device info
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA device(s)" << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        std::cout << "Device " << i << ": " << props.name 
                  << " (CC " << props.major << "." << props.minor << ")" << std::endl;
    }
    
    // Run test
    return test_cuda_hello_world() ? 0 : 1;
}