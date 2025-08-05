#include "../../include/kernel_launcher.hpp"
#include "../../src/utils/timer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

// Forward declarations for kernel launches
extern "C" {
    void launch_saxpy_float(int n, float alpha, const float* x, float* y, int block_size);
    void launch_saxpy_double(int n, double alpha, const double* x, double* y, int block_size);
    void launch_triad_float(int n, float alpha, const float* b, const float* c, float* a, int block_size);
    void launch_triad_double(int n, double alpha, const double* b, const double* c, double* a, int block_size);
}

class CUDALauncher : public KernelLauncher {
private:
    int device_id_;
    cudaDeviceProp device_props_;
    cublasHandle_t cublas_handle_;
    
    // Profiling support using Nsight Compute if available
    bool profiling_enabled_;
    std::string ncu_command_;
    
public:
    CUDALauncher(int device_id = 0) : device_id_(device_id), profiling_enabled_(false) {
        // Initialize CUDA
        cudaSetDevice(device_id_);
        cudaGetDeviceProperties(&device_props_, device_id_);
        
        // Initialize cuBLAS for GEMM kernels
        cublasCreate(&cublas_handle_);
        
        // Check if ncu (Nsight Compute) is available for profiling
        check_profiling_availability();
        
        std::cout << "CUDA Backend initialized on: " << device_props_.name << std::endl;
        std::cout << "Compute Capability: " << device_props_.major << "." << device_props_.minor << std::endl;
        std::cout << "Global Memory: " << device_props_.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    }
    
    ~CUDALauncher() {
        cublasDestroy(cublas_handle_);
    }
    
    void check_profiling_availability() {
        // Check if ncu command is available
        int result = system("which ncu > /dev/null 2>&1");
        profiling_enabled_ = (result == 0);
        
        if (profiling_enabled_) {
            std::cout << "Nsight Compute detected - enabling detailed profiling" << std::endl;
        } else {
            std::cout << "Nsight Compute not found - using basic timing only" << std::endl;
        }
    }
    
    KernelResult launch_saxpy(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "saxpy";
        result.device_type = "cuda";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = device_props_.name;
        result.total_memory_bytes = device_props_.totalGlobalMem;
        result.timestamp = get_timestamp();
        
        // Allocate device memory
        float *d_x, *d_y;
        size_t bytes = n * sizeof(float);
        
        cudaMalloc(&d_x, bytes);
        cudaMalloc(&d_y, bytes);
        
        // Initialize host data
        std::vector<float> h_x(n, 1.0f);
        std::vector<float> h_y(n, 2.0f);
        
        // Copy to device
        auto transfer_timer = CUDATimer();
        transfer_timer.start();
        cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice);
        transfer_timer.stop();
        result.memory_transfer_time_ms = transfer_timer.elapsed_ms();
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            launch_saxpy_float(n, alpha, d_x, d_y, 256);
            cudaDeviceSynchronize();
        }
        
        // Measurement runs
        auto compute_timer = CUDATimer();
        compute_timer.start();
        
        const int num_runs = 10;
        for (int i = 0; i < num_runs; i++) {
            launch_saxpy_float(n, alpha, d_x, d_y, 256);
        }
        
        compute_timer.stop();
        result.execution_time_ms = compute_timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2 * n;  // alpha * x[i] + y[i] = 2 FLOPs per element
        result.bytes_transferred = 2 * bytes;  // Read x, read+write y
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        // GPU utilization (basic estimation)
        result.gpu_utilization_percent = estimate_gpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        // Cleanup
        cudaFree(d_x);
        cudaFree(d_y);
        
        return result;
    }
    
    KernelResult launch_triad(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "triad";
        result.device_type = "cuda";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = device_props_.name;
        result.total_memory_bytes = device_props_.totalGlobalMem;
        result.timestamp = get_timestamp();
        
        // Allocate device memory
        float *d_a, *d_b, *d_c;
        size_t bytes = n * sizeof(float);
        
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        
        // Initialize host data
        std::vector<float> h_b(n, 1.0f);
        std::vector<float> h_c(n, 2.0f);
        
        // Copy to device
        auto transfer_timer = CUDATimer();
        transfer_timer.start();
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice);
        transfer_timer.stop();
        result.memory_transfer_time_ms = transfer_timer.elapsed_ms();
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            launch_triad_float(n, alpha, d_b, d_c, d_a, 256);
            cudaDeviceSynchronize();
        }
        
        // Measurement runs
        auto compute_timer = CUDATimer();
        compute_timer.start();
        
        const int num_runs = 10;
        for (int i = 0; i < num_runs; i++) {
            launch_triad_float(n, alpha, d_b, d_c, d_a, 256);
        }
        
        compute_timer.stop();
        result.execution_time_ms = compute_timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2 * n;  // b[i] + alpha * c[i] = 2 FLOPs per element
        result.bytes_transferred = 3 * bytes;  // Read b, read c, write a
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        // GPU utilization estimates
        result.gpu_utilization_percent = estimate_gpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        return result;
    }
    
    KernelResult launch_sgemm(size_t n, float alpha, float beta) override {
        KernelResult result;
        result.kernel_name = "sgemm";
        result.device_type = "cuda";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = device_props_.name;
        result.total_memory_bytes = device_props_.totalGlobalMem;
        result.timestamp = get_timestamp();
        
        // Allocate device memory for n x n matrices
        float *d_a, *d_b, *d_c;
        size_t bytes = n * n * sizeof(float);
        
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);
        
        // Initialize with random data
        std::vector<float> h_a(n * n, 1.0f);
        std::vector<float> h_b(n * n, 1.0f);
        std::vector<float> h_c(n * n, 0.0f);
        
        // Copy to device
        auto transfer_timer = CUDATimer();
        transfer_timer.start();
        cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, h_c.data(), bytes, cudaMemcpyHostToDevice);
        transfer_timer.stop();
        result.memory_transfer_time_ms = transfer_timer.elapsed_ms();
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
            cudaDeviceSynchronize();
        }
        
        // Measurement runs
        auto compute_timer = CUDATimer();
        compute_timer.start();
        
        const int num_runs = 10;
        for (int i = 0; i < num_runs; i++) {
            cublasSgemm(cublas_handle_, CUBLAS_OP_N, CUBLAS_OP_N,
                       n, n, n, &alpha, d_a, n, d_b, n, &beta, d_c, n);
        }
        
        compute_timer.stop();
        result.execution_time_ms = compute_timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2ULL * n * n * n;  // Standard GEMM: 2n³ FLOPs
        result.bytes_transferred = 3 * bytes;  // Read A, read B, write C
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        result.gpu_utilization_percent = estimate_gpu_utilization(result.execution_time_ms, n * n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        // Cleanup
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        
        return result;
    }
    
    std::string get_device_name() const override {
        return std::string(device_props_.name);
    }
    
    size_t get_memory_size() const override {
        return device_props_.totalGlobalMem;
    }

private:
    double estimate_gpu_utilization(double execution_time_ms, size_t problem_size) {
        // Simple heuristic based on problem size and execution time
        // This is a placeholder - real utilization would come from profiling
        double theoretical_time = problem_size / (device_props_.clockRate * 1000.0);
        return std::min(100.0, (theoretical_time / execution_time_ms) * 100.0);
    }
    
    double estimate_memory_utilization(double achieved_gbps) {
        // Estimate based on theoretical memory bandwidth
        // For most modern GPUs, this is around 500-1000 GB/s
        double theoretical_bandwidth = 800.0;  // GB/s - should be device-specific
        return std::min(100.0, (achieved_gbps / theoretical_bandwidth) * 100.0);
    }
};

// Factory function implementation
std::unique_ptr<KernelLauncher> create_cuda_launcher() {
    return std::make_unique<CUDALauncher>();
}