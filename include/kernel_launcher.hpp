#pragma once

#include <string>
#include <vector>
#include <chrono>

// Results from running a kernel
struct KernelResult {
    std::string kernel_name;
    std::string device_type;  // "cuda", "metal", "cpu"
    std::string precision;    // "float16", "float32", "float64"
    size_t problem_size;
    
    // Timing
    double execution_time_ms;
    double memory_transfer_time_ms;
    
    // Performance counters
    double gflops_achieved;
    double gbps_achieved;
    double operational_intensity;
    
    // Raw metrics
    size_t flops_executed;
    size_t bytes_transferred;
    
    // utilization estimates
    double gpu_utilization_percent;
    double memory_utilization_percent;
    
    // metadata
    std::string timestamp;
    std::string device_name;
    size_t total_memory_bytes;
};

// Base class for running kernels on different backends
class KernelLauncher {
public:
    virtual ~KernelLauncher() = default;
    
    virtual KernelResult launch_saxpy(size_t n, float alpha) = 0;
    virtual KernelResult launch_triad(size_t n, float alpha) = 0;
    virtual KernelResult launch_sgemm(size_t n, float alpha, float beta) = 0;
    
    virtual std::string get_device_name() const = 0;
    virtual size_t get_memory_size() const = 0;
    
protected:
    // helper to calc operational intensity
    double compute_operational_intensity(size_t flops, size_t bytes) const {
        return bytes > 0 ? static_cast<double>(flops) / bytes : 0.0;
    }
    
    // get unix timestamp
    std::string get_timestamp() const {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        return std::to_string(time_t);
    }
};

// factory to create launcher for specific device
std::unique_ptr<KernelLauncher> create_launcher(const std::string& device_type);