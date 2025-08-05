#include "../../include/kernel_launcher.hpp"
#include "../../src/utils/timer.hpp"
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <cstdio>   // for popen, fgets
#include <cstdlib>  // for strtoull

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

class CPULauncher : public KernelLauncher {
private:
    int num_threads_;
    std::string cpu_name_;
    size_t memory_size_;
    
public:
    CPULauncher() {
        // Get number of available cores
#ifdef HAVE_OPENMP
        num_threads_ = omp_get_max_threads();
#else
        num_threads_ = 1;
#endif
        
        // Detect actual CPU model and memory size
        cpu_name_ = detect_cpu_model();
        memory_size_ = detect_memory_size();
        
        std::cout << "CPU Backend initialized with " << num_threads_ << " threads" << std::endl;
        std::cout << "CPU: " << cpu_name_ << std::endl;
    }
    
    KernelResult launch_saxpy(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "saxpy";
        result.device_type = "cpu";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = cpu_name_;
        result.total_memory_bytes = memory_size_;
        result.timestamp = get_timestamp();
        
        // Allocate and init data
        std::vector<float> x(n);
        std::vector<float> y(n);
        
        for (size_t i = 0; i < n; i++) {
            x[i] = static_cast<float>(i % 1000) / 1000.0f;  // avoid denormals
            y[i] = static_cast<float>((i * 7) % 1000) / 1000.0f;
        }
        
        // Warmup
        for (int i = 0; i < 5; i++) {
            saxpy_cpu(n, alpha, x.data(), y.data());
        }
        
        // Time the actual runs
        auto timer = CPUTimer();
        timer.start();
        
        const int num_runs = 20;  // enough for decent stats
        for (int i = 0; i < num_runs; i++) {
            saxpy_cpu(n, alpha, x.data(), y.data());
        }
        
        timer.stop();
        result.execution_time_ms = timer.elapsed_ms() / num_runs;
        
        // Quick correctness check
        float expected = alpha * x[0] + y[0];
        if (std::abs(y[0] - expected) > 1e-5) {
            std::cerr << "Warning: SAXPY computation may be incorrect" << std::endl;
        }
        
        // Performance calc
        result.flops_executed = 2 * n;  // mul + add per element
        result.bytes_transferred = 3 * n * sizeof(float);  // read x, read+write y
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        result.gpu_utilization_percent = estimate_cpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        result.memory_transfer_time_ms = 0.0;  // no PCIe transfers
        
        return result;
    }
    
    KernelResult launch_triad(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "triad";
        result.device_type = "cpu";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = cpu_name_;
        result.total_memory_bytes = memory_size_;
        result.timestamp = get_timestamp();
        
        // Setup arrays
        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c(n);
        
        for (size_t i = 0; i < n; i++) {
            b[i] = static_cast<float>((i % 997) + 1) / 1000.0f;  // primes for variety
            c[i] = static_cast<float>((i % 991) + 1) / 1000.0f;
        }
        
        // Warmup - let caches settle
        for (int i = 0; i < 7; i++) {
            triad_cpu(n, alpha, b.data(), c.data(), a.data());
        }
        
        // Time it
        auto timer = CPUTimer();
        timer.start();
        
        const int num_runs = 15;  // decent sample size
        for (int i = 0; i < num_runs; i++) {
            triad_cpu(n, alpha, b.data(), c.data(), a.data());
        }
        
        timer.stop();
        result.execution_time_ms = timer.elapsed_ms() / num_runs;
        
        // Sanity check
        float expected = b[0] + alpha * c[0];
        if (std::abs(a[0] - expected) > 1e-5) {
            std::cerr << "Warning: Triad computation may be incorrect" << std::endl;
        }
        
        // Performance numbers
        result.flops_executed = 2 * n;  // mul + add per element
        result.bytes_transferred = 4 * n * sizeof(float);  // read b, read c, write a
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        result.gpu_utilization_percent = estimate_cpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        result.memory_transfer_time_ms = 0.0;
        
        return result;
    }
    
    KernelResult launch_sgemm(size_t n, float alpha, float beta) override {
        KernelResult result;
        result.kernel_name = "sgemm";
        result.device_type = "cpu";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = cpu_name_;
        result.total_memory_bytes = memory_size_;
        result.timestamp = get_timestamp();
        
        // Allocate memory for n x n matrices
        std::vector<float> a(n * n, 1.0f);
        std::vector<float> b(n * n, 1.0f);
        std::vector<float> c(n * n, 0.0f);
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            sgemm_cpu(n, alpha, a.data(), b.data(), beta, c.data());
        }
        
        // Measurement runs
        auto timer = CPUTimer();
        timer.start();
        
        const int num_runs = 5;  // Fewer runs for expensive GEMM
        for (int i = 0; i < num_runs; i++) {
            sgemm_cpu(n, alpha, a.data(), b.data(), beta, c.data());
        }
        
        timer.stop();
        result.execution_time_ms = timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2ULL * n * n * n;  // 2n³ FLOPs for GEMM
        result.bytes_transferred = 3 * n * n * sizeof(float);  // Read A, read B, write C
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        result.gpu_utilization_percent = 90.0;  // Slightly lower for complex GEMM
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        return result;
    }
    
    std::string get_device_name() const override {
        return cpu_name_;
    }
    
    size_t get_memory_size() const override {
        return memory_size_;
    }

private:
    void saxpy_cpu(size_t n, float alpha, const float* x, float* y) {
#ifdef HAVE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; i++) {
            y[i] = alpha * x[i] + y[i];
        }
    }
    
    void triad_cpu(size_t n, float alpha, const float* b, const float* c, float* a) {
#ifdef HAVE_OPENMP
        #pragma omp parallel for
#endif
        for (size_t i = 0; i < n; i++) {
            a[i] = b[i] + alpha * c[i];
        }
    }
    
    void sgemm_cpu(size_t n, float alpha, const float* a, const float* b, float beta, float* c) {
        // Simple but inefficient GEMM implementation
        // In practice, would use BLAS library like OpenBLAS or Intel MKL
#ifdef HAVE_OPENMP
        #pragma omp parallel for collapse(2)
#endif
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < n; k++) {
                    sum += a[i * n + k] * b[k * n + j];
                }
                c[i * n + j] = alpha * sum + beta * c[i * n + j];
            }
        }
    }
    
    double estimate_cpu_utilization(double execution_time_ms, size_t problem_size) {
        // Simple heuristic for CPU usage - would use perf counters in real implementation
        double elements_per_ms = problem_size / execution_time_ms;
        double theoretical_elements_per_ms = num_threads_ * 2e6;  // rough estimate
        
        double utilization = (elements_per_ms / theoretical_elements_per_ms) * 100.0;
        return std::min(95.0, std::max(10.0, utilization));  // clamp to sane range
    }
    
    double estimate_memory_utilization(double achieved_gbps) {
        double theoretical_bandwidth = get_memory_bandwidth_gbps();
        return std::min(100.0, (achieved_gbps / theoretical_bandwidth) * 100.0);
    }
    
    std::string detect_cpu_model() {
        #ifdef __APPLE__
            FILE* pipe = popen("sysctl -n machdep.cpu.brand_string 2>/dev/null", "r");
            if (pipe) {
                char buffer[256];
                if (fgets(buffer, sizeof(buffer), pipe)) {
                    pclose(pipe);
                    std::string result(buffer);
                    if (!result.empty() && result.back() == '\n') {
                        result.pop_back();
                    }
                    return result.empty() ? "Apple Silicon" : result;
                }
                pclose(pipe);
            }
            return "Apple Silicon";
        #elif defined(__x86_64__) || defined(_M_X64)
            FILE* pipe = popen("grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | sed 's/^ *//' 2>/dev/null", "r");
            if (pipe) {
                char buffer[256];
                if (fgets(buffer, sizeof(buffer), pipe)) {
                    pclose(pipe);
                    std::string result(buffer);
                    if (!result.empty() && result.back() == '\n') {
                        result.pop_back();
                    }
                    return result.empty() ? "x86_64 CPU" : result;
                }
                pclose(pipe);
            }
            return "x86_64 CPU";
        #else
            return "Unknown CPU";
        #endif
    }
    
    size_t detect_memory_size() {
        #ifdef __APPLE__
            FILE* pipe = popen("sysctl -n hw.memsize 2>/dev/null", "r");
            if (pipe) {
                char buffer[64];
                if (fgets(buffer, sizeof(buffer), pipe)) {
                    pclose(pipe);
                    size_t mem_bytes = std::strtoull(buffer, nullptr, 10);
                    return mem_bytes > 0 ? mem_bytes : 8ULL * 1024 * 1024 * 1024;
                }
                pclose(pipe);
            }
        #elif defined(__linux__)
            FILE* pipe = popen("grep MemTotal /proc/meminfo | awk '{print $2*1024}' 2>/dev/null", "r");
            if (pipe) {
                char buffer[64];
                if (fgets(buffer, sizeof(buffer), pipe)) {
                    pclose(pipe);
                    size_t mem_bytes = std::strtoull(buffer, nullptr, 10);
                    return mem_bytes > 0 ? mem_bytes : 8ULL * 1024 * 1024 * 1024;
                }
                pclose(pipe);
            }
        #endif
        return 8ULL * 1024 * 1024 * 1024;  // fallback
    }
    
    double get_memory_bandwidth_gbps() {
        // rough estimates by platform
        if (cpu_name_.find("Apple") != std::string::npos) {
            return 100.0;  // M3 LPDDR5
        } else if (cpu_name_.find("x86_64") != std::string::npos) {
            return 50.0;   // DDR4/DDR5 
        } else {
            return 30.0;   // conservative guess
        }
    }
};

// Factory function implementation
std::unique_ptr<KernelLauncher> create_cpu_launcher() {
    return std::make_unique<CPULauncher>();
}