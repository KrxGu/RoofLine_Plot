// CPU backend executable 
// Standalone program that run.py can call

#include "../../include/kernel_launcher.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>

// factory function from cpu_runner.cpp
std::unique_ptr<KernelLauncher> create_cpu_launcher();

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <kernel> <size> <alpha> [output_file]" << std::endl;
    std::cout << "  kernel: saxpy, triad, sgemm" << std::endl;
    std::cout << "  size: problem size (e.g., 1048576)" << std::endl;
    std::cout << "  alpha: scalar parameter" << std::endl;
    std::cout << "  output_file: JSON output file (optional)" << std::endl;
}

void write_result_json(const KernelResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open output file: " << filename << std::endl;
        return;
    }
    
    file << "{\n";
    file << "  \"kernel_name\": \"" << result.kernel_name << "\",\n";
    file << "  \"device_type\": \"" << result.device_type << "\",\n";
    file << "  \"precision\": \"" << result.precision << "\",\n";
    file << "  \"problem_size\": " << result.problem_size << ",\n";
    file << "  \"execution_time_ms\": " << result.execution_time_ms << ",\n";
    file << "  \"memory_transfer_time_ms\": " << result.memory_transfer_time_ms << ",\n";
    file << "  \"gflops_achieved\": " << result.gflops_achieved << ",\n";
    file << "  \"gbps_achieved\": " << result.gbps_achieved << ",\n";
    file << "  \"operational_intensity\": " << result.operational_intensity << ",\n";
    file << "  \"flops_executed\": " << result.flops_executed << ",\n";
    file << "  \"bytes_transferred\": " << result.bytes_transferred << ",\n";
    file << "  \"gpu_utilization_percent\": " << result.gpu_utilization_percent << ",\n";
    file << "  \"memory_utilization_percent\": " << result.memory_utilization_percent << ",\n";
    file << "  \"timestamp\": \"" << result.timestamp << "\",\n";
    file << "  \"device_name\": \"" << result.device_name << "\",\n";
    file << "  \"total_memory_bytes\": " << result.total_memory_bytes << "\n";
    file << "}\n";
    
    file.close();
    std::cout << "Results written to: " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string kernel = argv[1];
    size_t size = std::strtoull(argv[2], nullptr, 10);
    float alpha = std::strtof(argv[3], nullptr);
    std::string output_file = (argc >= 5) ? argv[4] : "";
    
    // Create CPU launcher
    auto launcher = create_cpu_launcher();
    if (!launcher) {
        std::cerr << "Error: Could not create CPU launcher" << std::endl;
        return 1;
    }
    
    // Execute the specified kernel
    KernelResult result;
    try {
        if (kernel == "saxpy") {
            result = launcher->launch_saxpy(size, alpha);
        } else if (kernel == "triad") {
            result = launcher->launch_triad(size, alpha);
        } else if (kernel == "sgemm") {
            result = launcher->launch_sgemm(size, alpha, 0.0f);  // beta = 0
        } else {
            std::cerr << "Error: Unknown kernel: " << kernel << std::endl;
            print_usage(argv[0]);
            return 1;
        }
        
        // print results
        std::cout << "Kernel: " << result.kernel_name << std::endl;
        std::cout << "Device: " << result.device_name << std::endl;
        std::cout << "Problem size: " << result.problem_size << std::endl;
        std::cout << "Execution time: " << result.execution_time_ms << " ms" << std::endl;
        std::cout << "Performance: " << result.gflops_achieved << " GFLOP/s" << std::endl;
        std::cout << "Bandwidth: " << result.gbps_achieved << " GB/s" << std::endl;
        std::cout << "Operational Intensity: " << result.operational_intensity << std::endl;
        
        // save JSON if needed
        if (!output_file.empty()) {
            write_result_json(result, output_file);
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error executing kernel: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}