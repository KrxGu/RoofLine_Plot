#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "../../include/kernel_launcher.hpp"
#include "../../src/utils/timer.hpp"
#include <iostream>
#include <vector>
#include <fstream>
#include <memory>

@implementation MetalTimer
- (instancetype)initWithDevice:(id<MTLDevice>)device {
    self = [super init];
    if (self) {
        _device = device;
        _startEvent = [device newSharedEvent];
        _stopEvent = [device newSharedEvent];
        _startValue = 1;
        _stopValue = 1;
    }
    return self;
}

- (void)start:(id<MTLCommandBuffer>)commandBuffer {
    [commandBuffer encodeSignalEvent:_startEvent value:_startValue];
}

- (void)stop:(id<MTLCommandBuffer>)commandBuffer {
    [commandBuffer encodeSignalEvent:_stopEvent value:_stopValue];
}

- (double)elapsedMs {
    // Wait for events to complete
    [_startEvent waitForValue:_startValue timeoutMS:1000];
    [_stopEvent waitForValue:_stopValue timeoutMS:1000];
    
    // Note: This is a simplified timing approach
    // For accurate GPU timing on Metal, we'd need GPU timestamps
    // which require more complex implementation
    return 0.0; // Placeholder - would use MTLCounterSampleBuffer for real timing
}
@end

class MetalLauncher : public KernelLauncher {
private:
    id<MTLDevice> device_;
    id<MTLCommandQueue> command_queue_;
    id<MTLLibrary> library_;
    NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipeline_cache_;
    
    // Profiling support using Instruments if available
    bool profiling_enabled_;
    
public:
    MetalLauncher() : profiling_enabled_(false) {
        // Get default Metal device (Apple Silicon GPU)
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            throw std::runtime_error("Metal device not available");
        }
        
        command_queue_ = [device_ newCommandQueue];
        pipeline_cache_ = [[NSMutableDictionary alloc] init];
        
        // Compile Metal shaders
        compile_shaders();
        
        // Check for Instruments profiling
        check_profiling_availability();
        
        std::cout << "Metal Backend initialized on: " << [device_.name UTF8String] << std::endl;
        std::cout << "Memory: " << device_.recommendedMaxWorkingSetSize / (1024*1024*1024) << " GB" << std::endl;
    }
    
    ~MetalLauncher() {
        // ARC will handle cleanup
    }
    
    void compile_shaders() {
        NSError* error = nil;
        
        // Load Metal shader source
        NSString* shader_path = @"../../src/kernels/saxpy.metal";
        NSString* shader_source = [NSString stringWithContentsOfFile:shader_path 
                                                            encoding:NSUTF8StringEncoding 
                                                               error:&error];
        
        if (error) {
            // Fallback: use embedded shader source
            shader_source = @R"(
                #include <metal_stdlib>
                using namespace metal;
                
                kernel void saxpy_float(constant float& alpha [[buffer(0)]],
                                       device const float* x [[buffer(1)]],
                                       device float* y [[buffer(2)]],
                                       uint index [[thread_position_in_grid]],
                                       uint grid_size [[threads_per_grid]]) {
                    if (index >= grid_size) return;
                    y[index] = alpha * x[index] + y[index];
                }
                
                kernel void triad_float(constant float& alpha [[buffer(0)]],
                                       device const float* b [[buffer(1)]],
                                       device const float* c [[buffer(2)]],
                                       device float* a [[buffer(3)]],
                                       uint index [[thread_position_in_grid]],
                                       uint grid_size [[threads_per_grid]]) {
                    if (index >= grid_size) return;
                    a[index] = b[index] + alpha * c[index];
                }
            )";
        }
        
        // Compile library
        library_ = [device_ newLibraryWithSource:shader_source options:nil error:&error];
        if (error) {
            NSLog(@"Metal shader compilation error: %@", error.localizedDescription);
            throw std::runtime_error("Failed to compile Metal shaders");
        }
        
        // Create compute pipeline states
        create_pipeline_state(@"saxpy_float");
        create_pipeline_state(@"triad_float");
    }
    
    void create_pipeline_state(NSString* function_name) {
        NSError* error = nil;
        id<MTLFunction> function = [library_ newFunctionWithName:function_name];
        if (!function) {
            throw std::runtime_error("Metal function not found");
        }
        
        id<MTLComputePipelineState> pipeline = [device_ newComputePipelineStateWithFunction:function error:&error];
        if (error) {
            NSLog(@"Pipeline creation error: %@", error.localizedDescription);
            throw std::runtime_error("Failed to create Metal pipeline");
        }
        
        pipeline_cache_[function_name] = pipeline;
    }
    
    void check_profiling_availability() {
        // Check if Instruments command line tools are available
        int result = system("which xcrun > /dev/null 2>&1");
        profiling_enabled_ = (result == 0);
        
        if (profiling_enabled_) {
            std::cout << "Xcode command line tools detected - profiling available" << std::endl;
        } else {
            std::cout << "Xcode tools not found - using basic timing only" << std::endl;
        }
    }
    
    KernelResult launch_saxpy(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "saxpy";
        result.device_type = "metal";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = [device_.name UTF8String];
        result.total_memory_bytes = device_.recommendedMaxWorkingSetSize;
        result.timestamp = get_timestamp();
        
        // Allocate Metal buffers
        size_t bytes = n * sizeof(float);
        id<MTLBuffer> buffer_alpha = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_x = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_y = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        
        // Initialize data
        *((float*)buffer_alpha.contents) = alpha;
        float* x_ptr = (float*)buffer_x.contents;
        float* y_ptr = (float*)buffer_y.contents;
        
        for (size_t i = 0; i < n; i++) {
            x_ptr[i] = 1.0f;
            y_ptr[i] = 2.0f;
        }
        
        // Get pipeline state
        id<MTLComputePipelineState> pipeline = pipeline_cache_[@"saxpy_float"];
        
        // Configure thread groups
        MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
        MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            execute_kernel(pipeline, buffer_alpha, buffer_x, buffer_y, nil, thread_group_size, grid_size);
        }
        
        // Measurement runs
        auto timer = CPUTimer();  // Using CPU timer for simplicity
        timer.start();
        
        const int num_runs = 10;
        for (int i = 0; i < num_runs; i++) {
            execute_kernel(pipeline, buffer_alpha, buffer_x, buffer_y, nil, thread_group_size, grid_size);
        }
        
        timer.stop();
        result.execution_time_ms = timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2 * n;
        result.bytes_transferred = 2 * bytes;  // Read x, read+write y
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        // Estimate utilization (simplified)
        result.gpu_utilization_percent = estimate_gpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        return result;
    }
    
    KernelResult launch_triad(size_t n, float alpha) override {
        KernelResult result;
        result.kernel_name = "triad";
        result.device_type = "metal";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = [device_.name UTF8String];
        result.total_memory_bytes = device_.recommendedMaxWorkingSetSize;
        result.timestamp = get_timestamp();
        
        // Allocate Metal buffers
        size_t bytes = n * sizeof(float);
        id<MTLBuffer> buffer_alpha = [device_ newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_a = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_b = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer_c = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        
        // Initialize data
        *((float*)buffer_alpha.contents) = alpha;
        float* b_ptr = (float*)buffer_b.contents;
        float* c_ptr = (float*)buffer_c.contents;
        
        for (size_t i = 0; i < n; i++) {
            b_ptr[i] = 1.0f;
            c_ptr[i] = 2.0f;
        }
        
        // Get pipeline state
        id<MTLComputePipelineState> pipeline = pipeline_cache_[@"triad_float"];
        
        // Configure thread groups
        MTLSize thread_group_size = MTLSizeMake(256, 1, 1);
        MTLSize grid_size = MTLSizeMake((n + 255) / 256, 1, 1);
        
        // Warmup runs
        for (int i = 0; i < 3; i++) {
            execute_kernel(pipeline, buffer_alpha, buffer_b, buffer_c, buffer_a, thread_group_size, grid_size);
        }
        
        // Measurement runs
        auto timer = CPUTimer();
        timer.start();
        
        const int num_runs = 10;
        for (int i = 0; i < num_runs; i++) {
            execute_kernel(pipeline, buffer_alpha, buffer_b, buffer_c, buffer_a, thread_group_size, grid_size);
        }
        
        timer.stop();
        result.execution_time_ms = timer.elapsed_ms() / num_runs;
        
        // Calculate performance metrics
        result.flops_executed = 2 * n;
        result.bytes_transferred = 3 * bytes;  // Read b, read c, write a
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        result.gpu_utilization_percent = estimate_gpu_utilization(result.execution_time_ms, n);
        result.memory_utilization_percent = estimate_memory_utilization(result.gbps_achieved);
        
        return result;
    }
    
    KernelResult launch_sgemm(size_t n, float alpha, float beta) override {
        // For now, return a placeholder result
        // TODO: Implement Metal matrix multiplication
        KernelResult result;
        result.kernel_name = "sgemm";
        result.device_type = "metal";
        result.precision = "float32";
        result.problem_size = n;
        result.device_name = [device_.name UTF8String];
        result.total_memory_bytes = device_.recommendedMaxWorkingSetSize;
        result.timestamp = get_timestamp();
        
        // Placeholder values - would implement actual GEMM kernel
        result.execution_time_ms = 1.0;
        result.flops_executed = 2ULL * n * n * n;
        result.bytes_transferred = 3 * n * n * sizeof(float);
        result.operational_intensity = compute_operational_intensity(result.flops_executed, result.bytes_transferred);
        result.gflops_achieved = (result.flops_executed / 1e9) / (result.execution_time_ms / 1000.0);
        result.gbps_achieved = (result.bytes_transferred / 1e9) / (result.execution_time_ms / 1000.0);
        
        return result;
    }
    
    std::string get_device_name() const override {
        return std::string([device_.name UTF8String]);
    }
    
    size_t get_memory_size() const override {
        return device_.recommendedMaxWorkingSetSize;
    }

private:
    void execute_kernel(id<MTLComputePipelineState> pipeline, 
                       id<MTLBuffer> buffer1, 
                       id<MTLBuffer> buffer2, 
                       id<MTLBuffer> buffer3, 
                       id<MTLBuffer> buffer4,
                       MTLSize thread_group_size, 
                       MTLSize grid_size) {
        id<MTLCommandBuffer> command_buffer = [command_queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:buffer1 offset:0 atIndex:0];
        [encoder setBuffer:buffer2 offset:0 atIndex:1];
        [encoder setBuffer:buffer3 offset:0 atIndex:2];
        if (buffer4) {
            [encoder setBuffer:buffer4 offset:0 atIndex:3];
        }
        
        [encoder dispatchThreadgroups:grid_size threadsPerThreadgroup:thread_group_size];
        [encoder endEncoding];
        
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
    }
    
    double estimate_gpu_utilization(double execution_time_ms, size_t problem_size) {
        // Simplified estimation for Metal
        return std::min(100.0, (problem_size / (execution_time_ms * 1000.0)) / 1e6);
    }
    
    double estimate_memory_utilization(double achieved_gbps) {
        // Apple Silicon LPDDR bandwidth is typically around 100-400 GB/s
        double theoretical_bandwidth = 200.0;  // GB/s - device specific
        return std::min(100.0, (achieved_gbps / theoretical_bandwidth) * 100.0);
    }
};

// Factory function implementation
std::unique_ptr<KernelLauncher> create_metal_launcher() {
    return std::make_unique<MetalLauncher>();
}