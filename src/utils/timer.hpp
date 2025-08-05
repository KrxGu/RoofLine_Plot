#pragma once

#include <chrono>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#ifdef __METAL_VERSION__
#import <Metal/Metal.h>
#endif

// High-resolution timer for CPU benchmarks
class CPUTimer {
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        end_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0;
    }
};

#ifdef __CUDACC__
// CUDA event timer with RAII
class CUDATimer {
    cudaEvent_t start_event, stop_event;
    
public:
    CUDATimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    
    void start() {
        cudaEventRecord(start_event);
    }
    
    void stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }
    
    double elapsed_ms() const {
        float time_ms;
        cudaEventElapsedTime(&time_ms, start_event, stop_event);
        return static_cast<double>(time_ms);
    }
};
#endif

#ifdef __OBJC__
// Metal timer using MTLSharedEvent
@interface MetalTimer : NSObject
@property (nonatomic, strong) id<MTLDevice> device;
@property (nonatomic, strong) id<MTLSharedEvent> startEvent;
@property (nonatomic, strong) id<MTLSharedEvent> stopEvent;
@property (nonatomic) uint64_t startValue;
@property (nonatomic) uint64_t stopValue;

- (instancetype)initWithDevice:(id<MTLDevice>)device;
- (void)start:(id<MTLCommandBuffer>)commandBuffer;
- (void)stop:(id<MTLCommandBuffer>)commandBuffer;
- (double)elapsedMs;
@end
#endif