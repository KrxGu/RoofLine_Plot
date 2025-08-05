# GPU Roofline Benchmark - Technical Overview

## What is a Roofline Plot?

A roofline plot is a visual performance model that helps developers understand the performance characteristics of their code on different hardware platforms. Originally proposed by Williams et al. in their 2009 paper, the roofline model plots performance (GFLOP/s) against operational intensity (FLOPs/Byte) on log-log axes.

## Key Concepts

### Operational Intensity (OI)
- **Definition**: The ratio of floating-point operations to bytes transferred from memory
- **Formula**: OI = FLOPs / Bytes
- **Interpretation**: Higher OI indicates more compute-intensive kernels

### Performance Bounds
1. **Memory Bandwidth Bound**: Performance limited by memory throughput
   - Diagonal line: Performance = Bandwidth × OI
   - Dominates for low operational intensity kernels

2. **Compute Bound**: Performance limited by arithmetic throughput
   - Horizontal line at peak GFLOP/s
   - Dominates for high operational intensity kernels

### The Roofline
The actual performance ceiling is the minimum of these two bounds, creating the characteristic "roofline" shape.

## Benchmark Kernels

### Memory-Bound Kernels

**SAXPY: Y = αX + Y**
- Operational Intensity: ~0.25 (2 FLOPs / 8 bytes)
- Memory Pattern: Stream through two vectors
- Purpose: Tests memory bandwidth efficiency

**Triad: A = B + α·C**
- Operational Intensity: ~0.167 (2 FLOPs / 12 bytes)
- Memory Pattern: Read two vectors, write one
- Purpose: Classic STREAM benchmark kernel

### Compute-Bound Kernels

**SGEMM: C = αAB + βC**
- Operational Intensity: ~85 (2N³ FLOPs / 3N² × 4 bytes for N×N matrices)
- Memory Pattern: Matrix multiplication with data reuse
- Purpose: Tests peak arithmetic performance

**WMMA GEMM: Tensor Core Matrix Multiplication**
- Operational Intensity: ~85+ (same as SGEMM but with mixed precision)
- Memory Pattern: FP16 input, FP32 accumulation
- Purpose: Tests specialized AI compute units

## Architecture

### Backend Abstraction
The toolkit uses a plugin architecture where each compute backend (CUDA, Metal, CPU) implements a common `KernelLauncher` interface:

```cpp
class KernelLauncher {
public:
    virtual KernelResult launch_saxpy(size_t n, float alpha) = 0;
    virtual KernelResult launch_triad(size_t n, float alpha) = 0;
    virtual KernelResult launch_sgemm(size_t n, float alpha, float beta) = 0;
    // ...
};
```

### Data Flow
1. **Compilation**: CMake builds backend-specific libraries
2. **Execution**: `run.py` orchestrates kernel launches and profiling
3. **Collection**: `collect.py` normalizes raw metrics into CSV
4. **Visualization**: `plot_roofline.py` generates roofline plots

### Performance Metrics
Each kernel run captures:
- **Timing**: Execution time via GPU events or high-resolution CPU timers
- **Throughput**: GFLOP/s and GB/s achieved
- **Utilization**: GPU and memory subsystem efficiency
- **Traffic**: Actual bytes transferred (via profiling tools)

## Profiling Integration

### CUDA: Nsight Compute
- Command: `ncu --metrics dram__bytes.sum,sm__cycles_elapsed.avg`
- Captures: Memory traffic, SM utilization, instruction throughput
- Output: CSV metrics for detailed analysis

### Metal: Instruments
- Command: `xcrun xctrace record --template "GPU Counters"`
- Captures: GPU bandwidth, shader core utilization
- Output: XML trace parsed for bandwidth metrics

### CPU: Performance Counters
- Tools: `perf` on Linux, native timing on other platforms
- Captures: Cache misses, memory bandwidth, CPU utilization

## Device Capability Database

The `device_caps.json` file stores theoretical peak performance for different GPUs:

```json
{
  "nvidia_a100": {
    "peak_bandwidth_gb_s": 1935,
    "peak_compute_gflops": {
      "float32": 19500,
      "float16": 78000,
      "float64": 9700
    }
  }
}
```

This enables accurate roofline bounds and efficiency calculations.

## Extending the Toolkit

### Adding New Kernels
1. Implement kernel in `src/kernels/` for each backend
2. Add configuration to `bench.yaml`
3. Update backend runners to expose new kernel
4. Add to orchestration scripts

### Supporting New Devices
1. Add device detection to `run.py`
2. Implement backend in `backends/new_device/`
3. Update CMake configuration
4. Add device capabilities to database

### Performance Analysis
The toolkit generates multiple analysis outputs:
- **PNG plots**: For papers and presentations
- **Interactive HTML**: For detailed exploration
- **CSV data**: For custom analysis in R/Python
- **JSON logs**: For integration with other tools

## References

1. Williams, S., Waterman, A., & Patterson, D. (2009). "Roofline: an insightful visual performance model for multicore architectures." Communications of the ACM, 52(4), 65-76.

2. Ilic, A., Pratas, F., & Sousa, L. (2014). "Cache-aware roofline model: Upgrading the loft." IEEE Computer Architecture Letters, 13(1), 21-24.

3. Yang, C., Kurth, T., & Williams, S. (2018). "Hierarchical roofline analysis for GPUs: Accelerating performance optimization for the NERSC-9 Perlmutter system." Concurrency and Computation: Practice and Experience, 32(20), e5547.