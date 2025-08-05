# GPU Roofline Benchmark - FAQ

## General Questions

### Q: What hardware do I need to run this benchmark?
**A:** The toolkit supports three backends:
- **CUDA**: NVIDIA GPU with CUDA 12.x on Linux/Windows
- **Metal**: Apple Silicon Mac (M1/M2/M3) with Xcode 15+
- **CPU**: Any system with OpenMP support

You can run all backends that are available on your system.

### Q: How accurate are the roofline plots?
**A:** Accuracy depends on several factors:
- **Theoretical peaks**: Based on published specifications, may vary with boost clocks
- **Profiling quality**: CUDA (Nsight Compute) is most accurate, Metal requires estimation
- **Kernel efficiency**: Results show achieved performance, not theoretical limits

Expect ±10% variance from published numbers due to thermal throttling, driver versions, etc.

### Q: Why don't my kernels reach the roofline?
**A:** This is normal! The roofline represents theoretical peak performance. Real kernels are limited by:
- **Launch overhead**: Kernel startup costs
- **Memory access patterns**: Coalescing, bank conflicts, cache misses
- **Instruction mix**: Not all operations are peak-rate FMA instructions
- **Occupancy**: Insufficient parallelism to saturate the hardware

Good optimization targets 50-80% of roofline performance.

## Setup and Installation

### Q: CMake fails to find CUDA/Metal. What should I do?
**A:** 
- **CUDA**: Ensure `nvcc` is in your PATH and `CUDA_ROOT` is set
- **Metal**: Only works on macOS with Xcode command line tools installed
- **Workaround**: Use `--device cpu` to run CPU-only benchmarks

### Q: Can I run this in Docker or cloud environments?
**A:** 
- **CUDA**: Yes, with NVIDIA Docker runtime on cloud instances with GPUs
- **Metal**: No, Metal requires native macOS hardware
- **CPU**: Yes, works in any containerized environment

### Q: The build is very slow. How can I speed it up?
**A:**
- Use `cmake --build build --parallel` for parallel compilation
- Set `CMAKE_CUDA_ARCHITECTURES` to only your target GPU architecture
- Use `-DCMAKE_BUILD_TYPE=Release` for optimized builds

## Benchmark Results

### Q: What's a "good" operational intensity?
**A:** It depends on your algorithm:
- **Memory-bound** (OI < 1): SAXPY, TRIAD, sparse operations
- **Balanced** (OI 1-10): Dense linear algebra, convolution
- **Compute-bound** (OI > 10): GEMM, FFT, iterative solvers

Most real applications fall in the 0.1-10 range.

### Q: Why do small problem sizes show poor performance?
**A:** Several factors affect small kernels:
- **Launch overhead**: Fixed cost dominates for small workloads
- **Low occupancy**: Insufficient work to fill all compute units
- **Memory subsystem**: Caches can't amortize access patterns

Try problem sizes that result in >1ms execution time for stable measurements.

### Q: How do I interpret efficiency percentages?
**A:**
- **>80%**: Excellent optimization
- **50-80%**: Good performance, minor optimizations possible
- **20-50%**: Moderate performance, significant optimization opportunity
- **<20%**: Poor utilization, major algorithmic or implementation issues

## Advanced Usage

### Q: Can I add custom kernels?
**A:** Yes! Follow these steps:
1. Implement your kernel in `src/kernels/your_kernel.cu/.metal`
2. Add configuration to `bench.yaml` with operational intensity
3. Update the backend runners to expose your kernel
4. The plotting system will automatically include it

### Q: How do I compare different precisions?
**A:** Run benchmarks with different precision settings:
```bash
python run.py --kernels sgemm --precision float16 float32 float64
```

The plotter will show multiple rooflines for different compute bounds.

### Q: Can I benchmark real application kernels?
**A:** Absolutely! The framework is designed for extensibility:
1. Extract your kernel into the toolkit format
2. Estimate operational intensity (FLOPs / bytes accessed)
3. Add appropriate profiling to capture memory traffic
4. Use the same analysis pipeline

### Q: How do I get memory traffic numbers for new kernels?
**A:**
- **CUDA**: Use `ncu --metrics dram__bytes.sum`
- **Metal**: Use Instruments GPU timeline or estimate from data structure sizes
- **CPU**: Use `perf stat -e cache-misses` or similar tools

## Troubleshooting

### Q: Benchmarks crash with "out of memory" errors
**A:** 
- Reduce problem sizes in `bench.yaml`
- Check available GPU memory with `nvidia-smi` or similar tools
- Some kernels (like GEMM) have O(N²) memory requirements

### Q: Results vary significantly between runs
**A:**
- Ensure GPU isn't being used by other processes
- Check for thermal throttling (GPU temperature)
- Increase warmup runs in configuration
- Use `nvidia-smi -l` to monitor GPU state during benchmarks

### Q: Plot generation fails with import errors
**A:** Install required Python packages:
```bash
pip install matplotlib pandas numpy pyyaml
```

### Q: CI/CD integration fails on GitHub Actions
**A:**
- GPU runners require self-hosted infrastructure
- Use CPU-only mode for basic CI: `--device cpu`
- Consider cloud GPU instances for automated benchmarking

## Performance Optimization Tips

### Q: How do I optimize memory-bound kernels?
**A:**
- **Vectorization**: Use float4/double2 loads when possible
- **Coalescing**: Ensure memory accesses are contiguous
- **Prefetching**: Overlap computation with memory transfers
- **Bandwidth**: Aim for 80%+ of theoretical memory bandwidth

### Q: How do I optimize compute-bound kernels?
**A:**
- **Occupancy**: Maximize active warps/wavefronts
- **Instruction mix**: Use fused multiply-add (FMA) operations
- **Tensor cores**: Use mixed precision (FP16) for AI workloads
- **Arithmetic intensity**: Increase FLOPs per byte transferred

### Q: Should I optimize for peak GFLOP/s or efficiency?
**A:** 
- **Peak GFLOP/s**: Good for benchmarking and marketing
- **Efficiency**: Better for real applications and energy consumption
- **Balance**: Aim for high efficiency first, then absolute performance

The roofline plot helps visualize this trade-off by showing how close you are to theoretical limits.