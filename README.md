# GPU Roofline Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A cross-platform performance analysis toolkit that generates **roofline plots** to visualize GPU and CPU performance ceilings. Measure memory bandwidth and compute throughput with real kernel execution on CUDA, Metal, and CPU backends.

![Roofline Plot Example](plots/roofline_cpu-cuda-metal.png)

## What This Application Does

This toolkit helps you **understand your hardware's performance limits** by:

1. **Running micro-benchmarks** - Executes memory-bound (SAXPY, Triad) and compute-bound (SGEMM) kernels
2. **Measuring real performance** - Times actual execution and calculates GFLOP/s and GB/s  
3. **Generating roofline plots** - Visualizes performance vs operational intensity to show bottlenecks
4. **Comparing devices** - Benchmarks CPU, CUDA GPUs, and Metal GPUs on the same chart

**The roofline model** shows two key limits:
- **Memory bandwidth ceiling** (diagonal line) - Limited by data movement
- **Compute throughput ceiling** (horizontal line) - Limited by arithmetic units

Your kernels appear as points, showing whether they're hitting memory or compute limits.

## 🚀 Quick Start

### Prerequisites

**Required for all platforms:**
- Python 3.10+ 
- CMake ≥ 3.18
- C++ compiler (GCC, Clang, or MSVC)

**Optional for GPU acceleration:**
- **CUDA**: CUDA Toolkit 11.0+ (NVIDIA GPUs)
- **Metal**: Xcode Command Line Tools (Apple Silicon/Intel Macs)  
- **OpenMP**: For CPU parallel execution (`brew install libomp` on macOS)

### Installation

```bash
# Clone the repository
git clone https://github.com/KrxGu/RoofLine_Plot.git
cd RoofLine_Plot

# Setup Python environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install pyyaml pandas numpy matplotlib

# Test the setup
python test_build.py
```

### Basic Usage

```bash
# Run benchmarks (auto-detects available devices)
python run.py --device auto --kernels saxpy triad --size 1M 4M 16M

# Process results and generate plots  
python collect.py results/*.json
python plot_roofline.py results/*.csv

# View plots in plots/ directory
open plots/roofline_*.png
```

### Example Output

```
CPU Backend: Apple M3
  saxpy    4M  :    6.2 GFLOP/s,   37.4 GB/s
  triad    4M  :    5.4 GFLOP/s,   43.6 GB/s

CUDA Backend: NVIDIA GPU (Emulated)  
  saxpy    4M  :  500.0 GFLOP/s,  400.0 GB/s
  triad    4M  :  500.0 GFLOP/s,  400.0 GB/s
```

## 📊 Understanding Roofline Plots

A **roofline plot** reveals your hardware's performance characteristics:

- **X-axis**: Operational Intensity (FLOPs/Byte) - How many operations per byte of data
- **Y-axis**: Performance (GFLOP/s) - Achieved compute throughput  
- **Diagonal line**: Memory bandwidth limit - Performance limited by data movement
- **Horizontal line**: Compute limit - Performance limited by arithmetic units

**Reading the plot:**
- **Left side** (low OI): Memory-bound kernels - Need faster memory
- **Right side** (high OI): Compute-bound kernels - Need faster processors  
- **Distance from roofline**: Optimization opportunity

## 🔧 Supported Kernels

| Kernel | Operation | Intensity | Characteristics |
|--------|-----------|-----------|-----------------|
| **SAXPY** | `Y = αX + Y` | ~0.17 | Memory-bound, 2 FLOPs/12 bytes |
| **Triad** | `A = B + α·C` | ~0.13 | Memory-bound, 2 FLOPs/16 bytes |  
| **SGEMM** | `C = αAB + βC` | ~64 | Compute-bound, O(N³) FLOPs |

## 🖥️ Device Support

| Backend | Status | Requirements | Notes |
|---------|--------|--------------|-------|
| **CPU** | ✅ Real execution | Any x86/ARM CPU | Uses NumPy + OpenMP |
| **CUDA** | ⚡ Real + Emulated | NVIDIA GPU + Toolkit | Real profiling with nvcc |
| **Metal** | ⚡ Real + Emulated | Apple Silicon/Intel | Real profiling with Instruments |

*Emulated mode provides realistic performance estimates when hardware isn't available*

## 🛠️ Advanced Usage

### Device-Specific Benchmarks

```bash
# CPU only (real execution with OpenMP)
python run.py --device cpu --kernels saxpy triad sgemm --size 1M 4M 16M 64M

# CUDA only (requires NVIDIA GPU + toolkit)  
python run.py --device cuda --kernels saxpy triad --size 16M 64M

# Metal only (requires macOS + Xcode)
python run.py --device metal --kernels saxpy triad --size 16M 64M

# Compare all available devices
python run.py --device auto --kernels saxpy triad --size 4M
```

### Custom Configuration

Edit `bench.yaml` to modify kernel parameters:

```yaml
kernels:
  saxpy:
    alpha: 2.0
    operational_intensity: 0.167
  
sizes: ["1K", "4K", "16K", "64K", "256K", "1M", "4M", "16M", "64M"]
```

### Data Analysis

```bash
# Generate summary statistics
python collect.py --summary results/*.json

# Create device-specific plots  
python plot_roofline.py results/cpu-*.csv --output plots/cpu_only

# Export data for further analysis
python collect.py results/*.json  # Creates CSV in results/
```

## 📁 Project Structure

```
RoofLine_Plot/
├── run.py              # Main benchmark orchestrator
├── collect.py           # Data processing and CSV generation  
├── plot_roofline.py     # Visualization and plotting
├── bench.yaml           # Kernel configuration
├── backends/            # Device-specific implementations
│   ├── cpu/            # CPU backend (OpenMP)
│   ├── cuda/           # CUDA backend  
│   └── metal/          # Metal backend
├── src/kernels/        # Kernel implementations
├── results/            # Benchmark output (JSON)
├── plots/              # Generated visualizations (PNG/SVG)
└── docs/               # Documentation
```

## 🎯 Use Cases

- **Performance Analysis**: Identify memory vs compute bottlenecks
- **Hardware Comparison**: Compare GPU models and architectures  
- **Optimization Guidance**: Focus effort on dominant bottlenecks
- **Algorithm Selection**: Choose kernels based on hardware characteristics
- **Educational**: Learn parallel computing performance principles

## 🤝 Contributing

We welcome contributions! See [docs/overview.md](docs/overview.md) for architecture details and [docs/faq.md](docs/faq.md) for common questions.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made with ❤️ for the HPC and GPU computing community**