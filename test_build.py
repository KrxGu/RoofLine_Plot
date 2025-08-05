#!/usr/bin/env python3
"""
Test script to verify the GPU Roofline benchmark toolkit build and basic functionality.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and report success/failure."""
    print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ SUCCESS: {description}")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"✗ COMMAND NOT FOUND: {cmd[0]}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ Found: {description} ({filepath})")
        return True
    else:
        print(f"✗ Missing: {description} ({filepath})")
        return False

def main():
    print("GPU Roofline Benchmark - Build Test")
    print("=" * 50)
    
    # Check prerequisites
    print("\n=== Checking Prerequisites ===")
    has_cmake = run_command(['cmake', '--version'], "CMake availability")
    has_python = run_command(['python3', '--version'], "Python 3 availability")
    has_git = run_command(['git', '--version'], "Git availability")
    
    # Check optional dependencies
    print("\n=== Checking Optional Dependencies ===")
    has_nvcc = run_command(['nvcc', '--version'], "CUDA toolkit (nvcc)")
    has_metal = False
    if sys.platform == 'darwin':
        has_metal = run_command(['xcrun', '-find', 'metal'], "Metal compiler")
    
    # Check project structure
    print("\n=== Checking Project Structure ===")
    structure_checks = [
        ("CMakeLists.txt", "Main CMake file"),
        ("bench.yaml", "Benchmark configuration"),
        ("run.py", "Main orchestration script"),
        ("collect.py", "Data collection script"),
        ("plot_roofline.py", "Plotting script"),
        ("src/kernels/saxpy.cu", "SAXPY CUDA kernel"),
        ("src/kernels/saxpy.metal", "SAXPY Metal kernel"),
        ("backends/cuda/cuda_runner.cpp", "CUDA backend"),
        ("backends/metal/metal_runner.mm", "Metal backend"),
        ("backends/cpu/cpu_runner.cpp", "CPU backend"),
        ("include/kernel_launcher.hpp", "Kernel launcher interface"),
    ]
    
    structure_ok = all(check_file_exists(filepath, desc) for filepath, desc in structure_checks)
    
    # Test Python dependencies
    print("\n=== Testing Python Dependencies ===")
    python_deps = ['yaml', 'pandas', 'numpy', 'matplotlib']
    deps_ok = True
    
    for dep in python_deps:
        try:
            result = subprocess.run([sys.executable, '-c', f'import {dep}'], 
                                  capture_output=True, check=True)
            print(f"✓ Python module: {dep}")
        except subprocess.CalledProcessError:
            print(f"✗ Missing Python module: {dep}")
            deps_ok = False
    
    if not deps_ok:
        print("\nTo install missing dependencies:")
        print(f"  pip install {' '.join(python_deps)}")
    
    # Test basic CMake configuration
    print("\n=== Testing CMake Configuration ===")
    cmake_ok = True
    if has_cmake:
        # Test CPU-only build
        os.makedirs("build/test", exist_ok=True)
        cmake_ok = run_command([
            'cmake', '-B', 'build/test', '-S', '.', 
            '-DENABLE_CPU=ON', '-DENABLE_CUDA=OFF', '-DENABLE_METAL=OFF'
        ], "CMake configuration (CPU only)")
    
    # Test run.py basic functionality
    print("\n=== Testing Run Script ===")
    run_test = run_command([sys.executable, 'run.py', '--help'], 
                          "Run script help")
    
    # Summary
    print("\n" + "=" * 50)
    print("BUILD TEST SUMMARY")
    print("=" * 50)
    
    all_good = True
    
    if has_cmake and has_python and has_git:
        print("✓ Core prerequisites available")
    else:
        print("✗ Missing core prerequisites")
        all_good = False
    
    if structure_ok:
        print("✓ Project structure complete")
    else:
        print("✗ Project structure incomplete")
        all_good = False
    
    if deps_ok:
        print("✓ Python dependencies available")
    else:
        print("✗ Missing Python dependencies")
        all_good = False
    
    if cmake_ok:
        print("✓ CMake configuration works")
    else:
        print("✗ CMake configuration failed")
        all_good = False
    
    print(f"\nAvailable backends:")
    if has_nvcc:
        print("  ✓ CUDA (GPU)")
    else:
        print("  ✗ CUDA (install CUDA toolkit)")
    
    if has_metal:
        print("  ✓ Metal (GPU)")
    elif sys.platform == 'darwin':
        print("  ✗ Metal (install Xcode)")
    else:
        print("  - Metal (macOS only)")
    
    print("  ✓ CPU (OpenMP)")
    
    if all_good:
        print(f"\n🎉 PROJECT READY! Next steps:")
        print(f"   1. python run.py --device auto --kernels saxpy")
        print(f"   2. python collect.py results/*.json") 
        print(f"   3. python plot_roofline.py results/*.csv")
    else:
        print(f"\n❌ PROJECT NEEDS ATTENTION")
        print(f"   Fix the issues above before proceeding")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())