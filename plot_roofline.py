#!/usr/bin/env python3
"""
GPU Roofline Plotter

Generates roofline plots from CSV benchmark data.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json

class RooflinePlotter:
    def __init__(self):
        self.fig = None
        self.ax = None
        self.device_colors = {
            'cuda': '#00D2FF',    # NVIDIA green-blue
            'metal': '#FF6B35',   # Apple orange
            'cpu': '#4ECDC4'      # Teal
        }
        self.kernel_markers = {
            'saxpy': 'o',
            'triad': 's', 
            'sgemm': '^',
            'wmma_gemm': 'D'
        }
        
    def load_csv_data(self, csv_files: List[str]) -> pd.DataFrame:
        """Load and combine CSV files."""
        dataframes = []
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                dataframes.append(df)
                print(f"✓ Loaded {csv_file}: {len(df)} rows")
            except Exception as e:
                print(f"✗ Error loading {csv_file}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No valid CSV files found")
        
        combined_df = pd.concat(dataframes, ignore_index=True)
        print(f"Combined dataset: {len(combined_df)} total rows")
        
        return combined_df
    
    def get_device_capabilities(self, device_type: str, device_name: str, precision: str) -> Dict[str, float]:
        """Get device specs for roofline bounds."""
        # TODO: load from device_caps.json
        
        capabilities = {
            'cuda': {
                'A100': {
                    'peak_bandwidth_gb_s': 1935,
                    'peak_compute_gflops': {'float32': 19500, 'float16': 78000, 'float64': 9700}
                },
                'V100': {
                    'peak_bandwidth_gb_s': 900,
                    'peak_compute_gflops': {'float32': 15700, 'float16': 31400, 'float64': 7800}
                },
                'default': {
                    'peak_bandwidth_gb_s': 500,
                    'peak_compute_gflops': {'float32': 10000, 'float16': 20000, 'float64': 5000}
                }
            },
            'metal': {
                'Apple M3': {
                    'peak_bandwidth_gb_s': 200,
                    'peak_compute_gflops': {'float32': 4000, 'float16': 8000, 'float64': 2000}
                },
                'Apple M2': {
                    'peak_bandwidth_gb_s': 100,
                    'peak_compute_gflops': {'float32': 3000, 'float16': 6000, 'float64': 1500}
                },
                'default': {
                    'peak_bandwidth_gb_s': 100,
                    'peak_compute_gflops': {'float32': 3000, 'float16': 6000, 'float64': 1500}
                }
            },
            'cpu': {
                'default': {
                    'peak_bandwidth_gb_s': 50,
                    'peak_compute_gflops': {'float32': 500, 'float16': 1000, 'float64': 250}
                }
            }
        }
        
        device_caps = capabilities.get(device_type, capabilities['cpu'])
        
        # Match device name or use default
        for key in device_caps:
            if key.lower() in device_name.lower():
                caps = device_caps[key]
                break
        else:
            caps = device_caps['default']
        
        return {
            'peak_bandwidth_gb_s': caps['peak_bandwidth_gb_s'],
            'peak_compute_gflops': caps['peak_compute_gflops'].get(precision, caps['peak_compute_gflops']['float32'])
        }
    
    def create_roofline_plot(self, df: pd.DataFrame, output_file: str = None, interactive: bool = False) -> None:
        """Create the main roofline plot."""
        # Setup plot
        plt.style.use('default')
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Log-log plot
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        
        # Draw rooflines for each device
        devices = df['device_type'].unique()
        
        for device in devices:
            device_df = df[df['device_type'] == device]
            if len(device_df) == 0:
                continue
                
            # Get device info for bounds
            sample_row = device_df.iloc[0]
            device_name = sample_row.get('device_name', f'{device} device')
            precision = sample_row.get('precision', 'float32')
            
            caps = self.get_device_capabilities(device, device_name, precision)
            
            # Draw the bounds
            self.draw_roofline_bounds(
                caps['peak_bandwidth_gb_s'], 
                caps['peak_compute_gflops'],
                device, device_name
            )
        
        # Plot data points
        self.plot_benchmark_points(df)
        
        # Format plot
        self.configure_plot_appearance()
        
        # Add legend
        self.add_legend(df)
        
        # Save it
        if output_file:
            self.save_plot(output_file)
        
        if interactive:
            plt.show()
    
    def draw_roofline_bounds(self, peak_bandwidth: float, peak_compute: float, device_type: str, device_name: str) -> None:
        """Draw roofline bounds for a device."""
        color = self.device_colors.get(device_type, '#808080')
        
        # OI range for plotting
        oi_min, oi_max = 0.01, 1000
        oi_range = np.logspace(np.log10(oi_min), np.log10(oi_max), 1000)
        
        # Memory bound (diagonal)
        memory_bound = peak_bandwidth * oi_range
        
        # Compute bound (horizontal)
        compute_bound = np.full_like(oi_range, peak_compute)
        
        # Roofline is min of both
        roofline = np.minimum(memory_bound, compute_bound)
        
        # Plot it
        self.ax.plot(oi_range, roofline, '--', color=color, linewidth=2, alpha=0.7,
                    label=f'{device_name} Roofline')
        
        # Text annotations
        mem_oi = peak_compute / peak_bandwidth * 0.5  # spot for text
        if oi_min <= mem_oi <= oi_max:
            self.ax.annotate(f'{peak_bandwidth:.0f} GB/s', 
                           xy=(mem_oi, peak_bandwidth * mem_oi),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, color=color, alpha=0.8)
        
        # Compute annotation
        comp_oi = oi_max * 0.3  
        self.ax.annotate(f'{peak_compute:.0f} GFLOP/s', 
                       xy=(comp_oi, peak_compute),
                       xytext=(10, -15), textcoords='offset points',
                       fontsize=8, color=color, alpha=0.8)
    
    def plot_benchmark_points(self, df: pd.DataFrame) -> None:
        """Plot benchmark points."""
        for _, row in df.iterrows():
            device_type = row['device_type']
            kernel_name = row['kernel_name']
            
            x = row['operational_intensity']
            y = row['gflops_achieved']
            
            color = self.device_colors.get(device_type, '#808080')
            marker = self.kernel_markers.get(kernel_name, 'o')
            
            # Plot point
            self.ax.scatter(x, y, 
                          c=color, marker=marker, s=100, 
                          alpha=0.8, edgecolors='black', linewidth=0.5,
                          label=f'{device_type}_{kernel_name}')
            
            # Label it
            label = f'{kernel_name}\n{row["problem_size"]//1000000}M'
            self.ax.annotate(label, (x, y), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.7)
    
    def configure_plot_appearance(self) -> None:
        """Style the plot."""
        self.ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=12)
        self.ax.set_ylabel('Performance (GFLOP/s)', fontsize=12)
        self.ax.set_title('GPU Roofline Performance Analysis', fontsize=14, fontweight='bold')
        
        # Axis limits
        self.ax.set_xlim(0.01, 1000)
        self.ax.set_ylim(0.1, 100000)
        
        # Grid
        self.ax.grid(True, alpha=0.3, which='both')
        
        # Region labels
        self.ax.text(0.05, 10000, 'Memory\nBound\nRegion', 
                    fontsize=10, alpha=0.6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.2))
        
        self.ax.text(50, 50000, 'Compute\nBound\nRegion', 
                    fontsize=10, alpha=0.6, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
    
    def add_legend(self, df: pd.DataFrame) -> None:
        """Add legend."""
        legend_elements = []
        
        # Device types
        for device in df['device_type'].unique():
            color = self.device_colors.get(device, '#808080')
            legend_elements.append(
                plt.Line2D([0], [0], color=color, linestyle='--', linewidth=2,
                          label=f'{device.upper()} Roofline')
            )
        
        # Kernel types
        for kernel in df['kernel_name'].unique():
            marker = self.kernel_markers.get(kernel, 'o')
            legend_elements.append(
                plt.Line2D([0], [0], marker=marker, color='gray', linestyle='None',
                          markersize=8, label=f'{kernel}')
            )
        
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    def save_plot(self, output_file: str) -> None:
        """Save plot."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # PNG for embedding
        png_file = output_path.with_suffix('.png')
        self.fig.savefig(png_file, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {png_file}")
        
        # SVG for scalability
        svg_file = output_path.with_suffix('.svg')
        self.fig.savefig(svg_file, bbox_inches='tight')
        print(f"✓ Vector plot saved to {svg_file}")

def main():
    parser = argparse.ArgumentParser(description="GPU Roofline Plotter")
    parser.add_argument('csv_files', nargs='+', help='CSV data files to plot')
    parser.add_argument('--output', '-o', default='plots/roofline', help='Output file prefix')
    parser.add_argument('--show', action='store_true', help='Show interactive plot')
    parser.add_argument('--title', help='Custom plot title')
    
    args = parser.parse_args()
    
    # Initialize plotter
    plotter = RooflinePlotter()
    
    # Load data
    df = plotter.load_csv_data(args.csv_files)
    
    if len(df) == 0:
        print("No data to plot. Exiting.")
        return
    
    # Create output filename
    devices = '-'.join(sorted(df['device_type'].unique()))
    output_file = f"{args.output}_{devices}"
    
    # Create plot
    plotter.create_roofline_plot(df, output_file, args.show)
    
    print("\n=== Roofline Plot Complete ===")
    print(f"Analyzed {len(df)} benchmark results")
    print(f"Devices: {', '.join(df['device_type'].unique())}")
    print(f"Kernels: {', '.join(df['kernel_name'].unique())}")

if __name__ == "__main__":
    main()