#!/usr/bin/env python3
"""
GPU Roofline Data Collector

Processes JSON benchmark results into CSV for plotting.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import sys

class DataCollector:
    def __init__(self):
        self.results = []
        
    def load_json_results(self, json_files: List[str]) -> None:
        """Load and validate JSON files."""
        print(f"Loading {len(json_files)} result files...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    result = json.load(f)
                    
                # Check for required fields
                required_fields = [
                    'kernel_name', 'device_type', 'precision', 'problem_size',
                    'execution_time_ms', 'gflops_achieved', 'gbps_achieved',
                    'operational_intensity', 'flops_executed', 'bytes_transferred'
                ]
                
                if all(field in result for field in required_fields):
                    self.results.append(result)
                    print(f"✓ Loaded {json_file}")
                else:
                    missing = [f for f in required_fields if f not in result]
                    print(f"✗ Skipping {json_file}: missing fields {missing}")
                    
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"✗ Error loading {json_file}: {e}")
                continue
    
    def calculate_derived_metrics(self) -> None:
        """Add extra metrics to results."""
        for result in self.results:
            # Double-check operational intensity calc
            if result['bytes_transferred'] > 0:
                calculated_oi = result['flops_executed'] / result['bytes_transferred']
                result['operational_intensity_calculated'] = calculated_oi
            else:
                result['operational_intensity_calculated'] = 0.0
            
            # Time per element
            result['time_per_element_ns'] = (result['execution_time_ms'] * 1e6) / result['problem_size']
            
            # Efficiency estimates
            peak_bandwidth_gb_s = self._get_peak_bandwidth(result['device_type'], result.get('device_name', ''))
            if peak_bandwidth_gb_s > 0:
                result['memory_bandwidth_efficiency'] = (result['gbps_achieved'] / peak_bandwidth_gb_s) * 100
            else:
                result['memory_bandwidth_efficiency'] = 0.0
            
            peak_compute_gflops = self._get_peak_compute(result['device_type'], result.get('device_name', ''), result['precision'])
            if peak_compute_gflops > 0:
                result['compute_efficiency'] = (result['gflops_achieved'] / peak_compute_gflops) * 100
            else:
                result['compute_efficiency'] = 0.0
    
    def _get_peak_bandwidth(self, device_type: str, device_name: str) -> float:
        """Get rough peak bandwidth estimates."""
        # TODO: load from device_caps.json instead
        bandwidth_map = {
            'cuda': {
                'A100': 1935,      # HBM2e
                'V100': 900,       # HBM2
                'RTX 4090': 1008,  # GDDR6X
                'default': 500     # Conservative estimate
            },
            'metal': {
                'Apple M3': 200,   # LPDDR5
                'Apple M2': 100,   # LPDDR5
                'Apple M1': 68,    # LPDDR4X
                'default': 100     # Conservative estimate
            },
            'cpu': {
                'default': 50      # DDR4/DDR5 estimate
            }
        }
        
        device_map = bandwidth_map.get(device_type, {'default': 100})
        
        # Match device name or use default
        for key in device_map:
            if key.lower() in device_name.lower():
                return device_map[key]
        
        return device_map['default']
    
    def _get_peak_compute(self, device_type: str, device_name: str, precision: str) -> float:
        """Get rough peak compute estimates."""
        compute_map = {
            'cuda': {
                'A100': {'float32': 19500, 'float16': 78000, 'float64': 9700},
                'V100': {'float32': 15700, 'float16': 31400, 'float64': 7800},
                'RTX 4090': {'float32': 35000, 'float16': 70000, 'float64': 1100},
                'default': {'float32': 10000, 'float16': 20000, 'float64': 5000}
            },
            'metal': {
                'Apple M3': {'float32': 4000, 'float16': 8000, 'float64': 2000},
                'Apple M2': {'float32': 3000, 'float16': 6000, 'float64': 1500},
                'Apple M1': {'float32': 2600, 'float16': 5200, 'float64': 1300},
                'default': {'float32': 3000, 'float16': 6000, 'float64': 1500}
            },
            'cpu': {
                'default': {'float32': 500, 'float16': 1000, 'float64': 250}
            }
        }
        
        device_map = compute_map.get(device_type, {'default': {'float32': 1000, 'float16': 2000, 'float64': 500}})
        
        # Match device or use default
        for key in device_map:
            if key.lower() in device_name.lower():
                precision_map = device_map[key]
                return precision_map.get(precision, precision_map.get('float32', 1000))
        
        precision_map = device_map['default']
        return precision_map.get(precision, precision_map.get('float32', 1000))
    
    def create_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        if not self.results:
            raise ValueError("No valid results to process")
        
        df = pd.DataFrame(self.results)
        
        # Add computed columns
        df['device_kernel'] = df['device_type'] + '_' + df['kernel_name']
        df['size_mb'] = df['problem_size'] * 4 / (1024 * 1024)  # assuming float32
        df['gflops_per_second'] = df['gflops_achieved']
        df['gbps_bandwidth'] = df['gbps_achieved']
        
        # Sort by device, kernel, size
        df = df.sort_values(['device_type', 'kernel_name', 'problem_size'])
        
        return df
    
    def save_csv(self, df: pd.DataFrame, output_file: str) -> None:
        """Save to CSV."""
        # Pick columns for roofline plotting
        columns = [
            'device_type', 'device_name', 'kernel_name', 'precision',
            'problem_size', 'size_mb',
            'operational_intensity', 'operational_intensity_calculated',
            'gflops_achieved', 'gbps_achieved',
            'execution_time_ms', 'time_per_element_ns',
            'memory_bandwidth_efficiency', 'compute_efficiency',
            'gpu_utilization_percent', 'memory_utilization_percent',
            'flops_executed', 'bytes_transferred', 'timestamp'
        ]
        
        # Only use columns that actually exist
        available_columns = [col for col in columns if col in df.columns]
        
        df_output = df[available_columns].copy()
        df_output.to_csv(output_file, index=False)
        print(f"✓ CSV saved to {output_file}")
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """Print summary stats."""
        print("\n=== Data Collection Summary ===")
        print(f"Total benchmark runs: {len(df)}")
        print(f"Devices tested: {df['device_type'].nunique()}")
        print(f"Kernels tested: {df['kernel_name'].nunique()}")
        print(f"Precisions tested: {df['precision'].nunique()}")
        
        print("\nPerformance ranges:")
        print(f"  GFLOP/s:     {df['gflops_achieved'].min():8.1f} - {df['gflops_achieved'].max():8.1f}")
        print(f"  GB/s:        {df['gbps_achieved'].min():8.1f} - {df['gbps_achieved'].max():8.1f}")
        print(f"  Op Intensity:{df['operational_intensity'].min():8.3f} - {df['operational_intensity'].max():8.3f}")
        
        print("\nBy device:")
        for device in df['device_type'].unique():
            device_df = df[df['device_type'] == device]
            avg_gflops = device_df['gflops_achieved'].mean()
            avg_gbps = device_df['gbps_achieved'].mean()
            print(f"  {device:8s}: {avg_gflops:8.1f} GFLOP/s, {avg_gbps:8.1f} GB/s (avg)")

def main():
    parser = argparse.ArgumentParser(description="GPU Roofline Data Collector")
    parser.add_argument('json_files', nargs='+', help='JSON result files to process')
    parser.add_argument('--output', '-o', help='Output CSV file')
    parser.add_argument('--summary', action='store_true', help='Print detailed summary')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = DataCollector()
    
    # Load and process results
    collector.load_json_results(args.json_files)
    
    if not collector.results:
        print("No valid results found. Exiting.")
        sys.exit(1)
    
    # Calculate derived metrics
    collector.calculate_derived_metrics()
    
    # Create DataFrame
    df = collector.create_dataframe()
    
    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        # Auto-generate filename based on devices and timestamp
        devices = '-'.join(sorted(df['device_type'].unique()))
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d")
        output_file = f"results/{timestamp}-{devices}.csv"
    
    # Save CSV
    collector.save_csv(df, output_file)
    
    # Print summary
    collector.print_summary(df)
    
    if args.summary:
        print("\n=== Detailed Results ===")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df[['device_type', 'kernel_name', 'problem_size', 'gflops_achieved', 'gbps_achieved', 'operational_intensity']].to_string())

if __name__ == "__main__":
    main()