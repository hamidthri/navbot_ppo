#!/usr/bin/env python3
"""
Read TensorBoard logs for LiDAR SAC and visualize them locally
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("ERROR: tensorboard package not found!")
    print("Install with: pip3 install tensorboard")
    sys.exit(1)


def read_tensorboard_logs(log_dir):
    """
    Read all scalar data from TensorBoard logs
    
    Returns:
        dict: {metric_name: [(step, value), ...]}
    """
    print(f"[INFO] Reading TensorBoard logs from: {log_dir}")
    
    # Find all event files
    event_files = list(Path(log_dir).glob("events.out.tfevents.*"))
    
    if not event_files:
        print(f"[ERROR] No TensorBoard event files found in {log_dir}")
        return {}
    
    print(f"[INFO] Found {len(event_files)} event files")
    
    # Aggregate data from all event files
    all_data = defaultdict(list)
    
    for event_file in sorted(event_files):
        print(f"[INFO] Reading: {event_file.name}")
        
        try:
            ea = event_accumulator.EventAccumulator(str(event_file))
            ea.Reload()
            
            # Get all scalar tags
            tags = ea.Tags()['scalars']
            print(f"  Found {len(tags)} metrics: {tags}")
            
            # Read each metric
            for tag in tags:
                events = ea.Scalars(tag)
                for event in events:
                    all_data[tag].append((event.step, event.value))
                    
        except Exception as e:
            print(f"  [WARN] Error reading {event_file.name}: {e}")
            continue
    
    # Sort by step
    for tag in all_data:
        all_data[tag] = sorted(all_data[tag], key=lambda x: x[0])
        print(f"[INFO] Metric '{tag}': {len(all_data[tag])} data points")
    
    return dict(all_data)


def plot_metrics(data, save_dir):
    """
    Plot all metrics and save to files
    """
    if not data:
        print("[ERROR] No data to plot!")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[INFO] Creating plots in: {save_dir}")
    
    # Create a plot for each metric
    for metric_name, points in data.items():
        if not points:
            continue
            
        steps, values = zip(*points)
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, values, linewidth=2, alpha=0.7, color='#2E86AB')
        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'LiDAR SAC: {metric_name} over Training', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = metric_name.replace('/', '_').replace(' ', '_')
        save_path = os.path.join(save_dir, f'{filename}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {filename}.png")
        
        # Print statistics
        print(f"    Stats: min={min(values):.2f}, max={max(values):.2f}, "
              f"mean={np.mean(values):.2f}, final={values[-1]:.2f}")
    
    # Create a combined plot if we have multiple metrics
    if len(data) > 1:
        n_metrics = len(data)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes
        
        for idx, (metric_name, points) in enumerate(data.items()):
            if idx >= len(axes):
                break
            
            steps, values = zip(*points)
            ax = axes[idx]
            ax.plot(steps, values, linewidth=2, alpha=0.7, color='#2E86AB')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(metric_name, fontsize=10)
            ax.set_title(metric_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(data), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        combined_path = os.path.join(save_dir, 'all_metrics_combined.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n  ✓ Saved: all_metrics_combined.png")


def main():
    # Get paths - use absolute path
    script_dir = Path(__file__).parent.resolve()
    sac_lidar_dir = script_dir.parent.resolve()
    log_dir = sac_lidar_dir / "models" / "sac_lidar" / "logs"
    viz_dir = script_dir
    
    # Verify log directory exists
    if not log_dir.exists():
        print(f"[ERROR] Log directory does not exist: {log_dir}")
        print(f"Script dir: {script_dir}")
        print(f"SAC lidar dir: {sac_lidar_dir}")
        return 1
    
    print("="*70)
    print("LiDAR SAC TensorBoard Visualization Script")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {viz_dir}")
    print("="*70 + "\n")
    
    # Read data
    data = read_tensorboard_logs(str(log_dir))
    
    if not data:
        print("\n[ERROR] No data found in TensorBoard logs!")
        print("Possible reasons:")
        print("  1. Training hasn't logged any data yet")
        print("  2. TensorBoard logs are corrupted")
        print("  3. Wrong log directory")
        return 1
    
    print(f"\n[SUCCESS] Found {len(data)} metrics with data")
    
    # Plot data
    plot_metrics(data, str(viz_dir))
    
    print("\n" + "="*70)
    print("✓ Visualization complete!")
    print(f"✓ Plots saved to: {viz_dir}")
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
