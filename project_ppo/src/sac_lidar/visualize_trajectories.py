#!/usr/bin/env python3
"""
Visualize navigation trajectories on environment map
Creates publication-quality figures
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import glob
import os


# Environment regions
REGIONS = [
    {"id": 1, "name": "R1", "x": [2.78, 8.40], "y": [-4.40, -0.27]},
    {"id": 2, "name": "R2", "x": [-2.00, 2.78], "y": [-4.40, -3.46]},
    {"id": 3, "name": "R3", "x": [-2.02, 2.00], "y": [-0.27, 3.44]},
    {"id": 4, "name": "R4", "x": [-1.90, 4.29], "y": [4.71, 5.18]},
    {"id": 5, "name": "R5", "x": [-8.21, -3.03], "y": [-1.05, 1.00]},
    {"id": 6, "name": "R6", "x": [-8.85, -3.18], "y": [-3.60, -1.84]},
    {"id": 7, "name": "R7", "x": [-4.96, -3.90], "y": [-0.78, 2.40]},
    {"id": 8, "name": "R8", "x": [-8.13, -7.43], "y": [1.16, 2.34]},
    {"id": 9, "name": "R9", "x": [5.22, 8.79], "y": [2.25, 2.85]},
    {"id": 10, "name": "R10", "x": [7.65, 8.79], "y": [0.00, 2.25]},
    {"id": 11, "name": "R11", "x": [0.28, 0.57], "y": [-1.79, -0.74]},
    {"id": 12, "name": "R12", "x": [-6.42, -3.53], "y": [-5.02, -3.88]},
    {"id": 13, "name": "R13", "x": [2.81, 3.49], "y": [1.78, 2.88]},
]


def plot_environment():
    """Plot environment regions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot regions
    for region in REGIONS:
        width = region["x"][1] - region["x"][0]
        height = region["y"][1] - region["y"][0]
        rect = patches.Rectangle(
            (region["x"][0], region["y"][0]), width, height,
            linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.3
        )
        ax.add_patch(rect)
        
        # Add region label
        center_x = (region["x"][0] + region["x"][1]) / 2
        center_y = (region["y"][0] + region["y"][1]) / 2
        ax.text(center_x, center_y, region["name"], 
               ha='center', va='center', fontsize=8, alpha=0.5)
    
    ax.set_xlim(-10, 10)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    
    return fig, ax


def visualize_all_trajectories(results_dir):
    """Create visualization of all trajectories"""
    traj_dir = os.path.join(results_dir, 'trajectories')
    traj_files = glob.glob(os.path.join(traj_dir, '*.txt'))
    
    if not traj_files:
        print("No trajectory files found!")
        return
    
    print(f"Visualizing {len(traj_files)} trajectories...")
    
    # Combined view
    fig, ax = plot_environment()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(traj_files)))
    
    for i, (traj_file, color) in enumerate(zip(traj_files, colors)):
        data = np.loadtxt(traj_file, delimiter=',', skiprows=1)
        x, y = data[:, 1], data[:, 2]
        
        # Plot trajectory
        ax.plot(x, y, color=color, linewidth=2, alpha=0.7, 
               label=os.path.basename(traj_file).replace('.txt', '').replace('_', ' '))
        
        # Mark start and end
        ax.plot(x[0], y[0], 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(x[-1], y[-1], '*', color=color, markersize=12, markeredgecolor='black', markeredgewidth=1.5)
    
    ax.set_title('Navigation Trajectories - All Scenarios', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, 'all_trajectories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    
    # Individual trajectory plots
    for traj_file in traj_files:
        fig, ax = plot_environment()
        
        data = np.loadtxt(traj_file, delimiter=',', skiprows=1)
        time, x, y, distance = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
        
        # Plot trajectory with color gradient
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='viridis', linewidth=3)
        lc.set_array(time)
        line = ax.add_collection(lc)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'go', markersize=12, markeredgecolor='black', 
               markeredgewidth=2, label='Start', zorder=10)
        ax.plot(x[-1], y[-1], 'r*', markersize=15, markeredgecolor='black', 
               markeredgewidth=2, label='End', zorder=10)
        
        scenario_name = os.path.basename(traj_file).replace('.txt', '').replace('_', ' ')
        ax.set_title(f'Trajectory: {scenario_name}', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(line, ax=ax, label='Time (s)')
        ax.legend()
        
        plt.tight_layout()
        output_path = os.path.join(results_dir, f'trajectory_{os.path.basename(traj_file).replace(".txt", ".png")}')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✓ Saved {len(traj_files)} individual trajectory plots")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_trajectories.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    visualize_all_trajectories(results_dir)