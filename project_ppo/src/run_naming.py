#!/usr/bin/env python3
"""
Standardized run naming convention for PPO training experiments.

Format: YYYYMMDD_HHMM_<world>_<vision>_<sampler>_<reach>_<dist>_<reward>_T<timesteps>

Example: 20260124_2310_small_house_dinov2_vits14_mapSampler_reachBFS_distUniform_rewardV1_T200k
"""

from datetime import datetime
from pathlib import Path
import yaml
import os


def generate_run_name(
    world="small_house",
    vision_backbone="resnet18",
    use_map_sampler=True,
    use_reachability=True,
    distance_uniform=True,
    reward_version="V1",
    max_timesteps=200000
):
    """
    Generate standardized run name based on configuration.
    
    Args:
        world: Gazebo world name (e.g., 'small_house', 'office', 'maze')
        vision_backbone: Vision backbone name (e.g., 'resnet18', 'dinov2_vits14', 'mobilenet')
        use_map_sampler: Whether map-based sampling is enabled
        use_reachability: Whether reachability checking is enabled
        distance_uniform: Whether distance sampling is uniform (vs spatial_uniform)
        reward_version: Reward function version tag
        max_timesteps: Total training timesteps
        
    Returns:
        Standardized run name string
    """
    # Date-time prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # World (clean up any path separators)
    world_clean = world.replace("/", "_").replace(".", "_")
    
    # Vision backbone (clean up)
    vision_clean = vision_backbone.replace("_", "")  # dinov2_vits14 -> dinov2vits14
    
    # Sampler mode
    sampler = "mapSampler" if use_map_sampler else "randomSampler"
    
    # Reachability
    reach = "reachBFS" if use_reachability else "reachNone"
    
    # Distance sampling
    dist = "distUniform" if distance_uniform else "spatialUniform"
    
    # Reward version
    reward_tag = f"reward{reward_version}"
    
    # Timesteps (format as 200k, 500k, 1M, etc.)
    if max_timesteps >= 1_000_000:
        timesteps_str = f"T{max_timesteps // 1_000_000}M"
    elif max_timesteps >= 1_000:
        timesteps_str = f"T{max_timesteps // 1_000}k"
    else:
        timesteps_str = f"T{max_timesteps}"
    
    # Assemble full name
    run_name = f"{timestamp}_{world_clean}_{vision_clean}_{sampler}_{reach}_{dist}_{reward_tag}_{timesteps_str}"
    
    return run_name


def create_run_structure(base_runs_dir, run_name, config_dict=None):
    """
    Create standardized directory structure for a training run.
    
    Args:
        base_runs_dir: Path to the 'runs/' directory
        run_name: Generated run name
        config_dict: Optional config dictionary to save as config.yml
        
    Returns:
        Dictionary with paths to all subdirectories
    """
    base_runs_dir = Path(base_runs_dir)
    run_dir = base_runs_dir / run_name
    
    # Create directory structure
    dirs = {
        "run_dir": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "logs": run_dir / "logs",
        "tb": run_dir / "tb",
        "plots": run_dir / "plots",
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Save config snapshot
    if config_dict:
        config_path = run_dir / "config.yml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Create empty RUN_SUMMARY.md template
    summary_path = run_dir / "RUN_SUMMARY.md"
    if not summary_path.exists():
        with open(summary_path, 'w') as f:
            f.write(f"# Run Summary: {run_name}\n\n")
            f.write(f"**Start Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Configuration\n\n")
            f.write("(Auto-filled from config.yml)\n\n")
            f.write("## Results\n\n")
            f.write("(To be filled after training)\n\n")
            f.write("## Notes\n\n")
            f.write("(Add observations here)\n\n")
    
    return dirs


def get_comparison_table_path(base_runs_dir):
    """Get path to the global runs comparison markdown."""
    return Path(base_runs_dir) / "RUNS_COMPARISON.md"


def init_comparison_table(base_runs_dir):
    """Initialize the global runs comparison table if it doesn't exist."""
    comparison_path = get_comparison_table_path(base_runs_dir)
    
    if not comparison_path.exists():
        with open(comparison_path, 'w') as f:
            f.write("# Training Runs Comparison\n\n")
            f.write("Comprehensive comparison of all PPO training runs.\n\n")
            f.write("## Runs Table\n\n")
            f.write("| Run Name | Date | World | Vision | Sampler | Reach | Dist | Reward | Timesteps | Success % | Collision % | Mean Reward | Notes |\n")
            f.write("|----------|------|-------|--------|---------|-------|------|--------|-----------|-----------|-------------|-------------|-------|\n")
    
    return comparison_path


def add_run_to_comparison(
    base_runs_dir,
    run_name,
    world,
    vision,
    sampler,
    reach,
    dist,
    reward_ver,
    timesteps,
    success_rate=None,
    collision_rate=None,
    mean_reward=None,
    notes=""
):
    """Add a completed run to the comparison table."""
    comparison_path = init_comparison_table(base_runs_dir)
    
    # Parse date from run_name (first 13 chars: YYYYMMDD_HHMM)
    date_str = run_name[:13].replace("_", " ")
    
    # Format metrics
    success_str = f"{success_rate:.1f}%" if success_rate is not None else "TBD"
    collision_str = f"{collision_rate:.1f}%" if collision_rate is not None else "TBD"
    reward_str = f"{mean_reward:.1f}" if mean_reward is not None else "TBD"
    
    # Create table row
    row = f"| {run_name} | {date_str} | {world} | {vision} | {sampler} | {reach} | {dist} | {reward_ver} | {timesteps} | {success_str} | {collision_str} | {reward_str} | {notes} |\n"
    
    # Append to file
    with open(comparison_path, 'a') as f:
        f.write(row)
    
    print(f"✓ Added to comparison table: {comparison_path}")


if __name__ == "__main__":
    # Test
    run_name = generate_run_name(
        world="small_house",
        vision_backbone="dinov2_vits14",
        use_map_sampler=True,
        use_reachability=True,
        distance_uniform=True,
        reward_version="V1",
        max_timesteps=200000
    )
    print(f"Generated run name: {run_name}")
