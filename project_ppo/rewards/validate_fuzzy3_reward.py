#!/usr/bin/env python3
"""
Validation and Visualization for Fuzzy3 Reward System

Usage:
  python3 validate_fuzzy3_reward.py                  # Interactive plots
  python3 validate_fuzzy3_reward.py --save           # Save to runs_test/fuzzy3_viz/
  python3 validate_fuzzy3_reward.py --save --outdir my_dir/  # Custom output
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rewards'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import math

# Import fuzzy reward
from fuzzy3_reward import Fuzzy3Reward, wrap_to_pi


def test_trapezoid_shoulders():
    """Test trapezoid function handles shoulder cases correctly."""
    print("="*70)
    print("TESTING TRAPEZOID SHOULDER CASES")
    print("="*70)
    
    fuzzy = Fuzzy3Reward()
    
    tests = [
        ("Aligned at e=0 (left shoulder)", 
         lambda: fuzzy._heading_aligned(0.0), 1.0),
        ("Aligned at e=0.10", 
         lambda: fuzzy._heading_aligned(0.10), 1.0),
        ("Aligned at e=0.20", 
         lambda: fuzzy._heading_aligned(0.20), 1.0),
        ("Aligned outside at e=-0.01", 
         lambda: fuzzy._heading_aligned(-0.01), 0.0),
        
        ("Danger at c=0 (left shoulder)", 
         lambda: fuzzy._clearance_danger(0.0), 1.0),
        ("Danger at c=0.10", 
         lambda: fuzzy._clearance_danger(0.10), 1.0),
        ("Danger at c=0.20", 
         lambda: fuzzy._clearance_danger(0.20), 1.0),
        
        ("Toward at dmax (right shoulder)", 
         lambda: fuzzy._progress_toward(fuzzy.max_translation), 1.0),
        ("Toward at dmax-0.01", 
         lambda: fuzzy._progress_toward(fuzzy.max_translation - 0.01), 1.0),
        
        ("Clear at c_max (right shoulder)", 
         lambda: fuzzy._clearance_clear(fuzzy.c_max), 1.0),
        ("Clear at c_max-0.1", 
         lambda: fuzzy._clearance_clear(fuzzy.c_max - 0.1), 1.0),
        
        ("Misaligned at pi (right shoulder)", 
         lambda: fuzzy._heading_misaligned(math.pi), 1.0),
        
        ("Away at -dmax (left shoulder)", 
         lambda: fuzzy._progress_away(-fuzzy.max_translation), 1.0),
    ]
    
    all_pass = True
    for name, func, expected in tests:
        result = func()
        passed = abs(result - expected) < 0.01
        all_pass = all_pass and passed
        status = "✓" if passed else "✗"
        print(f"{status} {name:45s}: {result:.3f} (expected {expected:.3f})")
    
    print("-"*70)
    print(f"{'✓ ALL SHOULDER TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
    print("="*70 + "\n")
    
    return all_pass


def plot_membership_functions(fuzzy, save=False, outdir=None):
    """Plot all membership functions."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Progress memberships
    ax = axes[0]
    delta_d = np.linspace(-fuzzy.max_translation, fuzzy.max_translation, 200)
    away = [fuzzy._progress_away(d) for d in delta_d]
    zero = [fuzzy._progress_zero(d) for d in delta_d]
    toward = [fuzzy._progress_toward(d) for d in delta_d]
    
    ax.plot(delta_d, away, 'r-', label='Away', linewidth=2)
    ax.plot(delta_d, zero, 'y-', label='Zero', linewidth=2)
    ax.plot(delta_d, toward, 'g-', label='Toward', linewidth=2)
    ax.set_xlabel('Progress Δd (m/step)', fontsize=11)
    ax.set_ylabel('Membership', fontsize=11)
    ax.set_title('Progress Membership Functions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([-0.05, 1.1])
    
    # Plot 2: Heading error memberships
    ax = axes[1]
    e_vals = np.linspace(0, math.pi, 200)
    aligned = [fuzzy._heading_aligned(e) for e in e_vals]
    medium = [fuzzy._heading_medium(e) for e in e_vals]
    misaligned = [fuzzy._heading_misaligned(e) for e in e_vals]
    
    ax.plot(np.degrees(e_vals), aligned, 'g-', label='Aligned', linewidth=2)
    ax.plot(np.degrees(e_vals), medium, 'y-', label='Medium', linewidth=2)
    ax.plot(np.degrees(e_vals), misaligned, 'r-', label='Misaligned', linewidth=2)
    ax.set_xlabel('Heading Error e (degrees)', fontsize=11)
    ax.set_ylabel('Membership', fontsize=11)
    ax.set_title('Heading Error Membership Functions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([-0.05, 1.1])
    
    # Plot 3: Clearance memberships
    ax = axes[2]
    c_vals = np.linspace(0, fuzzy.c_max, 200)
    danger = [fuzzy._clearance_danger(c) for c in c_vals]
    caution = [fuzzy._clearance_caution(c) for c in c_vals]
    clear = [fuzzy._clearance_clear(c) for c in c_vals]
    
    ax.plot(c_vals, danger, 'r-', label='Danger', linewidth=2)
    ax.plot(c_vals, caution, 'y-', label='Caution', linewidth=2)
    ax.plot(c_vals, clear, 'g-', label='Clear', linewidth=2)
    ax.set_xlabel('Front Clearance c (m)', fontsize=11)
    ax.set_ylabel('Membership', fontsize=11)
    ax.set_title('Clearance Membership Functions', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    ax.set_ylim([-0.05, 1.1])
    
    plt.tight_layout()
    
    if save:
        os.makedirs(outdir, exist_ok=True)
        filepath = os.path.join(outdir, 'membership_functions.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def plot_reward_surfaces(fuzzy, save=False, outdir=None):
    """Plot 3D reward surfaces for different input combinations."""
    
    # Surface 1: reward(Δd, e | c fixed)
    fig = plt.figure(figsize=(18, 5))
    
    delta_d_grid = np.linspace(-fuzzy.max_translation, fuzzy.max_translation, 40)
    e_grid = np.linspace(0, math.pi, 40)
    DD, EE = np.meshgrid(delta_d_grid, e_grid)
    
    c_slices = [0.2, 0.6, 1.5]  # Danger, Caution, Clear
    c_labels = ['Danger (c=0.2m)', 'Caution (c=0.6m)', 'Clear (c=1.5m)']
    
    for i, (c_val, c_label) in enumerate(zip(c_slices, c_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Compute reward surface
        ZZ = np.zeros_like(DD)
        for ii in range(DD.shape[0]):
            for jj in range(DD.shape[1]):
                delta_d = DD[ii, jj]
                e = EE[ii, jj]
                rules = fuzzy._evaluate_rules(delta_d, e, c_val)
                ZZ[ii, jj] = fuzzy._defuzzify_sugeno(rules)
        
        surf = ax.plot_surface(DD, np.degrees(EE), ZZ, cmap='RdYlGn', 
                               alpha=0.8, edgecolor='none')
        ax.set_xlabel('Progress Δd (m)', fontsize=9)
        ax.set_ylabel('Heading Error (°)', fontsize=9)
        ax.set_zlabel('Reward', fontsize=9)
        ax.set_title(c_label, fontsize=11, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.view_init(elev=20, azim=135)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(outdir, 'reward_surface_progress_heading.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()
    
    # Surface 2: reward(Δd, c | e fixed)
    fig = plt.figure(figsize=(18, 5))
    
    c_grid = np.linspace(0, fuzzy.c_max, 40)
    DD, CC = np.meshgrid(delta_d_grid, c_grid)
    
    e_slices = [0.1, 0.8, 1.6]  # Aligned, Medium, Misaligned
    e_labels = ['Aligned (e=0.1rad ≈6°)', 'Medium (e=0.8rad ≈46°)', 'Misaligned (e=1.6rad ≈92°)']
    
    for i, (e_val, e_label) in enumerate(zip(e_slices, e_labels)):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Compute reward surface
        ZZ = np.zeros_like(DD)
        for ii in range(DD.shape[0]):
            for jj in range(DD.shape[1]):
                delta_d = DD[ii, jj]
                c = CC[ii, jj]
                rules = fuzzy._evaluate_rules(delta_d, e_val, c)
                ZZ[ii, jj] = fuzzy._defuzzify_sugeno(rules)
        
        surf = ax.plot_surface(DD, CC, ZZ, cmap='RdYlGn', 
                               alpha=0.8, edgecolor='none')
        ax.set_xlabel('Progress Δd (m)', fontsize=9)
        ax.set_ylabel('Clearance c (m)', fontsize=9)
        ax.set_zlabel('Reward', fontsize=9)
        ax.set_title(e_label, fontsize=11, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        ax.view_init(elev=20, azim=135)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(outdir, 'reward_surface_progress_clearance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")
        plt.close()
    else:
        plt.show()


def test_rule_firing(fuzzy):
    """Test rule firing for key scenarios."""
    print("="*70)
    print("RULE FIRING SANITY CHECKS")
    print("="*70)
    
    test_cases = [
        ("Ideal: toward+aligned+clear", 
         fuzzy.max_translation, 0.05, 1.5, 
         "Should be strongly positive (VeryGood)"),
        
        ("Danger override: toward+aligned+danger", 
         fuzzy.max_translation, 0.05, 0.1, 
         "Should be strongly negative (VeryBad)"),
        
        ("Misaligned but progressing", 
         fuzzy.max_translation, 1.8, 1.5, 
         "Should be near Neutral or slightly negative"),
        
        ("Retreating in open space", 
         -fuzzy.max_translation, 0.1, 1.5, 
         "Should be Bad"),
        
        ("Stuck and misaligned", 
         0.0, 1.5, 1.2, 
         "Should be Bad"),
        
        ("Perfect alignment, no progress, clear", 
         0.0, 0.0, 1.5, 
         "Should be Neutral (mild negative)"),
    ]
    
    for name, delta_d, e, c, expected in test_cases:
        rules = fuzzy._evaluate_rules(delta_d, e, c)
        reward = fuzzy._defuzzify_sugeno(rules)
        
        # Show top 3 firing rules
        sorted_rules = sorted(rules, key=lambda x: x[0], reverse=True)
        top_rules = sorted_rules[:3]
        
        print(f"\n{name}:")
        print(f"  Inputs: Δd={delta_d:+.3f}m, e={e:.2f}rad ({math.degrees(e):.1f}°), c={c:.2f}m")
        print(f"  Reward: {reward:+.3f}")
        print(f"  Expected: {expected}")
        print(f"  Top rules:")
        for w, z in top_rules:
            if w > 0.01:
                print(f"    w={w:.3f} → output={z:+.2f}")
    
    print("\n" + "="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Validate Fuzzy3 Reward')
    parser.add_argument('--save', action='store_true', 
                       help='Save plots instead of showing interactively')
    parser.add_argument('--outdir', type=str, default='runs_test/fuzzy3_viz',
                       help='Output directory for saved plots')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("FUZZY3 REWARD VALIDATION & VISUALIZATION")
    print("="*70 + "\n")
    
    # Initialize fuzzy reward
    fuzzy = Fuzzy3Reward()
    fuzzy.reset(5.0)  # Set initial distance
    
    # Test 1: Trapezoid shoulders
    shoulder_pass = test_trapezoid_shoulders()
    
    # Test 2: Rule firing
    test_rule_firing(fuzzy)
    
    # Test 3: Membership functions
    print("Plotting membership functions...")
    plot_membership_functions(fuzzy, save=args.save, outdir=args.outdir)
    
    # Test 4: Reward surfaces
    print("Plotting reward surfaces (this may take a moment)...")
    plot_reward_surfaces(fuzzy, save=args.save, outdir=args.outdir)
    
    if args.save:
        print(f"\n✓ All plots saved to: {args.outdir}/")
    else:
        print("\n✓ All plots displayed interactively")
    
    print("\n" + "="*70)
    print("VALIDATION COMPLETE")
    print("="*70)
    
    if not shoulder_pass:
        print("\n⚠ WARNING: Some shoulder tests failed!")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == '__main__':
    main()
