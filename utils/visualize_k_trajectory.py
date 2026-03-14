#!/usr/bin/env python3
"""
FEDS K Trajectory Visualization

Generates publication-ready figures showing:
1. Per-client K values over training rounds
2. Loss improvement correlation with K changes
3. Communication savings compared to baseline

Usage:
    python visualize_k_trajectory.py [--input feds_k_trajectory.json] [--output figures/]
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def plot_k_trajectory(trajectory_data, output_dir):
    """Plot per-client K values over rounds."""
    k_history = np.array(trajectory_data['k_history'])
    num_users = trajectory_data['num_users']
    K_min = trajectory_data.get('K_min', 0)
    K_max = trajectory_data.get('K_max', k_history.max())

    rounds = np.arange(1, len(k_history) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Per-client K values
    ax1 = axes[0]
    for i in range(num_users):
        ax1.plot(rounds, k_history[:, i], alpha=0.6, linewidth=1, label=f'Client {i}')

    ax1.axhline(y=K_min, color='r', linestyle='--', alpha=0.5, label=f'K_min={K_min}')
    ax1.axhline(y=K_max, color='g', linestyle='--', alpha=0.5, label=f'K_max={K_max}')
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('K (Parameters Retained)', fontsize=12)
    ax1.set_title('FEDS: Per-Client Adaptive K Values Over Training', fontsize=14)
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: K statistics (mean, std, min, max)
    ax2 = axes[1]
    k_mean = k_history.mean(axis=1)
    k_std = k_history.std(axis=1)
    k_min_per_round = k_history.min(axis=1)
    k_max_per_round = k_history.max(axis=1)

    ax2.plot(rounds, k_mean, 'b-', linewidth=2, label='Mean K')
    ax2.fill_between(rounds, k_mean - k_std, k_mean + k_std, alpha=0.2, color='blue', label='±1 Std')
    ax2.plot(rounds, k_min_per_round, 'r--', alpha=0.7, label='Min K')
    ax2.plot(rounds, k_max_per_round, 'g--', alpha=0.7, label='Max K')
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('K Value', fontsize=12)
    ax2.set_title('FEDS: K Value Statistics Across Clients', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feds_k_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'feds_k_trajectory.png')}")


def plot_loss_k_correlation(trajectory_data, output_dir):
    """Plot correlation between loss improvement and K changes."""
    if 'loss_history' not in trajectory_data or len(trajectory_data['loss_history']) < 2:
        print("Insufficient loss history for correlation plot")
        return

    loss_history = np.array(trajectory_data['loss_history'])
    k_history = np.array(trajectory_data['k_history'])

    # Calculate loss improvements
    loss_improvements = loss_history[:-1] - loss_history[1:]

    # Calculate K changes
    k_changes = k_history[1:] - k_history[:-1]

    # Flatten for scatter plot
    loss_imp_flat = loss_improvements.flatten()
    k_change_flat = k_changes.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Scatter of loss improvement vs K change
    ax1 = axes[0]
    ax1.scatter(loss_imp_flat, k_change_flat, alpha=0.3, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Loss Improvement (ΔL)', fontsize=12)
    ax1.set_ylabel('K Change (ΔK)', fontsize=12)
    ax1.set_title('FEDS: Loss Improvement vs K Adaptation', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add correlation coefficient
    if len(loss_imp_flat) > 1:
        corr = np.corrcoef(loss_imp_flat, k_change_flat)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                transform=ax1.transAxes, fontsize=12,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Mean loss over rounds
    ax2 = axes[1]
    rounds = np.arange(1, len(loss_history) + 1)
    mean_loss = loss_history.mean(axis=1)
    std_loss = loss_history.std(axis=1)

    ax2.plot(rounds, mean_loss, 'b-', linewidth=2)
    ax2.fill_between(rounds, mean_loss - std_loss, mean_loss + std_loss, alpha=0.2)
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('FEDS: Mean Client Training Loss', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feds_loss_k_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'feds_loss_k_correlation.png')}")


def plot_communication_savings(trajectory_data, output_dir):
    """Plot communication savings compared to full transmission."""
    k_history = np.array(trajectory_data['k_history'])
    K_max = trajectory_data.get('K_max', k_history.max())
    num_users = trajectory_data['num_users']

    rounds = np.arange(1, len(k_history) + 1)

    # Calculate per-round communication ratio
    comm_ratio = k_history.mean(axis=1) / K_max
    savings = 1 - comm_ratio

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(rounds, savings * 100, 'b-', linewidth=2)
    ax.fill_between(rounds, 0, savings * 100, alpha=0.3, color='blue')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Add baseline comparison (e.g., DSFL static 50%)
    ax.axhline(y=50, color='g', linestyle=':', alpha=0.5, label='Fixed 50% compression')

    ax.set_xlabel('Communication Round', fontsize=12)
    ax.set_ylabel('Communication Savings (%)', fontsize=12)
    ax.set_title('FEDS: Adaptive Communication Savings', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feds_communication_savings.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(output_dir, 'feds_communication_savings.png')}")


def main():
    parser = argparse.ArgumentParser(description='Visualize FEDS K trajectory')
    parser.add_argument('--input', type=str, default='feds_k_trajectory.json',
                        help='Path to K trajectory JSON file')
    parser.add_argument('--output', type=str, default='figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load trajectory data
    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found. Run FEDS training first.")
        return

    with open(args.input, 'r') as f:
        trajectory_data = json.load(f)

    print(f"Loaded trajectory data: {len(trajectory_data['k_history'])} rounds")

    # Generate figures
    plot_k_trajectory(trajectory_data, args.output)
    plot_loss_k_correlation(trajectory_data, args.output)
    plot_communication_savings(trajectory_data, args.output)

    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
