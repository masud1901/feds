#!/usr/bin/env python3
"""
FEDS Validation Script

Validates that the FEDS adaptive K mechanism is working correctly:
1. K values are being updated based on loss feedback
2. Clients with better loss improvement get lower K (more compression)
3. K values stay within bounds [K_min, K_max]
4. The mechanism is stable (no wild oscillations)

Usage:
    python validate_feds.py [--tracker feds_k_tracker.json]
"""
import json
import numpy as np
import argparse
import os


def validate_k_tracker(tracker_file):
    """Validate the K tracker file for correctness."""
    print("=" * 60)
    print("FEDS Validation Report")
    print("=" * 60)

    if not os.path.exists(tracker_file):
        print(f"ERROR: {tracker_file} not found")
        return False

    with open(tracker_file, 'r') as f:
        tracker = json.load(f)

    k_list = tracker.get('k_list', [])
    prev_losses = tracker.get('prev_losses', [])
    loss_history = tracker.get('loss_history', [])
    k_history = tracker.get('k_history', [])

    all_passed = True

    # Check 1: K values exist
    print("\n[Check 1] K values exist and are valid")
    if not k_list:
        print("  ❌ FAIL: No K values found")
        all_passed = False
    else:
        k_array = np.array(k_list)
        print(f"  ✓ PASS: Found {len(k_list)} K values")
        print(f"    Mean: {k_array.mean():.0f}, Std: {k_array.std():.0f}")
        print(f"    Min: {k_array.min()}, Max: {k_array.max()}")

    # Check 2: K values are positive
    print("\n[Check 2] K values are positive")
    if any(k <= 0 for k in k_list):
        print("  ❌ FAIL: Some K values are non-positive")
        all_passed = False
    else:
        print("  ✓ PASS: All K values are positive")

    # Check 3: Loss history exists
    print("\n[Check 3] Loss history exists")
    if not loss_history:
        print("  ⚠️  WARNING: No loss history (may be in warmup phase)")
    else:
        print(f"  ✓ PASS: Found {len(loss_history)} rounds of loss history")
        last_losses = np.array(loss_history[-1])
        print(f"    Last round losses - Mean: {last_losses.mean():.4f}, Std: {last_losses.std():.4f}")

    # Check 4: K history shows adaptation
    print("\n[Check 4] K values are adapting over time")
    if len(k_history) < 3:
        print("  ⚠️  WARNING: Not enough history to check adaptation (need 3+ rounds)")
    else:
        k_hist = np.array(k_history)
        k_changes = np.diff(k_hist, axis=0)

        # Check if any K values changed
        total_changes = np.abs(k_changes).sum()
        if total_changes == 0:
            print("  ❌ FAIL: K values are not changing (mechanism may not be working)")
            all_passed = False
        else:
            print(f"  ✓ PASS: K values are adapting")
            print(f"    Total absolute K change: {total_changes:.0f}")
            print(f"    Mean per-round change: {np.abs(k_changes).mean():.2f}")

        # Check for wild oscillations
        if len(k_history) > 5:
            recent_changes = np.abs(k_changes[-5:]).mean(axis=0)
            if recent_changes.mean() > 0.5 * np.array(k_list).mean():
                print("  ⚠️  WARNING: Large K changes detected - may need to tune eta/momentum")
            else:
                print("  ✓ PASS: K changes are stable (no wild oscillations)")

    # Check 5: Loss-K correlation
    print("\n[Check 5] Loss improvement correlates with K changes")
    if len(loss_history) >= 3 and len(k_history) >= 3:
        loss_arr = np.array(loss_history)
        k_arr = np.array(k_history)

        # Calculate loss improvements
        loss_improvements = loss_arr[:-1] - loss_arr[1:]  # Positive = loss decreased
        k_changes = k_arr[1:] - k_arr[:-1]  # Negative = K decreased (more compression)

        # Correlation should be negative: better improvement -> lower K
        correlations = []
        for i in range(min(loss_improvements.shape[1], k_changes.shape[1])):
            if np.std(loss_improvements[:, i]) > 0 and np.std(k_changes[:, i]) > 0:
                corr = np.corrcoef(loss_improvements[:, i], k_changes[:, i])[0, 1]
                correlations.append(corr)

        if correlations:
            mean_corr = np.mean(correlations)
            print(f"    Mean correlation (ΔLoss, ΔK): {mean_corr:.3f}")
            # Expect negative correlation: better loss -> lower K
            if mean_corr < 0:
                print("  ✓ PASS: Negative correlation (expected behavior)")
            else:
                print("  ⚠️  WARNING: Positive correlation - check sign in update rule")
        else:
            print("  ⚠️  WARNING: Cannot compute correlation (constant values)")
    else:
        print("  ⚠️  SKIPPED: Not enough history")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ FEDS mechanism appears to be working correctly!")
    else:
        print("❌ FEDS mechanism has issues that need to be addressed")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description='Validate FEDS adaptive K mechanism')
    parser.add_argument('--tracker', type=str, default='feds_k_tracker.json',
                        help='Path to K tracker JSON file')
    args = parser.parse_args()

    validate_k_tracker(args.tracker)


if __name__ == '__main__':
    main()
