# -*- coding: utf-8 -*-
"""
FEDS: Loss-Feedback Adaptive Sparsification for Federated Learning

Enhanced implementation with:
- Per-layer adaptive K values driven by loss feedback
- Momentum-smoothed K updates for stability
- Normalized loss improvement for robust adaptation
- Warmup phase for initial stability
- K trajectory logging for analysis
"""
import json
import os
from datetime import datetime

import numpy as np



def layerSparsification(flatten_weights, history, oldseperation, metrics=None):
    """
    FEDS Layer-wise Sparsification with adaptive K based on loss feedback.

    Args:
        flatten_weights: List of flattened weight arrays, one per client
        history: History object tracking rounds, errors, and global model
        oldseperation: Layer boundary indices
        metrics: List of per-client metrics dicts containing 'train_loss'

    Returns:
        List of sparsified global models for each client
    """
    iteration = history.round - 1  # global round (0-indexed)
    num_user = len(flatten_weights)
    dynamic_sparsification = True

    # Layer boundaries (take every other element as in original)
    seperation = oldseperation[::2]
    num_layers = len(seperation) - 1

    # Calculate total parameters
    d = len(flatten_weights[0])

    # === FEDS Configuration ===
    K_min = max(100, d // 100)  # At least 1% of parameters
    K_max = d
    WARMUP_ROUNDS = 5  # Use static K for initial stability
    MOMENTUM = 0.3  # Smoothing factor for K updates (0 = no smoothing)
    ETA_BASE = 1000000.0  # Base learning rate for K updates
    ETA_DECAY = 0.995  # Decay factor per round

    # === Load or Initialize K Tracker ===
    k_tracker_file = 'feds_k_tracker.json'
    try:
        with open(k_tracker_file, 'r') as f:
            k_tracker = json.load(f)
            k_list = k_tracker['k_list']
            prev_losses = k_tracker['prev_losses']
            loss_history = k_tracker.get('loss_history', [])
            k_history = k_tracker.get('k_history', [])
    except (FileNotFoundError, json.JSONDecodeError):
        # Initialize from K_initial.json or K_Speech_initial.json
        k_list = [d // 2] * num_user
        for k_file in ('K_initial.json', 'K_Speech_initial.json'):
            if os.path.exists(k_file):
                try:
                    with open(k_file, 'r') as f:
                        data = json.load(f)
                        initial_k = data.get('k_list', data)
                    if isinstance(initial_k[0], list):
                        k_list = [sum(k) for k in initial_k]
                    else:
                        k_list = list(initial_k)
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass
                break

        prev_losses = [None] * num_user
        loss_history = []
        k_history = []

    # Ensure k_list has correct length
    if len(k_list) != num_user:
        k_list = [k_list[0] if k_list else d // 2] * num_user

    # === Load CKA similarity for per-layer weighting ===
    cka_weights = None
    if iteration > 1 and os.path.exists('cka_similarity.json'):
        try:
            with open('cka_similarity.json', 'r') as f:
                cka_data = json.load(f)
                cka_weights = cka_data.get('per_layer_weights', None)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # === FEDS Adaptive K Update ===
    if iteration >= 1 and metrics is not None and iteration >= WARMUP_ROUNDS:
        # Extract current losses
        current_losses = []
        for i in range(num_user):
            current_losses.append(metrics[i].get('train_loss', 0.0))

        # Calculate loss improvements (Delta L)
        delta_L = []
        for i in range(num_user):
            if prev_losses[i] is not None and prev_losses[i] > 0:
                # Relative improvement: (prev - current) / prev
                # Positive means loss decreased (good)
                rel_improvement = (prev_losses[i] - current_losses[i]) / max(prev_losses[i], 1e-8)
                delta_L.append(rel_improvement)
            else:
                delta_L.append(0.0)

        # Calculate threshold tau (average improvement)
        valid_delta_L = [d for d in delta_L if d != 0.0]
        if valid_delta_L:
            tau = np.mean(valid_delta_L)
            # Add small epsilon to prevent division by zero
            std_delta = np.std(valid_delta_L) + 1e-8
        else:
            tau = 0.0
            std_delta = 1.0

        # Adaptive eta with decay
        eta = ETA_BASE * (ETA_DECAY ** iteration)

        # Per-client K update with momentum
        k_updates = []
        for i in range(num_user):
            if prev_losses[i] is not None:
                # Normalized delta (z-score like)
                normalized_delta = (delta_L[i] - tau) / std_delta

                # K update: better progress -> compress more (lower K)
                # The sign is negative because K represents number of parameters to KEEP
                # More progress = can afford to keep fewer parameters
                raw_update = -eta * normalized_delta

                # Apply momentum to smooth updates
                if len(k_history) > 0:
                    prev_k = k_history[-1][i] if i < len(k_history[-1]) else k_list[i]
                    momentum_update = MOMENTUM * raw_update + (1 - MOMENTUM) * (k_list[i] - prev_k)
                else:
                    momentum_update = raw_update

                # Update K with clipping
                new_k = int(np.clip(k_list[i] + momentum_update, K_min, K_max))
                k_list[i] = new_k
                k_updates.append(momentum_update)

        # Store current state
        prev_losses = current_losses
        loss_history.append(current_losses)
        k_history.append(list(k_list))

        # Keep history bounded
        if len(loss_history) > 100:
            loss_history = loss_history[-100:]
        if len(k_history) > 100:
            k_history = k_history[-100:]

        # Log K statistics
        k_array = np.array(k_list)
        print(f"[FEDS Round {iteration + 1}] K stats: mean={k_array.mean():.0f}, "
              f"std={k_array.std():.0f}, min={k_array.min()}, max={k_array.max()}")
        print(f"[FEDS Round {iteration + 1}] Loss stats: tau={tau:.6f}, "
              f"mean_loss={np.mean(current_losses):.4f}")

    elif iteration < WARMUP_ROUNDS:
        print(f"[FEDS Round {iteration + 1}] Warmup phase ({WARMUP_ROUNDS - iteration} rounds remaining)")

    # === Save K Tracker ===
    k_tracker = {
        'k_list': k_list,
        'prev_losses': prev_losses,
        'loss_history': loss_history,
        'k_history': k_history,
        'last_updated': datetime.now().isoformat()
    }
    with open(k_tracker_file, 'w') as f:
        json.dump(k_tracker, f)

    # === Save K Trajectory for Analysis ===
    if len(k_history) > 0:
        trajectory_file = 'feds_k_trajectory.json'
        trajectory_data = {
            'k_history': k_history,
            'loss_history': loss_history,
            'num_users': num_user,
            'K_min': K_min,
            'K_max': K_max
        }
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory_data, f)

    # === Calculate Model Difference ===
    model_difference = [user_weights - history.globals[iteration]
                        for user_weights in flatten_weights]

    # === Error Accumulation ===
    model_difference_AccError = [
        model_difference[i] + history.error[iteration][i]
        for i in range(num_user)
    ]

    # === CKA-based Layer Masking ===
    # Random masking based on CKA similarity (layers with high similarity
    # across clients can be more aggressively sparsified)
    mult_list = [np.array([1.0] * len(model_difference_AccError[0])) for _ in range(num_user)]
    if iteration > 2 and cka_weights is not None:
        try:
            for user in range(num_user):
                for layer in range(num_layers):
                    layer_start = seperation[layer]
                    layer_end = seperation[layer + 1]
                    # Higher CKA = more similar = can compress more
                    compress_prob = 1.0 - cka_weights[user][layer] if user < len(cka_weights) else 0.5
                    mult_list[user][layer_start:layer_end] = np.random.binomial(
                        1, compress_prob, size=layer_end - layer_start
                    )
        except (IndexError, TypeError):
            pass  # Fall back to uniform masking

    # === Top-K Sparsification ===
    mask_list = [np.array([0] * len(model_difference_AccError[0])) for _ in range(num_user)]
    actual_k_used = []
    for user in range(num_user):
        k = k_list[user] if isinstance(k_list[user], (int, float)) else int(sum(k_list[user]))
        if k > 0:
            # Apply CKA-based weighting
            weighted_diff = np.multiply(model_difference_AccError[user], mult_list[user])
            abs_diff = np.absolute(weighted_diff)

            # Select top-K elements
            k = min(k, len(abs_diff))  # Ensure k doesn't exceed array size
            if k > 0:
                indices = np.argpartition(abs_diff, -k)[-k:]
                mask_list[user][indices] = 1
            actual_k_used.append(k)
        else:
            actual_k_used.append(0)

    # Print sparsification stats
    print(f"[FEDS Round {iteration + 1}] Sparsification: "
          f"K_used={[int(k) for k in actual_k_used[:5]]}...")

    # === Apply Mask and Prepare Model to Send ===
    model_to_send = [
        np.multiply(mask_list[user], model_difference_AccError[user])
        for user in range(num_user)
    ]

    # === Update Error for Next Round ===
    new_error = [
        np.multiply(np.array([1] * len(mask_list[0])) - mask_list[user],
                    model_difference_AccError[user])
        for user in range(num_user)
    ]
    history.updateError(new_error)

    # === Aggregate Model Updates ===
    global_model_difference = model_to_send[0].copy()
    for user in range(1, num_user):
        global_model_difference += model_to_send[user]
    global_model_difference /= num_user

    # === Update Global Model ===
    global_model = history.globals[iteration] + global_model_difference
    history.updateGlobal(global_model)

    # Return global model replicated for each client
    return [global_model for _ in range(num_user)]


# =====================================================================
# New-style aggregators used by the v10/v11 FEDS notebooks
# =====================================================================


def params_to_flat(params_list):
    """Flatten a list of numpy arrays into a single 1D array."""
    return np.concatenate([p.astype(np.float32).ravel() for p in params_list])


def flat_to_params(flat, shapes):
    """Reshape a flat array back into a list of parameter arrays."""
    out, i = [], 0
    for s in shapes:
        n = int(np.prod(s))
        out.append(flat[i : i + n].reshape(s).astype(np.float32))
        i += n
    return out


def topk_sparse(delta, k):
    """Magnitude-based Top-K sparsification with error feedback residual."""
    k = max(1, min(int(k), len(delta)))
    mask = np.zeros_like(delta, dtype=np.float32)
    idx = np.argpartition(np.abs(delta), -k)[-k:]
    mask[idx] = 1.0
    sparse = delta * mask
    residual = delta * (1.0 - mask)
    return sparse, residual


def compute_server_priority(client_deltas, d, ema_alpha=0.7, prev_priority=None):
    """Rank-normalised variance priority, as in the v10/v11 notebooks."""
    stack = np.stack(client_deltas, axis=0)  # [N, d]
    mean = stack.mean(axis=0)
    var = ((stack - mean) ** 2).mean(axis=0)

    ranks = np.argsort(np.argsort(var)).astype(np.float32)  # [0..d-1]
    p = ranks / max(d - 1, 1)

    if prev_priority is not None:
        p = ema_alpha * p + (1.0 - ema_alpha) * prev_priority

    return p.astype(np.float32)


def priority_sparse(delta, priority, k):
    """FEDS client-side sparsification: phi_j = |delta_j| * p_j."""
    k = max(1, min(int(k), len(delta)))
    phi = np.abs(delta) * priority
    mask = np.zeros_like(delta, dtype=np.float32)
    idx = np.argpartition(phi, -k)[-k:]
    mask[idx] = 1.0
    sparse = delta * mask
    residual = delta * (1.0 - mask)
    return sparse, residual


def fixed_topk_aggregate(client_params, global_flat, errors, shapes, k_fixed):
    """Fixed Top-K aggregation with error feedback."""
    sparse_deltas, new_errors = [], []
    for i, params in enumerate(client_params):
        delta = (params_to_flat(params) - global_flat) + errors[i]
        sd, residual = topk_sparse(delta, k_fixed)
        sparse_deltas.append(sd)
        new_errors.append(residual)
    mean_sparse = np.mean(sparse_deltas, axis=0)
    new_flat = global_flat + mean_sparse
    return flat_to_params(new_flat, shapes), new_flat, new_errors


def dsfl_aggregate(
    client_params,
    global_flat,
    errors,
    shapes,
    k_list,
    layer_map,
    cka_scores_per_client,
    rng,
):
    """DSFL-style two-stage LSS + Top-K aggregation compatible with FEDS code."""
    sparse_deltas, new_errors = [], []
    for cid, params in enumerate(client_params):
        flat = params_to_flat(params)
        delta = (flat - global_flat) + errors[cid]

        mask = np.ones(len(delta), dtype=np.float32)
        cka_scores = cka_scores_per_client[cid]
        for lidx, sim in cka_scores.items():
            pos = np.where(layer_map == lidx)[0]
            if len(pos) == 0:
                continue
            keep_p = float(np.clip(1.0 - sim, 0.05, 1.0))
            mask[pos] = (rng.random(len(pos)) < keep_p).astype(np.float32)
        delta_lss = delta * mask

        sd, _ = topk_sparse(delta_lss, k_list[cid])
        sparse_deltas.append(sd)
        new_errors.append(delta - sd)

    mean_sparse = np.mean(sparse_deltas, axis=0)
    new_flat = global_flat + mean_sparse
    return flat_to_params(new_flat, shapes), new_flat, new_errors


def feds_aggregate(
    client_params,
    global_flat,
    errors,
    shapes,
    d,
    k_list,
    priority,
    round_idx,
    comp_warmup,
    prev_priority,
    ema_alpha=0.7,
):
    """FEDS v10/v11 variance-priority aggregation.

    ``k_list`` is the target K_i for each client after warmup; during warmup
    we interpolate from d/2.
    """
    warmup_frac = min(1.0, round_idx / max(comp_warmup, 1))
    eff_k = [int(d // 2 + warmup_frac * (ki - d // 2)) for ki in k_list]

    sparse_deltas, new_errors, raw_deltas = [], [], []
    for cid, params in enumerate(client_params):
        flat = params_to_flat(params)
        delta = (flat - global_flat) + errors[cid]
        raw_deltas.append(delta.copy())

        sd, residual = priority_sparse(delta, priority, eff_k[cid])
        sparse_deltas.append(sd)
        new_errors.append(residual)

    mean_sparse = np.mean(sparse_deltas, axis=0)
    new_flat = global_flat + mean_sparse
    new_priority = compute_server_priority(
        raw_deltas, d, ema_alpha=ema_alpha, prev_priority=prev_priority
    )

    k_arr = np.array(eff_k)
    print(
        f"  [FEDS] K={k_arr.mean():.0f}  compress={1 - k_arr.mean()/d:.1%}  "
        f"p_mean={new_priority.mean():.3f}"
    )

    return flat_to_params(new_flat, shapes), new_flat, new_errors, eff_k, new_priority

