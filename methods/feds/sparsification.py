# -*- coding: utf-8 -*-
"""FEDS sparsification utilities used by the unified experiments."""

import numpy as np


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

