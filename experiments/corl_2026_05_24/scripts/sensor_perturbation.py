"""JAMA-Annex-E-inspired sensor / perception perturbation operators for
Diffusion-Planner inputs (CoRL §6.5 / §7 robustness ablation).

JAMA Ver.4.0 Annex E lists perception perturbations (radar / LiDAR / camera).
Since our planner consumes already-tracked objects (perception output, not raw
LiDAR), we operate at the *perception output* level. Reviewer can verify this
is consistent with JAMA Annex E.3 (LiDAR) + F.3 (perturbation validity).

Operators (each takes a planner-input dict and returns a perturbed copy):

  drop_objects(p_drop)              # randomly remove a fraction of neighbors
  jitter_positions(sigma_m)         # Gaussian noise on neighbor xy positions
  jitter_velocities(sigma_mps)      # Gaussian noise on neighbor vx, vy
  fp_objects(n_fake, range_m)       # inject n fake objects near ego
  freeze_perception(n_frames)       # hold ego_neighbors_past constant for N frames
  occlude_sector(theta_deg, width)  # remove objects in a sector around ego heading

We also provide:
  combined_perturbation(p_drop, sigma_pos, sigma_vel)  # use all at low intensity

Runner:
  run_perturbation_sweep(model_callable, base_input, perturbations, n_trials=5)
    -> {perturbation_name: list_of_C&C_results}
    For each perturbation level, applies perturbation, runs model, scores
    trajectory via cc_violation_score.

Doc only — model_callable is provided by caller; we focus on data perturbation
and result aggregation. Full integration with Diffusion_Planner happens at
runtime where the caller wraps model inference.
"""
from __future__ import annotations
import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class PerturbationSpec:
    """One perturbation to apply. operator is a function (input, **kwargs) -> input."""
    name: str
    operator: Callable
    params: dict = field(default_factory=dict)


# ---- Operators (npz-dict in → npz-dict out, deep-copies) ----

def drop_objects(inp: dict, p_drop: float = 0.2, seed: int | None = None) -> dict:
    """Randomly zero out a fraction of neighbor tracks."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()  # [N, T, 11]
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    keep_mask = rng.random(n) >= p_drop
    arr[~keep_mask] = 0.0
    out['neighbor_agents_past'] = arr
    return out


def jitter_positions(inp: dict, sigma_m: float = 0.5, seed: int | None = None) -> dict:
    """Add Gaussian noise to neighbor xy positions (columns 0, 1).
    Same noise per (neighbor, t) — simulates registration noise."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()  # [N, T, 11]
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma_m, size=(arr.shape[0], arr.shape[1], 2)).astype(arr.dtype)
    arr[..., 0:2] += noise
    out['neighbor_agents_past'] = arr
    return out


def jitter_velocities(inp: dict, sigma_mps: float = 0.5, seed: int | None = None) -> dict:
    """Add Gaussian noise to neighbor velocity components (cols 5, 6 typically vx, vy).
    Note: column indices for vx/vy depend on neighbor_agents_past schema; default to
    (5, 6) for autoware nuPlan format."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma_mps, size=(arr.shape[0], arr.shape[1], 2)).astype(arr.dtype)
    arr[..., 5:7] += noise
    out['neighbor_agents_past'] = arr
    return out


def fp_objects(inp: dict, n_fake: int = 2, range_m: float = 15.0,
               seed: int | None = None) -> dict:
    """Inject n_fake false-positive neighbors at random angles within range_m."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()  # [N, T, 11]
    rng = np.random.default_rng(seed)
    n_slots = arr.shape[0]
    empty_slots = [i for i in range(n_slots) if not np.any(arr[i] != 0.0)]
    n_to_inject = min(n_fake, len(empty_slots))
    for slot in rng.choice(empty_slots, n_to_inject, replace=False):
        theta = rng.uniform(0, 2 * math.pi)
        d = rng.uniform(2.0, range_m)
        x, y = d * math.cos(theta), d * math.sin(theta)
        arr[slot, :, 0] = x + rng.normal(0, 0.1, arr.shape[1])
        arr[slot, :, 1] = y + rng.normal(0, 0.1, arr.shape[1])
        arr[slot, :, 5] = rng.normal(0, 0.3, arr.shape[1])  # vx noise
        arr[slot, :, 6] = rng.normal(0, 0.3, arr.shape[1])  # vy noise
    out['neighbor_agents_past'] = arr
    return out


def freeze_perception(inp: dict, n_frames: int = 3) -> dict:
    """Hold neighbor history constant for last n_frames (simulates perception lag).
    Effectively duplicates the t=-n_frames frame across the last n_frames slots."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()
    t = arr.shape[1]
    if n_frames >= t:
        n_frames = t - 1
    arr[:, -n_frames:, :] = arr[:, -n_frames-1:-n_frames, :]
    out['neighbor_agents_past'] = arr
    return out


def occlude_sector(inp: dict, theta_deg_center: float = 0.0,
                   width_deg: float = 90.0) -> dict:
    """Remove objects within ±width_deg/2 of theta_deg_center (relative to ego heading)."""
    out = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in inp.items()}
    if 'neighbor_agents_past' not in out:
        return out
    arr = out['neighbor_agents_past'].copy()
    half = math.radians(width_deg / 2)
    center = math.radians(theta_deg_center)
    # use last-step position to determine angle from ego
    for i in range(arr.shape[0]):
        if not np.any(arr[i] != 0.0):
            continue
        x = arr[i, -1, 0]
        y = arr[i, -1, 1]
        theta = math.atan2(y, x)
        delta = math.atan2(math.sin(theta - center), math.cos(theta - center))
        if abs(delta) <= half:
            arr[i] = 0.0
    out['neighbor_agents_past'] = arr
    return out


def combined_perturbation(inp: dict, p_drop: float = 0.1, sigma_pos: float = 0.3,
                          sigma_vel: float = 0.3, seed: int | None = None) -> dict:
    out = drop_objects(inp, p_drop=p_drop, seed=seed)
    out = jitter_positions(out, sigma_m=sigma_pos, seed=seed)
    out = jitter_velocities(out, sigma_mps=sigma_vel, seed=seed)
    return out


# ---- Sweep runner (model-agnostic) ----

def default_perturbation_grid() -> list[PerturbationSpec]:
    """The default sweep we report in the paper. Covers JAMA Annex E categories."""
    grid = []
    for p in [0.10, 0.20, 0.30]:
        grid.append(PerturbationSpec(f'drop_p{p:.2f}', drop_objects, dict(p_drop=p)))
    for s in [0.1, 0.5, 1.0]:
        grid.append(PerturbationSpec(f'jitter_pos_s{s:.1f}m', jitter_positions, dict(sigma_m=s)))
    for s in [0.5, 1.0, 2.0]:
        grid.append(PerturbationSpec(f'jitter_vel_s{s:.1f}mps', jitter_velocities, dict(sigma_mps=s)))
    for n in [1, 2, 3]:
        grid.append(PerturbationSpec(f'fp_objects_n{n}', fp_objects, dict(n_fake=n)))
    for n in [1, 3, 5]:
        grid.append(PerturbationSpec(f'freeze_n{n}', freeze_perception, dict(n_frames=n)))
    grid.append(PerturbationSpec('occlude_front_90', occlude_sector,
                                  dict(theta_deg_center=0, width_deg=90)))
    grid.append(PerturbationSpec('combined_low', combined_perturbation,
                                  dict(p_drop=0.1, sigma_pos=0.3, sigma_vel=0.3)))
    grid.append(PerturbationSpec('combined_high', combined_perturbation,
                                  dict(p_drop=0.3, sigma_pos=1.0, sigma_vel=1.0)))
    return grid


def run_perturbation_sweep(
    model_inference_fn: Callable,
    cc_score_fn: Callable,
    base_inputs: list[dict],
    perturbations: list[PerturbationSpec] | None = None,
    n_trials_per_perturbation: int = 3,
    seed_base: int = 0,
) -> dict:
    """Run a sweep:
       for each perturbation:
         for each n_trials random seed:
           for each base input (e.g., 50 npz scenes):
             perturbed = perturbation(base_input)
             ego_traj_pred = model_inference_fn(perturbed)
             cc = cc_score_fn(ego_traj_pred)
             accumulate

    Returns:
      {perturbation_name: {
         'n_inputs': N,
         'n_trials': T,
         'mean_violation_rate': float,
         'p95_violation_rate': float,
         'per_trial': list[float],
         'delta_vs_unperturbed': float,
      }, ...}
    """
    if perturbations is None:
        perturbations = default_perturbation_grid()

    # Baseline (no perturbation) violation rate
    base_violations = []
    for inp in base_inputs:
        traj = model_inference_fn(inp)
        s = cc_score_fn(traj)
        base_violations.append(1 if s.get('violation', False) else 0)
    base_rate = float(np.mean(base_violations)) if base_violations else 0.0

    out = {'_baseline_no_perturbation': {
        'n_inputs': len(base_inputs), 'mean_violation_rate': base_rate}}

    for spec in perturbations:
        trial_rates = []
        for t in range(n_trials_per_perturbation):
            seed = seed_base + t
            v = []
            for inp in base_inputs:
                params = dict(spec.params)
                if 'seed' not in params and spec.operator not in (freeze_perception, occlude_sector):
                    params['seed'] = seed
                perturbed = spec.operator(inp, **params)
                traj = model_inference_fn(perturbed)
                s = cc_score_fn(traj)
                v.append(1 if s.get('violation', False) else 0)
            trial_rates.append(float(np.mean(v)) if v else 0.0)
        mean_rate = float(np.mean(trial_rates))
        out[spec.name] = {
            'n_inputs': len(base_inputs),
            'n_trials': n_trials_per_perturbation,
            'mean_violation_rate': mean_rate,
            'p95_violation_rate': float(np.percentile(trial_rates, 95)) if trial_rates else 0.0,
            'per_trial': trial_rates,
            'delta_vs_unperturbed': mean_rate - base_rate,
        }
    return out


# ---- Smoke test ----

def _self_test():
    """Confirm operators don't crash on a synthetic npz dict and produce changes."""
    rng = np.random.default_rng(0)
    base = {
        'neighbor_agents_past': rng.normal(0, 5, size=(8, 21, 11)).astype(np.float32),
        'ego_current_state': np.array([0, 0, 0, 5], dtype=np.float32),
    }
    base['neighbor_agents_past'][6:] = 0.0  # 2 empty slots

    for spec in default_perturbation_grid():
        p = dict(spec.params)
        if 'seed' not in p and spec.operator not in (freeze_perception, occlude_sector):
            p['seed'] = 42
        out = spec.operator(base, **p)
        diff = np.abs(out['neighbor_agents_past'] - base['neighbor_agents_past']).sum()
        print(f"  {spec.name:<28s} sum|Δ| = {diff:.3f}")


if __name__ == '__main__':
    _self_test()
