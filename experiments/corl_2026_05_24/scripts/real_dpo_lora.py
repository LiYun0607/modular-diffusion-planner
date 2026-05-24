"""Real-Vehicle DPO LoRA Training (CoRL §7 method).

Pipeline:
  1. extract_pref_pairs_from_bag(real_bag, reward_lora_ckpt) -> list[dict]
     For each frame in a real-vehicle ros2 bag:
       - winner_traj = real driver's next 8s ego XY (filtered IFF passes C&C)
       - loser_traj  = reward-LoRA's sampled traj at same input (IFF VIOLATES C&C)
     Returns preference records {npz_path, trajectory_w, trajectory_l, ts_ns}.

  2. train_dpo_lora(pairs, base_pth, lora_cfg, args) -> lora_state_dict
     Wraps the existing train_dpo.compute_dpo_loss but with:
       - base model frozen
       - LoRA layers added (rank, alpha from lora_cfg)
       - Only LoRA params receive gradients
       - DPO loss = -log σ(-β * ((l_w - l_ref_w) - (l_l - l_ref_l)))
       - Reference model = frozen base (NOT base+reward-LoRA — that would bias)

  3. merge_and_export(lora_sd, base_pth, out_dir) -> ONNX
     Standard merge + ONNX export (same as v3_dpo).

This file provides a SELF-CONTAINED reference implementation and the C&C-filter
logic. The DPO loss + diffusion training kernel reuses
/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization/train_dpo.py
since that's been validated end-to-end (it produced v3_dpo, the only LoRA
variant that survived real-vehicle deployment).

USAGE (per-stage CLI):
  # 1. Generate pref pairs from a real bag
  python real_dpo_lora.py extract \\
     --real-bag  /root/corl_work/from_car/field_2026-05-01_R1_rep1_baseline \\
     --reward-lora-ckpt /tmp/lora_kashiwa_v5_pure.pth \\
     --base-ckpt /root/corl_work/from_car/.../best_model.pth \\
     --out /root/corl_work/outputs/pairs/real_dpo_pairs.jsonl

  # 2. Train DPO LoRA (delegates to existing train_dpo.py with LoRA wrapper)
  python real_dpo_lora.py train \\
     --pairs /root/corl_work/outputs/pairs/real_dpo_pairs.jsonl \\
     --base-ckpt /root/.../best_model.pth \\
     --lora-rank 8 --lora-alpha 32 \\
     --beta 0.1 --epochs 10 \\
     --out-dir /root/corl_work/outputs/real_dpo_lora_v1

  # 3. Merge + ONNX
  python real_dpo_lora.py export \\
     --base-ckpt /root/.../best_model.pth \\
     --lora-ckpt /root/corl_work/outputs/real_dpo_lora_v1/lora_best.pth \\
     --out-dir   /root/autoware_data/diffusion_planner/v4_real_dpo

IMPORTANT: This file does NOT execute training itself; it provides the
extraction + filter glue + thin wrappers around the existing train_dpo.py
that already has the validated DPO loss kernel. Actual training needs GPU
and bag→npz extraction pipeline (see _extract_real_traj_from_bag below for
the algorithm; ROS-specific implementation needs to be wired against the
real bag's /localization/kinematic_state + /tf topics).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from cc_violation_score import score_trajectory, CCConfig


# ---------------------------------------------------------------------------
# C&C-filter for preference pair acceptance
# ---------------------------------------------------------------------------

@dataclass
class PrefFilterConfig:
    """Acceptance criteria for chosen / rejected trajectories.

    A pair (winner, loser) is accepted ONLY if:
      - winner passes ALL C&C checks (the real driver did the right thing)
      - loser violates AT LEAST ONE C&C check (the reward-LoRA messed up)
    """
    cc: CCConfig = None  # if None, defaults
    min_loser_severity: int = 1  # ≥1 violations needed in loser
    require_winner_clean: bool = True  # winner must violate 0


def accept_pair(winner: dict, loser: dict, cfg: PrefFilterConfig) -> tuple[bool, dict]:
    """Decide whether to keep this (winner, loser) preference pair.

    winner/loser dicts: {'ego_xy': [...], 'ego_v': [...], 'route_xy': [...], 'neighbors': [...]}
    """
    if cfg.cc is None:
        cfg.cc = CCConfig()
    w_score = score_trajectory(cfg=cfg.cc, **winner)
    l_score = score_trajectory(cfg=cfg.cc, **loser)
    w_n_violations = sum(1 for v in w_score['violations'].values() if v)
    l_n_violations = sum(1 for v in l_score['violations'].values() if v)
    accept = (
        (not cfg.require_winner_clean or w_n_violations == 0) and
        (l_n_violations >= cfg.min_loser_severity)
    )
    return accept, {
        'winner_violations': w_score['violations'],
        'loser_violations': l_score['violations'],
        'winner_n_vio': w_n_violations,
        'loser_n_vio': l_n_violations,
    }


# ---------------------------------------------------------------------------
# Real-trajectory extraction from a ros2 bag (skeleton)
# ---------------------------------------------------------------------------

def _extract_real_traj_from_bag(bag_dir: str | Path, horizon_sec: float = 8.0,
                                dt: float = 0.1) -> list[dict]:
    """For each planning trigger (e.g., every 0.1s ego pose), extract the next
    horizon_sec of REAL ego trajectory from /localization/kinematic_state.

    Returns: list of {ts_ns, ego_xy (real future), ego_v, ego_v0}.

    SKELETON: real implementation needs:
      - sqlite3 connect to each .db3
      - SELECT messages from /localization/kinematic_state ORDER BY timestamp
      - For each frame ts_now, slice the next horizon_sec/dt frames
      - Convert poses (map frame) to ego frame at ts_now
      - Bundle as the 80-point future trajectory in ego frame
    The script that produced 4model.parquet has equivalent logic — wrap it.
    """
    # Placeholder: returns empty list. Actual extraction requires ROS env;
    # we assume car-side already produced parquet with ego state + use that
    # as the seed for pair generation in real run.
    return []


def _sample_reward_lora_traj(npz_path: str, reward_lora_ckpt: str,
                             base_ckpt: str, n_samples: int = 1) -> list[dict]:
    """Run the reward-LoRA model on a given npz input and sample n trajectories.

    SKELETON: real impl loads base model, applies reward-LoRA delta, runs DPM
    solver forward `n_samples` times with different noise seeds, and returns the
    decoded trajectories in the format expected by score_trajectory.

    Uses /root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization/
    train_reward_backprop.py:dpm_sample() or equivalent.
    """
    return []


def extract_pref_pairs_from_bag(real_bag: str | Path,
                                reward_lora_ckpt: str | Path,
                                base_ckpt: str | Path,
                                npz_dir: str | Path,
                                out_jsonl: str | Path,
                                horizon_sec: float = 8.0,
                                dt: float = 0.1,
                                pref_cfg: PrefFilterConfig | None = None,
                                max_pairs: int = 10000) -> int:
    """Produce preference pairs from one real-vehicle bag.

    For each frame:
      1. Build the input npz (from /lidar/perception/route/ego_history at ts).
      2. winner = real ego trajectory over next horizon_sec.
      3. loser = sample from reward-LoRA model at this input.
      4. apply C&C filter; if accepted, write a pair.

    Writes JSONL where each line is:
      {npz_path, ts_ns, winner: {...}, loser: {...}, filter_meta: {...}}
    """
    if pref_cfg is None:
        pref_cfg = PrefFilterConfig(cc=CCConfig(), min_loser_severity=1, require_winner_clean=True)

    real_traj_list = _extract_real_traj_from_bag(real_bag, horizon_sec=horizon_sec, dt=dt)
    n_accepted = 0
    with open(out_jsonl, 'w') as f:
        for real in real_traj_list:
            npz_path = os.path.join(npz_dir, f"frame_{real['ts_ns']}.npz")
            if not os.path.exists(npz_path):
                continue
            loser_samples = _sample_reward_lora_traj(npz_path, str(reward_lora_ckpt),
                                                    str(base_ckpt), n_samples=1)
            if not loser_samples:
                continue
            loser = loser_samples[0]
            ok, meta = accept_pair(real, loser, pref_cfg)
            if ok:
                rec = {
                    'ts_ns': real['ts_ns'],
                    'npz_path': npz_path,
                    'trajectory_w': np.asarray([(x, y) for (x, y, *_rest) in real['ego_xy']]).tolist(),
                    'trajectory_l': np.asarray([(x, y) for (x, y, *_rest) in loser['ego_xy']]).tolist(),
                    'filter_meta': meta,
                }
                f.write(json.dumps(rec) + '\n')
                n_accepted += 1
                if n_accepted >= max_pairs:
                    break
    return n_accepted


# ---------------------------------------------------------------------------
# Thin wrapper to launch DPO-LoRA training (delegates to existing train_dpo.py)
# ---------------------------------------------------------------------------

def train_dpo_lora_cli(pairs_jsonl: str, base_ckpt: str, out_dir: str,
                       lora_rank: int = 8, lora_alpha: int = 32,
                       beta: float = 0.1, epochs: int = 10, batch_size: int = 4,
                       lr: float = 1e-5) -> None:
    """Construct CLI invocation of the validated train_dpo.py with a LoRA wrapper.

    NOTE: train_dpo.py operates on the full model by default. To run DPO with
    LoRA we either:
      (a) Patch train_dpo.py to call apply_molora() after model load and freeze
          base params (one-line change near line ~75 after `model.to(DEVICE)`),
          OR
      (b) Write a thin wrapper script that does the patching at runtime
          (monkey-patch the train_one_epoch loop).
    The (a) approach is recommended for the paper §7 implementation since
    train_dpo.py is already validated for v3_dpo.

    Then preference JSONL is read by an adapted DPODataset (line 131 in
    train_dpo.py) — replace its rule-based pair generation with a 'realbag'
    mode that loads from our JSONL.
    """
    cmd = [
        'python', '/root/autoware_ws/scripts/train/Diffusion-Planner/'
        'preference_optimization/train_dpo.py',
        '--exp_name', 'real_dpo_lora',
        '--model_path', str(base_ckpt),
        '--train_npz_list', str(pairs_jsonl),  # adapted mode
        '--valid_npz_list', str(pairs_jsonl),  # split inside; here for skeleton
        '--preference_mode', 'rule',
        '--beta', str(beta),
        '--train_epochs', str(epochs),
        '--batch_size', str(batch_size),
        '--learning_rate', str(lr),
    ]
    # Pass --lora_rank, --lora_alpha via env so we don't have to extend the
    # arg parser of train_dpo.py — patch reads from env.
    env_extra = {
        'LORA_RANK': str(lora_rank),
        'LORA_ALPHA': str(lora_alpha),
        'OUT_DIR': str(out_dir),
    }
    print("=== Real-DPO LoRA training invocation:")
    for k, v in env_extra.items():
        print(f"  export {k}={v}")
    print('  ' + ' '.join(cmd))


# ---------------------------------------------------------------------------
# Self-test of the C&C filter
# ---------------------------------------------------------------------------

def _self_test():
    """Verify accept_pair selects (good winner, bad loser) pairs correctly."""
    n = 20; dt = 0.1
    # winner: clean slow drive at v=2 m/s
    winner = {
        'ego_xy': [(i * 0.2, 0.0, 0.0, i * dt) for i in range(n)],
        'ego_v': [2.0] * n,
        'ego_v0': 2.0,
    }
    # loser_a: v=0 cold-start drift (planner predicts 22 m/s)
    loser_a = {
        'ego_xy': [(i * 2.2 * dt, 0.0, 0.0, i * dt) for i in range(n)],
        'ego_v': [22.0] * n,
        'ego_v0': 0.0,
    }
    # loser_b: accelerate-through-turn (v8-style)
    R = 10.0
    loser_b = {
        'ego_xy': [(R * math.sin(i*0.1), R*(1 - math.cos(i*0.1)), i*0.1, i*dt) for i in range(n)],
        'ego_v': [4.0 + 0.5*i for i in range(n)],
        'ego_v0': 4.0,
    }
    cfg = PrefFilterConfig(cc=CCConfig(), min_loser_severity=1, require_winner_clean=True)

    for label, l in [('cold_start', loser_a), ('accel_through_turn', loser_b)]:
        ok, meta = accept_pair(winner, l, cfg)
        print(f"pair (winner=clean, loser={label}): accept={ok} winner_violations={meta['winner_n_vio']} loser_violations={meta['loser_n_vio']}")
        print(f"  loser bits: {[k for k, v in meta['loser_violations'].items() if v]}")

    # opposite: winner is dirty, loser is clean → should NOT accept
    bad_winner = loser_b
    clean_loser = winner
    ok, meta = accept_pair(bad_winner, clean_loser, cfg)
    print(f"pair (winner=dirty, loser=clean): accept={ok}  (should be False)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')
    sub.add_parser('self-test')
    e = sub.add_parser('extract')
    e.add_argument('--real-bag', required=True)
    e.add_argument('--reward-lora-ckpt', required=True)
    e.add_argument('--base-ckpt', required=True)
    e.add_argument('--npz-dir', required=True)
    e.add_argument('--out', required=True)
    t = sub.add_parser('train')
    t.add_argument('--pairs', required=True)
    t.add_argument('--base-ckpt', required=True)
    t.add_argument('--out-dir', required=True)
    t.add_argument('--lora-rank', type=int, default=8)
    t.add_argument('--lora-alpha', type=int, default=32)
    t.add_argument('--beta', type=float, default=0.1)
    t.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    if args.cmd == 'self-test' or args.cmd is None:
        _self_test()
    elif args.cmd == 'extract':
        n = extract_pref_pairs_from_bag(args.real_bag, args.reward_lora_ckpt,
                                        args.base_ckpt, args.npz_dir, args.out)
        print(f"wrote {n} pairs to {args.out}")
    elif args.cmd == 'train':
        train_dpo_lora_cli(args.pairs, args.base_ckpt, args.out_dir,
                           args.lora_rank, args.lora_alpha, args.beta, args.epochs)
