"""End-to-end runner for the JAMA-inspired sensor perturbation sweep (§6.5).

Composes:
  - sensor_perturbation.default_perturbation_grid()
  - sensor_perturbation.run_perturbation_sweep()
  - cc_violation_score.score_trajectory()

Real usage: provide a model_inference_fn(scene_npz) -> trajectory dict, which
wraps your model of choice (base, reward-LoRA, SE-LoRA, real-DPO, ood-scaled).
The runner produces a JSON of per-model × per-perturbation violation rates.

See `mock_perturbation_sweep_demo.json` in outputs/ for example shape using a
mock model.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import numpy as np

# Allow co-located imports
sys.path.insert(0, os.path.dirname(__file__))
from sensor_perturbation import default_perturbation_grid, run_perturbation_sweep
from cc_violation_score import score_trajectory, CCConfig


def cc_score_wrapper(traj):
    """Convenience wrapper for the sweep runner: accepts the model output dict
    {ego_xy, ego_v, ...} and returns the C&C score dict."""
    return score_trajectory(**traj)


def sweep_all_models(model_inference_fns: dict,
                     base_inputs: list[dict],
                     out_json: str,
                     n_trials: int = 3,
                     seed_base: int = 0) -> dict:
    """For each named model, run the perturbation sweep and save results.

    Args:
      model_inference_fns: dict of {name: callable(scene_npz)->trajectory_dict}
      base_inputs: list of unperturbed scene_npz inputs
      out_json: write per-model sweep results to this JSON file
      n_trials: number of random-seed trials per perturbation level
      seed_base: starting random seed for trials
    Returns:
      {model_name: sweep_dict_from_run_perturbation_sweep}
    """
    grid = default_perturbation_grid()
    all_results = {}
    for name, fn in model_inference_fns.items():
        print(f"\n=== sweeping model: {name} ===")
        res = run_perturbation_sweep(
            model_inference_fn=fn,
            cc_score_fn=cc_score_wrapper,
            base_inputs=base_inputs,
            perturbations=grid,
            n_trials_per_perturbation=n_trials,
            seed_base=seed_base,
        )
        all_results[name] = res
        base_rate = res['_baseline_no_perturbation']['mean_violation_rate']
        print(f"  baseline (no perturbation): {base_rate*100:.2f}% violation")
        worst = sorted([(k, v['mean_violation_rate']) for k, v in res.items()
                        if k != '_baseline_no_perturbation'],
                       key=lambda x: -x[1])[:3]
        print(f"  worst-3 perturbations:")
        for k, r in worst:
            d = r - base_rate
            print(f"    {k:<26s} {r*100:>5.2f}%  Δ={d*100:+.2f}pp")
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved per-model sweep results: {out_json}")
    return all_results


def make_paper_figure(results: dict, out_png: str) -> None:
    """Generate the §6.5 paper figure: bar chart of per-model violation rate
    increase under each perturbation category.

    Categories grouped:
      drop_*       → object dropout
      jitter_pos_* → position noise
      jitter_vel_* → velocity noise
      fp_objects_* → false positive
      freeze_*     → perception freeze
      occlude_*    → sector occlusion
      combined_*   → combined low/high
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    categories = {
        'drop': lambda k: k.startswith('drop'),
        'jitter_pos': lambda k: k.startswith('jitter_pos'),
        'jitter_vel': lambda k: k.startswith('jitter_vel'),
        'fp_objects': lambda k: k.startswith('fp_objects'),
        'freeze': lambda k: k.startswith('freeze'),
        'occlude': lambda k: k.startswith('occlude'),
        'combined': lambda k: k.startswith('combined'),
    }

    model_names = list(results.keys())
    cat_names = list(categories.keys())
    deltas = {m: [] for m in model_names}
    for cat, pred in categories.items():
        for m in model_names:
            res = results[m]
            base_rate = res['_baseline_no_perturbation']['mean_violation_rate']
            ds = [v['mean_violation_rate'] - base_rate for k, v in res.items() if pred(k)]
            deltas[m].append(float(np.mean(ds)) if ds else 0.0)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cat_names))
    w = 0.8 / len(model_names)
    for i, m in enumerate(model_names):
        offset = (i - (len(model_names) - 1) / 2) * w
        ax.bar(x + offset, [d * 100 for d in deltas[m]], width=w, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=20)
    ax.set_ylabel('Δ C&C violation rate (pp)')
    ax.set_title('Per-model sensor-perturbation degradation, JAMA-inspired (§6.5)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    print(f"Saved paper figure: {out_png}")


if __name__ == '__main__':
    # Demo with mock model (real usage replaces these with actual models)
    rng = np.random.default_rng(42)

    def make_mock_inference(severity: float = 1.0):
        """Severity scales the model's sensitivity to neighbor proximity."""
        def fn(inp):
            arr = inp['neighbor_agents_past']
            ds = [np.hypot(arr[i, -1, 0], arr[i, -1, 1])
                  for i in range(arr.shape[0]) if np.any(arr[i] != 0)]
            nearest = min(ds) if ds else 100.0
            n_pts = 20
            dt = 0.1
            if nearest < 5.0 * severity:
                ego_xy = []
                ego_v = []
                v = 8.0
                x = 0.0
                for i in range(n_pts):
                    v = max(v - 2.0, 0.0)
                    x += v * dt
                    ego_xy.append((x, 0.0, 0.0, i * dt))
                    ego_v.append(v)
            else:
                ego_xy = [(i * 0.5, 0.0, 0.0, i * dt) for i in range(n_pts)]
                ego_v = [5.0] * n_pts
            return {'ego_xy': ego_xy, 'ego_v': ego_v, 'ego_v0': ego_v[0]}
        return fn

    base_inputs = []
    for s in range(30):
        arr = rng.normal(0, 10, size=(8, 21, 11)).astype(np.float32)
        n_active = rng.integers(2, 7)
        arr[n_active:] = 0.0
        base_inputs.append({'neighbor_agents_past': arr})

    models = {
        'mock_base':      make_mock_inference(severity=1.0),
        'mock_reward':    make_mock_inference(severity=1.5),  # over-reactive
        'mock_real_dpo':  make_mock_inference(severity=0.8),  # smoother
    }
    out = sweep_all_models(models, base_inputs,
                            '/root/corl_work/outputs/perturbation_sweep_3model_mock.json',
                            n_trials=2)
    try:
        make_paper_figure(out, '/root/corl_work/outputs/perturbation_sweep_paper_fig.png')
    except Exception as e:
        print(f"figure failed: {e}")
