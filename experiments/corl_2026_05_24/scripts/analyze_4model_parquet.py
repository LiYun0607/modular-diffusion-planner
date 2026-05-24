"""Analyze the car-side 4-model AB parquet (per-frame ego_speed, ego_lat_acc,
ego_jerk_rms, route_dev, traj_end_x/y, traj_L2_b_s).

Outputs:
  outputs/4model_per_model_stats.json
  outputs/4model_cc_violation_rates.json   (parquet-level: route_dev + traj_L2 only)
  outputs/4model_route_dev_hist.png
  outputs/4model_traj_L2_hist.png
  outputs/4model_p95_bars.png
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# JAMA-inspired thresholds
ROUTE_DEV_MAX = 1.5  # m
TRAJ_DIVERGENCE_MAX = 5.0  # m


def main(parquet_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"  models: {sorted(df['model'].unique())}")

    models = sorted(df['model'].unique())
    stats = {}
    for m in models:
        sub = df[df['model'] == m]
        s = {
            'n_frames': int(len(sub)),
            'route_dev_mean': float(sub['route_dev'].mean()),
            'route_dev_p50': float(sub['route_dev'].quantile(0.50)),
            'route_dev_p95': float(sub['route_dev'].quantile(0.95)),
            'route_dev_p99': float(sub['route_dev'].quantile(0.99)),
            'route_dev_max': float(sub['route_dev'].max()),
        }
        if 'traj_L2_b_s' in sub.columns and sub['traj_L2_b_s'].notna().any():
            v = sub['traj_L2_b_s'].dropna()
            s.update({
                'traj_L2_b_s_n': int(len(v)),
                'traj_L2_b_s_mean': float(v.mean()),
                'traj_L2_b_s_p95': float(v.quantile(0.95)),
                'traj_L2_b_s_p99': float(v.quantile(0.99)),
                'traj_L2_b_s_max': float(v.max()),
            })
        stats[m] = s
    with open(os.path.join(out_dir, '4model_per_model_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print("== per-model stats:")
    for m, s in stats.items():
        line = (f"  {m:>14s}: route_dev mean={s['route_dev_mean']:.3f} "
                f"p95={s['route_dev_p95']:.3f} max={s['route_dev_max']:.3f}")
        if 'traj_L2_b_s_p95' in s:
            line += f"  traj_L2 p95={s['traj_L2_b_s_p95']:.3f} max={s['traj_L2_b_s_max']:.3f}"
        print(line)

    # C&C-style violation (parquet-level)
    cc = {}
    for m in models:
        sub = df[df['model'] == m]
        v_route = (sub['route_dev'] > ROUTE_DEV_MAX)
        if 'traj_L2_b_s' in sub.columns:
            v_traj = (sub['traj_L2_b_s'] > TRAJ_DIVERGENCE_MAX).fillna(False)
        else:
            v_traj = pd.Series([False] * len(sub))
        v_any = (v_route | v_traj)
        n = len(sub)
        cc[m] = {
            'n_frames': int(n),
            'route_dev_violation_rate': float(v_route.sum() / n),
            'traj_divergence_violation_rate': float(v_traj.sum() / n),
            'any_violation_rate': float(v_any.sum() / n),
        }
    with open(os.path.join(out_dir, '4model_cc_violation_rates.json'), 'w') as f:
        json.dump(cc, f, indent=2)
    print(f"== C&C violation (route>{ROUTE_DEV_MAX}m OR traj>{TRAJ_DIVERGENCE_MAX}m):")
    for m, c in cc.items():
        print(f"  {m:>14s}: any={c['any_violation_rate']*100:.2f}%")

    # plots
    colors = {'base_nolora': '#444', 'joint_replay': '#1f77b4',
              'kashiwa_alone': '#d62728', 'shared_only': '#9467bd'}
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, df['route_dev'].max() * 1.05, 60)
    for m in models:
        sub = df[df['model'] == m]
        ax.hist(sub['route_dev'], bins=bins, alpha=0.4,
                label=f"{m} (p95={sub['route_dev'].quantile(0.95):.2f}m)",
                color=colors.get(m))
    ax.axvline(ROUTE_DEV_MAX, color='red', linestyle='--', label=f'C&C {ROUTE_DEV_MAX}m')
    ax.set_xlabel('route deviation (m)')
    ax.set_ylabel('count')
    ax.set_title(f'Route deviation per model, garage v=0 replay')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '4model_route_dev_hist.png'), dpi=120)
    plt.close(fig)

    if 'traj_L2_b_s' in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in models:
            sub = df[df['model'] == m]
            v = sub['traj_L2_b_s'].dropna()
            if len(v) == 0:
                continue
            bins = np.logspace(-3, np.log10(max(v.max(), 1) + 0.1), 50)
            ax.hist(v, bins=bins, alpha=0.5,
                    label=f"{m} (p95={v.quantile(0.95):.2f}m, max={v.max():.2f}m)",
                    color=colors.get(m))
        ax.axvline(TRAJ_DIVERGENCE_MAX, color='red', linestyle='--',
                   label=f'C&C drift {TRAJ_DIVERGENCE_MAX}m')
        ax.set_xscale('log')
        ax.set_xlabel('trajectory L2 from baseline (m)')
        ax.set_ylabel('count')
        ax.set_title('Trajectory drift from base_nolora per model')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, '4model_traj_L2_hist.png'), dpi=120)
        plt.close(fig)

    xs = np.arange(len(models))
    w = 0.4
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(xs - w/2, [stats[m]['route_dev_p95'] for m in models], width=w,
           label='route_dev p95 (m)', color='#1f77b4')
    ax.bar(xs + w/2, [stats[m].get('traj_L2_b_s_p95', 0) for m in models], width=w,
           label='traj_L2_baseline p95 (m)', color='#d62728')
    ax.set_xticks(xs)
    ax.set_xticklabels(models, rotation=20)
    ax.set_ylabel('meters')
    ax.set_title('Per-model p95 deviation, garage v=0 replay')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, '4model_p95_bars.png'), dpi=120)
    plt.close(fig)

    print(f"\n== outputs saved to {out_dir}/")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--parquet', default='/root/corl_work/from_car/field_2026-05-21_offline_ab_4model/4model.parquet')
    p.add_argument('--out', default='/root/corl_work/outputs')
    args = p.parse_args()
    main(args.parquet, args.out)
