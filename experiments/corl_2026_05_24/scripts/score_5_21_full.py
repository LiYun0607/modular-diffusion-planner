"""Full-trajectory C&C scoring of the 4 model bags from the 2026-05-21 garage v=0
offline AB. Reads each model's trajectory bag via rclpy + sqlite, parses
80-point Trajectory messages, applies cc_violation_score.score_trajectory,
overrides route_dev violation using the parquet's per-frame route_dev value,
and saves per-model JSON.

Usage:
  source /opt/autoware/setup.bash && source /root/autoware_ws/install/setup.bash
  python score_5_21_full.py
"""
from __future__ import annotations
import glob
import json
import math
import os
import sqlite3
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cc_violation_score import score_trajectory, CCConfig


BAG_ROOT = '/root/corl_work/from_car/field_2026-05-21_offline_ab_4model'
DT = 0.1

# Mapping from (stage, side) → model label per fill.log's model order
MAPPING = {
    ('stage1', 'baseline_traj'):       'base_nolora',
    ('stage1', 'kashiwa_selora_traj'): 'kashiwa_alone',
    ('stage2', 'baseline_traj'):       'joint_replay',
    ('stage2', 'kashiwa_selora_traj'): 'shared_only',
}


def main(out_dir: str = '/root/corl_work/outputs'):
    try:
        from rclpy.serialization import deserialize_message
        from autoware_planning_msgs.msg import Trajectory
    except Exception as e:
        print(f"ROS env missing: {e}; source /opt/autoware/setup.bash first")
        sys.exit(1)

    parq = pd.read_parquet(os.path.join(BAG_ROOT, '4model.parquet'))
    cfg = CCConfig()
    all_results = {}

    for (stage, side), label in MAPPING.items():
        bag_dir = os.path.join(BAG_ROOT, stage, 'garage_v0_180856', side)
        db3s = sorted(glob.glob(os.path.join(bag_dir, '*.db3')))
        if not db3s:
            print(f"!! no db3 in {bag_dir}")
            continue

        parq_m = parq[parq['model'] == label].set_index('stamp_ns').sort_index()
        parq_route_dev = parq_m['route_dev'].to_dict()
        parq_ego_speed = parq_m['ego_speed'].to_dict()

        n_frames = 0
        n_violations = 0
        per_crit = {}
        details_list = []

        for db3 in db3s:
            conn = sqlite3.connect(db3)
            cur = conn.cursor()
            cur.execute("SELECT id FROM topics WHERE name='/planning/diffusion_planner/trajectory'")
            row = cur.fetchone()
            if row is None:
                conn.close()
                continue
            tid = row[0]
            cur.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
            for ts_ns, raw in cur.fetchall():
                msg = deserialize_message(raw, Trajectory)
                if len(msg.points) < 5:
                    continue
                ego_xy = [(p.pose.position.x, p.pose.position.y,
                           2 * math.atan2(p.pose.orientation.z, p.pose.orientation.w),
                           i * DT) for i, p in enumerate(msg.points)]
                ego_v = [float(p.longitudinal_velocity_mps) for p in msg.points]
                ego_v0 = parq_ego_speed.get(ts_ns, 0.0)

                r = score_trajectory(ego_xy, ego_v, route_xy=None, neighbors=None,
                                     ego_v0=ego_v0, dt=DT, cfg=cfg)
                parq_dev = parq_route_dev.get(ts_ns)
                if parq_dev is not None:
                    r['violations']['route_dev'] = parq_dev > cfg.route_dev_max_m
                    r['details']['route_dev_max'] = round(parq_dev, 3)
                r['violation'] = any(r['violations'].values())

                n_frames += 1
                if r['violation']:
                    n_violations += 1
                for k, v in r['violations'].items():
                    per_crit.setdefault(k, 0)
                    if v:
                        per_crit[k] += 1
                details_list.append(r['details'])
            conn.close()

        if n_frames == 0:
            continue

        all_results[label] = {
            'n_frames': n_frames,
            'any_violation_rate': n_violations / n_frames,
            'per_criterion_rate': {k: v / n_frames for k, v in per_crit.items()},
        }
        print(f"== {label} (n_frames={n_frames}): "
              f"any_violation={n_violations / n_frames * 100:.2f}%")
        for k, c in sorted(per_crit.items(), key=lambda x: -x[1]):
            if c > 0:
                print(f"    {k:<14s} {c / n_frames * 100:>6.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, '4model_full_cc_scoring.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
