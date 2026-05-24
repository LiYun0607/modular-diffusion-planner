"""Extract REAL ego trajectories from a 2026-05-01-style sensor ros2 bag.

For each ego-pose message in /localization/kinematic_state, build a "next 8s"
future trajectory by accumulating positions+velocities from the subsequent
80 ego-pose messages (assuming 10Hz). The resulting trajectories are the
basis for the §7.1 Real-DPO LoRA's "chosen" trajectories.

Output format (per frame, JSONL):
  {
    "ts_ns": int,
    "ego_xy_now": [x, y, heading_rad],
    "ego_v_now": float,
    "ego_future": [(x_t, y_t, heading_t, v_t), ...],   # 80 points
    "is_clean_cc": bool,                                 # passes C&C filter
    "cc_violations": {criterion: bool},
    "cc_details": {...},
  }

Usage:
  python extract_real_ego_trajectories.py \\
    --bag /root/corl_work/from_car/field_2026-05-01_R1_rep1_baseline \\
    --out /root/corl_work/outputs/real_ego_trajectories_R1_rep1_baseline.jsonl \\
    [--horizon 8.0] [--dt 0.1]

Requires ROS humble + autoware_planning_msgs in env.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np


def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def extract_from_bag(bag_dir: str, horizon_sec: float = 8.0, dt: float = 0.1,
                     out_path: str = None, cc_check: bool = True) -> int:
    """Extract real ego trajectories. Returns number of frames processed."""
    try:
        from rclpy.serialization import deserialize_message
        from nav_msgs.msg import Odometry
    except Exception as e:
        print(f"ROS humble env missing: {e}")
        return 0

    sys.path.insert(0, os.path.dirname(__file__))
    if cc_check:
        try:
            from cc_violation_score import score_trajectory, CCConfig
            cfg = CCConfig()
        except Exception:
            cc_check = False
            cfg = None

    bag_dir = Path(bag_dir)
    db3s = sorted(bag_dir.glob('*.db3'))
    if not db3s:
        print(f"no db3 in {bag_dir}")
        return 0

    # First pass: collect all ego poses across all db3 chunks
    print(f"reading {len(db3s)} db3 chunks...")
    ego_records = []  # list of (ts_ns, x, y, yaw, vx)
    for db3 in db3s:
        conn = sqlite3.connect(str(db3))
        cur = conn.cursor()
        cur.execute("SELECT id FROM topics WHERE name='/localization/kinematic_state'")
        row = cur.fetchone()
        if row is None:
            conn.close()
            continue
        tid = row[0]
        cur.execute(f"SELECT timestamp, data FROM messages WHERE topic_id={tid} ORDER BY timestamp")
        for ts_ns, raw in cur.fetchall():
            try:
                msg = deserialize_message(raw, Odometry)
                p = msg.pose.pose.position
                o = msg.pose.pose.orientation
                yaw = quat_to_yaw(o.x, o.y, o.z, o.w)
                vx = msg.twist.twist.linear.x
                vy = msg.twist.twist.linear.y
                ego_records.append((int(ts_ns), float(p.x), float(p.y), float(yaw),
                                    float(math.hypot(vx, vy))))
            except Exception:
                continue
        conn.close()
    print(f"collected {len(ego_records)} ego poses")

    if not ego_records:
        return 0

    # Sort by timestamp; dedupe by ts
    ego_records.sort(key=lambda r: r[0])
    horizon_ns = int(horizon_sec * 1e9)
    dt_ns = int(dt * 1e9)

    out_file = open(out_path, 'w') if out_path else None
    n_written = 0
    n_clean = 0
    for i, (ts0, x0, y0, yaw0, v0) in enumerate(ego_records):
        # Find ego poses at ts0 + k*dt for k=1..80
        future = []
        target_ts = ts0
        j = i
        for k in range(80):
            target_ts += dt_ns
            # advance j to first ts >= target_ts
            while j < len(ego_records) and ego_records[j][0] < target_ts:
                j += 1
            if j >= len(ego_records):
                break
            t, x, y, y_, v = ego_records[j]
            if t - target_ts > 5 * dt_ns:  # too far in future, missing data
                break
            # transform to ego frame at ts0
            dx = x - x0
            dy = y - y0
            cos0 = math.cos(yaw0); sin0 = math.sin(yaw0)
            ex = dx * cos0 + dy * sin0
            ey = -dx * sin0 + dy * cos0
            ey_h = y_ - yaw0
            future.append((float(ex), float(ey), float(ey_h), float(v)))

        if len(future) < 40:  # need at least 4s into future
            continue

        record = {
            'ts_ns': ts0,
            'ego_xy_now_map': [x0, y0, yaw0],
            'ego_v_now': v0,
            'ego_future_egoframe': future,
        }

        if cc_check and cfg is not None:
            ego_xy = [(p[0], p[1], p[2], k * dt) for k, p in enumerate(future)]
            ego_v = [p[3] for p in future]
            s = score_trajectory(ego_xy=ego_xy, ego_v=ego_v, ego_v0=v0, dt=dt, cfg=cfg)
            record['is_clean_cc'] = not s['violation']
            record['cc_violations'] = {k: bool(v) for k, v in s['violations'].items()}
            record['cc_details'] = s['details']
            if not s['violation']:
                n_clean += 1

        if out_file is not None:
            out_file.write(json.dumps(record, default=float) + '\n')
        n_written += 1

    if out_file is not None:
        out_file.close()
    print(f"wrote {n_written} trajectory records; {n_clean} pass C&C filter ({100*n_clean/max(n_written,1):.1f}%)")
    return n_written


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bag', required=True)
    p.add_argument('--out', default=None)
    p.add_argument('--horizon', type=float, default=8.0)
    p.add_argument('--dt', type=float, default=0.1)
    p.add_argument('--no-cc', action='store_true', help='skip C&C filtering')
    args = p.parse_args()
    extract_from_bag(args.bag, args.horizon, args.dt, args.out, cc_check=not args.no_cc)
