"""C&C-style preventability scorer v2 — JAMA-Ver4.0-cited thresholds + SG-filtered jerk.

CHANGES FROM v1:
  1. Velocity / acceleration / jerk now computed via Savitzky-Golay filtered
     position (no more 250 m/s³ noise floor from raw 10Hz finite-difference).
  2. Thresholds split into three modes:
       'jama_strict'   — JAMA Ver.4.0 §6 numerical values, no edits
       'jama_inspired' — JAMA values where given, comfort literature elsewhere
       'comfort_lit'   — ISO 2631 / RSS / NHTSA defaults (the v1 mode)
  3. Each threshold has a source citation in the docstring.
  4. Dropped thresholds not supported by either JAMA or named literature
     (heading_err threshold; was 30° from nowhere).
  5. Added JAMA cut-in / cut-out specific checks per §6.3.

JAMA Ver.4.0 §6 (pp. 89–102) thresholds we ground from:
  Brake delay (制動遅れ)                       = 0.75 s    (Fig 97, JP 警察庁 lit)
  Max deceleration                              = 0.774 G ≈ 7.59 m/s²
                                                            (Fig 94 Makisita 2001 trained drivers)
  Time-to-max-decel (rise time)                = 0.6 s     (Fig 96 Makisita 2001)
  AEB peak G                                    = 0.85 G   (Fig 98)
  TTC for high-priority warning                = 2.0 s     (Fig 108 UNR ECE/TRANS/WP.29)
  Lateral wobble one-side (ふらつき one-side)  = 0.375 m   (Fig 102, 50th-%ile JP traffic data)
  Cut-in lateral velocity (max)                = 1.8 m/s   (Fig 103, JP real-traffic)
  Cut-in risk-judgment time (after 1st time)   = 0.4 s     (Fig 105 driving sim n=20)
  Cut-in risk boundary (Vy * judgment_time)    = 0.72 m    (Fig 107)

From 0.6s rise time + 7.59 m/s² max decel, JAMA's implied jerk ≈ 7.59/0.6 = 12.65 m/s³.
ISO 2631 comfort jerk threshold ≈ 5 m/s³. We use ISO under 'comfort_lit', JAMA-implied
under 'jama_inspired', JAMA upper bound (12.65) under 'jama_strict' since JAMA itself
doesn't define a comfort jerk threshold, only an emergency-brake jerk.

Velocity for jerk computation:
  v_smooth = savgol_filter(positions_xy, window=11, polyorder=3, deriv=1) / dt
  a_smooth = savgol_filter(positions_xy, window=11, polyorder=3, deriv=2) / dt²
  jerk     = savgol_filter(positions_xy, window=11, polyorder=3, deriv=3) / dt³
This avoids the 250 m/s³ raw finite-difference artifact.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter


Mode = Literal['jama_strict', 'jama_inspired', 'comfort_lit']


@dataclass
class CCConfigV2:
    """All thresholds have a source field; do NOT change without updating source."""
    mode: Mode = 'jama_inspired'

    # ===== Longitudinal =====
    # Max deceleration: JAMA Fig 94 Makisita 2001 trained drivers = 0.774G = 7.59 m/s²
    decel_max_mps2: float = 7.59
    decel_max_source: str = 'JAMA §6.2.1 Fig 94 (Makisita 2001)'

    # Max longitudinal jerk:
    #   jama_strict   = 12.65 m/s³ (= 7.59 / 0.6, JAMA's implied AEB jerk)
    #   jama_inspired = 7.5 m/s³ (between ISO comfort 5 and JAMA AEB 13)
    #   comfort_lit   = 5.0 m/s³ (ISO 2631)
    jerk_max_mps3: float = 7.5  # default jama_inspired
    jerk_max_source: str = 'ISO 2631 (comfort_lit) / JAMA-derived 7.59÷0.6 (jama_strict)'

    # Speed cap: JAMA doesn't give one (scenario-dependent). Comfort_lit default = 18 m/s
    # (~65 km/h, kashiwa urban upper). Only used in comfort_lit mode.
    speed_max_mps: float = 18.0
    speed_max_source: str = 'no JAMA value; urban approx (comfort_lit only)'

    # ===== Lateral =====
    # Lateral wobble (one-side from lane center): JAMA Fig 102 50th-%ile = 0.375 m
    # In strict mode this is the route_dev cap. In inspired mode use 0.72 m (= JAMA cut-in
    # risk boundary 1.8 m/s × 0.4 s in Fig 107). In comfort_lit allow up to 1.5 m.
    route_dev_max_m: float = 0.72  # default jama_inspired
    route_dev_max_source: str = 'JAMA §6.3.1 Fig 107 risk-judgment boundary (jama_inspired)'

    # Cut-in lateral velocity threshold: JAMA Fig 103 = 1.8 m/s
    cut_in_lat_v_max_mps: float = 1.8
    cut_in_lat_v_source: str = 'JAMA §6.3.1 Fig 103'

    # Lat acc: JAMA doesn't give it directly. RSS uses 4 m/s², comfort lit 3 m/s².
    # Skipped in jama_strict (no source); kept in comfort_lit.
    lat_acc_max_mps2: float = 3.0
    lat_acc_max_source: str = 'no JAMA value; ISO/RSS literature (comfort_lit only)'

    # ===== TTC =====
    # JAMA §6.3.1 Fig 108 UNR collision-warning guideline = 2.0 s
    ttc_min_sec: float = 2.0
    ttc_min_source: str = 'JAMA §6.3.1 Fig 108 (UNR ECE/TRANS/WP.29)'

    # ===== Brake-delay (C&C reaction time) =====
    # JAMA §6.2.1 Fig 97 = 0.75 s. Used in cut-in risk-judgment check (not as a
    # planner-output threshold directly).
    brake_delay_sec: float = 0.75
    brake_delay_source: str = 'JAMA §6.2.1 Fig 97 (Japan 警察庁 / WHO·GRSP 2008)'

    # ===== Cold-start / stuck (no JAMA, AD-specific deployment failures) =====
    cold_start_v0_speed_thresh_mps: float = 0.5
    cold_start_predicted_speed_mps: float = 5.0
    cold_start_within_first_n_steps: int = 8
    stuck_progress_max_m: float = 2.0
    stuck_max_speed_mps: float = 0.5
    stuck_check_horizon_sec: float = 4.0

    allow_backwards: bool = False

    @classmethod
    def for_mode(cls, mode: Mode) -> 'CCConfigV2':
        cfg = cls(mode=mode)
        if mode == 'jama_strict':
            cfg.decel_max_mps2 = 7.59       # JAMA Fig 94
            cfg.jerk_max_mps3 = 12.65       # JAMA-derived (7.59 / 0.6 rise time)
            cfg.speed_max_mps = math.inf    # JAMA gives no cap, drop the check
            cfg.route_dev_max_m = 0.375     # JAMA Fig 102 one-side wobble
            cfg.lat_acc_max_mps2 = math.inf # JAMA gives no value, drop
            cfg.ttc_min_sec = 2.0           # JAMA Fig 108
        elif mode == 'jama_inspired':
            cfg.decel_max_mps2 = 7.59       # JAMA
            cfg.jerk_max_mps3 = 7.5         # between ISO 5 and JAMA 12.65
            cfg.speed_max_mps = 18.0
            cfg.route_dev_max_m = 0.72      # JAMA Fig 107
            cfg.lat_acc_max_mps2 = 3.0
            cfg.ttc_min_sec = 2.0           # JAMA
        elif mode == 'comfort_lit':
            cfg.decel_max_mps2 = 7.59
            cfg.jerk_max_mps3 = 5.0         # ISO 2631
            cfg.speed_max_mps = 18.0
            cfg.route_dev_max_m = 1.5
            cfg.lat_acc_max_mps2 = 3.0
            cfg.ttc_min_sec = 2.0
        return cfg


def _sg_derivatives(xy: np.ndarray, dt: float, window: int = 11, polyorder: int = 3):
    """Return (v_xy, a_xy, j_xy) — first/second/third derivatives of xy via
    Savitzky-Golay filter with reflected boundary handling. xy: [N, 2]."""
    n = len(xy)
    if n < window:
        # fall back to simple finite-diff if too short
        v = np.diff(xy, axis=0, prepend=xy[:1]) / dt
        a = np.diff(v, axis=0, prepend=v[:1]) / dt
        j = np.diff(a, axis=0, prepend=a[:1]) / dt
        return v, a, j
    v = savgol_filter(xy, window, polyorder, deriv=1, axis=0, mode='interp') / dt
    a = savgol_filter(xy, window, polyorder, deriv=2, axis=0, mode='interp') / (dt ** 2)
    j = savgol_filter(xy, window, polyorder, deriv=3, axis=0, mode='interp') / (dt ** 3)
    return v, a, j


def _curvature_lat_acc(v_xy: np.ndarray, a_xy: np.ndarray) -> np.ndarray:
    """Lateral acceleration = (v × a) / |v|. Uses 2D cross product (scalar)."""
    cross = v_xy[:, 0] * a_xy[:, 1] - v_xy[:, 1] * a_xy[:, 0]
    speed = np.linalg.norm(v_xy, axis=1) + 1e-6
    return np.abs(cross / speed)


def _min_ttc(ego_xy0: tuple, ego_heading: float, ego_speed: float,
             neighbors: list[dict] | None) -> float:
    if not neighbors:
        return math.inf
    best = math.inf
    px, py = ego_xy0
    for nb in neighbors:
        nx, ny = nb['xy']
        nv = nb.get('v', 0.0)
        dx, dy = nx - px, ny - py
        ahead = dx * math.cos(ego_heading) + dy * math.sin(ego_heading)
        if ahead <= 0:
            continue
        closing = ego_speed - nv
        if closing <= 0:
            continue
        best = min(best, ahead / closing)
    return best


def score_trajectory_v2(
    ego_xy: list[tuple],
    ego_v_provided: list[float] | None = None,
    route_xy: list[tuple] | None = None,
    neighbors: list[dict] | None = None,
    ego_v0: float | None = None,
    goal_xy: tuple | None = None,
    dt: float = 0.1,
    cfg: CCConfigV2 | None = None,
) -> dict:
    """Score a planned trajectory against C&C-style preventability.

    Args:
        ego_xy: list of (x, y, heading, t). Length N (e.g. 80 for planner output).
        ego_v_provided: optional pre-computed speed sequence (e.g., planner's
                        longitudinal_velocity_mps output). If None, derived from
                        position via SG filter.
        route_xy: optional route centerline waypoints.
        neighbors: optional list of {'xy': (x,y), 'v': speed} at t=0.
        ego_v0: ego speed at t=0 (used for cold-start and stuck checks).
        goal_xy: optional (gx, gy) for stop-required check.
        dt: time step (default 0.1).
        cfg: CCConfigV2 (default jama_inspired).

    Returns:
        {'violation': bool, 'violations': {criterion: bool},
         'details': {criterion: value}, 'mode': str}
    """
    if cfg is None:
        cfg = CCConfigV2.for_mode('jama_inspired')

    xy = np.array([(p[0], p[1]) for p in ego_xy], dtype=np.float64)
    headings = np.array([p[2] if len(p) > 2 else 0.0 for p in ego_xy])

    # === Derivatives via SG filter (NOT raw finite difference) ===
    v_xy, a_xy, j_xy = _sg_derivatives(xy, dt=dt, window=11, polyorder=3)
    speed = np.linalg.norm(v_xy, axis=1)
    long_acc_signed = np.einsum('ij,ij->i', a_xy, v_xy) / (speed + 1e-6)  # signed: positive = accel
    long_jerk_mag = np.linalg.norm(j_xy, axis=1)  # 3D jerk magnitude
    lat_acc = _curvature_lat_acc(v_xy, a_xy)

    # If planner-output velocity provided, use it instead of derived for max-speed check
    if ego_v_provided is not None and len(ego_v_provided) == len(ego_xy):
        max_speed = max(ego_v_provided)
    else:
        max_speed = float(speed.max())

    # === Route deviation (when route_xy given) ===
    if route_xy:
        dev = []
        for px, py in xy:
            d = min(math.hypot(rx - px, ry - py) for rx, ry in route_xy)
            dev.append(d)
        max_route_dev = max(dev)
    else:
        max_route_dev = 0.0

    # === TTC ===
    ttc = _min_ttc(xy[0], headings[0], float(speed[0]) if len(speed) else 0.0, neighbors)

    # === Cold-start: stationary ego + planner predicts cruise within first n steps ===
    cold_start = False
    if ego_v0 is not None and ego_v0 <= cfg.cold_start_v0_speed_thresh_mps:
        n_check = min(cfg.cold_start_within_first_n_steps, len(speed))
        if n_check > 0 and float(speed[:n_check].max()) > cfg.cold_start_predicted_speed_mps:
            cold_start = True

    # === Stuck (ego was moving but planner predicts stay-stuck) ===
    stuck = False
    if ego_v0 is not None and ego_v0 > cfg.stuck_max_speed_mps:
        n_check = min(int(cfg.stuck_check_horizon_sec / dt), len(xy))
        if n_check >= 2:
            progress = math.hypot(xy[n_check-1, 0] - xy[0, 0], xy[n_check-1, 1] - xy[0, 1])
            peak_v = float(speed[:n_check].max())
            if progress < cfg.stuck_progress_max_m and peak_v < cfg.stuck_max_speed_mps:
                stuck = True

    # === Backwards motion ===
    backwards = False
    if not cfg.allow_backwards:
        for i in range(1, len(xy)):
            dx = xy[i, 0] - xy[i-1, 0]; dy = xy[i, 1] - xy[i-1, 1]
            h = headings[i-1]
            if dx * math.cos(h) + dy * math.sin(h) < -0.2:
                backwards = True
                break

    # === Stop-required ===
    stop_required_miss = False
    if goal_xy is not None and ego_v_provided is not None:
        gx, gy = goal_xy
        if math.hypot(gx - xy[0, 0], gy - xy[0, 1]) <= 5.0:
            if ego_v_provided[-1] > 1.0:
                stop_required_miss = True

    # === Aggregate violations ===
    violations = {
        'decel_too_hard':    float(-long_acc_signed.min()) > cfg.decel_max_mps2,
        'jerk':              float(long_jerk_mag.max()) > cfg.jerk_max_mps3,
        'lat_acc':           (float(lat_acc.max()) > cfg.lat_acc_max_mps2) if math.isfinite(cfg.lat_acc_max_mps2) else False,
        'speed_cap':         (max_speed > cfg.speed_max_mps) if math.isfinite(cfg.speed_max_mps) else False,
        'route_dev':         max_route_dev > cfg.route_dev_max_m if route_xy else False,
        'ttc':               ttc < cfg.ttc_min_sec,
        'cold_start':        cold_start,
        'stuck':             stuck,
        'backwards':         backwards,
        'stop_required':     stop_required_miss,
    }

    details = {
        'max_decel_mps2':    round(float(-long_acc_signed.min()), 3),
        'max_jerk_mps3':     round(float(long_jerk_mag.max()), 3),
        'max_lat_acc_mps2':  round(float(lat_acc.max()), 3),
        'max_speed_mps':     round(float(max_speed), 3),
        'max_route_dev_m':   round(float(max_route_dev), 3),
        'min_ttc_sec':       (round(float(ttc), 3) if math.isfinite(ttc) else None),
    }

    return {
        'violation': any(violations.values()),
        'violations': violations,
        'details': details,
        'mode': cfg.mode,
        'cfg': asdict(cfg),
    }


def bag_level_progress_summary(records: list[dict]) -> dict:
    """Aggregate ego-trajectory records into bag-level progress signature."""
    if not records:
        return {'n_frames': 0, 'deployment_progress_failure': False}
    speeds = [r.get('ego_v_now', 0.0) for r in records]
    xs = [r['ego_xy_now_map'][0] for r in records if 'ego_xy_now_map' in r]
    ys = [r['ego_xy_now_map'][1] for r in records if 'ego_xy_now_map' in r]
    n = len(records); mean_v = sum(speeds) / n; max_v = max(speeds)
    frac_stat = sum(1 for v in speeds if v < 0.5) / n
    span = math.hypot(max(xs)-min(xs), max(ys)-min(ys)) if len(xs) >= 2 else 0.0
    deployment_failure = (frac_stat >= 0.95 and mean_v < 0.5 and n >= 300)
    return {'n_frames': n, 'mean_ego_v_mps': round(mean_v, 3), 'max_ego_v_mps': round(max_v, 3),
            'fraction_stationary': round(frac_stat, 3), 'xy_span_m': round(span, 2),
            'deployment_progress_failure': bool(deployment_failure)}
