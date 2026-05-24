"""JAMA-inspired C&C Preventability PROXY v3 (NOT a full implementation).

EXPLICIT NON-CLAIM:
  This module is a JAMA-INSPIRED PROXY. It uses select numerical thresholds
  drawn from JAMA Ver.4.0 §6 (pp. 89-102) where they are explicit, and
  ADS comfort literature elsewhere. We make NO claim of implementing the
  full JAMA C&C Driver model — that requires a brake-delay simulator
  (Fig 97), a Responder/Initiator role-switching judge (§6.1.1), and
  scenario-specific risk-judgment boundaries (§6.3) that we do not build.

THREE LAYERS:
  1. strict_safety       — hard constraints only (collision, corridor, extreme g)
  2. c_and_c_driverlike  — behavior a careful & competent driver would not exhibit
  3. comfort             — comfort degradation (jerk, smoothness); NOT a safety claim

PER-VIOLATION OUTPUT:
  {name, threshold, measured, severity (= measured / threshold), triggered (bool),
   explanation (human-readable)}

PER-TRAJECTORY OUTPUT also includes:
  - scenario_bucket ∈ {cold_start, curve, straight, route_following, object_interaction}
  - details: raw measured signals
  - any_strict_safety_violation: bool (the headline safety summary)
  - any_driverlike_violation: bool
  - any_comfort_violation: bool

VELOCITY/ACCEL/JERK COMPUTATION:
  Savitzky-Golay filter (window=11, polyorder=3, mode='interp'), eliminating
  the raw-finite-difference noise floor (~250 m/s³ on 10Hz human data).
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, field
from typing import Literal

import numpy as np
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

@dataclass
class CCProxyConfig:
    """Each threshold field has a companion `_source` field documenting its
    origin (JAMA section / figure, or named literature). DO NOT change
    a threshold without updating its source."""

    # =================================================================
    # Layer 1: strict_safety (hard constraints, safety-critical)
    # =================================================================
    # Collision proxy: distance to nearest neighbor at any future step.
    # JAMA does not give an absolute number; we use a conservative 0.5 m
    # (vehicle half-width buffer). Treat as hard-fail.
    collision_min_distance_m: float = 0.5
    collision_min_distance_source: str = 'proxy: vehicle half-width buffer; JAMA gives no absolute'

    # Time-to-collision: planner-output trajectory's min TTC to any neighbor.
    # JAMA Fig 108 cites UNR ECE/TRANS/WP.29/1091 = 2.0 s for collision warning.
    ttc_min_sec: float = 2.0
    ttc_min_sec_source: str = 'JAMA Ver.4.0 §6.3.1 Fig 108 (UNR ECE/TRANS/WP.29/1091)'

    # Route corridor: max lateral distance from planned route.
    # JAMA Fig 102 50%-ile one-side wobble is 0.375 m (in-lane keeping).
    # We use 1.75 m as the HARD corridor (wobble + buffer to lane edge, since
    # lane is ~3.5 m). Beyond this counts as departure from planned route.
    route_corridor_max_m: float = 1.75
    route_corridor_source: str = 'proxy: JAMA Fig 102 wobble + half-lane buffer'

    # Extreme longitudinal deceleration: harder than C&C driver max.
    # JAMA Fig 94 trained drivers reach 0.774 G = 7.59 m/s². Anything beyond
    # 1 G = 9.81 m/s² is beyond physically common emergency braking.
    extreme_decel_mps2: float = 9.81
    extreme_decel_source: str = 'proxy: 1 G upper bound; JAMA C&C max is 0.774 G'

    # Extreme lateral acceleration: harder than comfortable cornering.
    # 0.6 G ≈ 6 m/s² is approaching tire-grip-loss in dry asphalt.
    extreme_lat_acc_mps2: float = 6.0
    extreme_lat_acc_source: str = 'proxy: ~0.6 G physics-of-grip approximation'

    # =================================================================
    # Layer 2: c_and_c_driverlike
    # =================================================================
    # Cold-start acceleration: ego ≈ stationary at t=0, but planner predicts
    # excessive speed within the first ~0.8 s. A C&C driver releases brake
    # gently, not slamming accelerator.
    cold_start_v0_thresh_mps: float = 0.5
    cold_start_window_n: int = 8                # 8 steps × 0.1s = 0.8s
    cold_start_max_predicted_speed_mps: float = 5.0
    cold_start_source: str = 'proxy: 0-30 km/h transition smoothness; not in JAMA explicit'

    # Cornering speed: v²·κ (= lat acc) within C&C envelope.
    # JAMA gives no direct cornering lat_acc; AASHTO/RSS uses ≤3 m/s² for
    # comfort + safe operation. Stricter than the strict_safety extreme.
    driverlike_lat_acc_max_mps2: float = 3.0
    driverlike_lat_acc_source: str = 'ADS lit: RSS / AASHTO comfort limit'

    # Heading-velocity consistency: predicted heading angle should not
    # deviate from the velocity-vector direction by more than ~20°.
    # (Big deviation means the model thinks the car is going somewhere
    # different than it's pointing — sliding / skid prediction).
    heading_velocity_max_rad: float = math.radians(20.0)
    heading_velocity_source: str = 'proxy: vehicle-dynamics sanity; not in JAMA explicit'

    # Smooth deceleration: ≤ JAMA C&C max 0.774 G.
    # This is the C&C ceiling — strict_safety allows up to 1 G, but a C&C
    # driver wouldn't normally need to exceed this.
    driverlike_decel_max_mps2: float = 7.59
    driverlike_decel_source: str = 'JAMA Ver.4.0 §6.2.1 Fig 94 (Makisita 2001 trained drivers)'

    # Smooth acceleration: ≤ 4 m/s² (~0.4 G), typical comfort upper bound.
    driverlike_accel_max_mps2: float = 4.0
    driverlike_accel_source: str = 'ADS lit: comfort accel cap'

    # Stuck-detection (deployment failure, not in JAMA): ego was moving
    # but planner predicts ≤0.5 m/s + ≤2 m progress over 4 s. Caught
    # R1_rep2 SE-LoRA real-vehicle failure.
    stuck_progress_max_m: float = 2.0
    stuck_max_speed_mps: float = 0.5
    stuck_check_horizon_sec: float = 4.0
    stuck_source: str = 'proxy: addresses observed deployment failures (R1_rep2)'

    # =================================================================
    # Layer 3: comfort (degradation, not safety)
    # =================================================================
    # Longitudinal jerk: ISO 2631 places comfort at ~5 m/s³.
    # We separate this from the strict layer because exceeding it is
    # uncomfortable but not unsafe (passenger lurch, not collision risk).
    jerk_comfort_max_mps3: float = 5.0
    jerk_comfort_source: str = 'ISO 2631 / general comfort literature'

    # Lateral acceleration comfort: < driverlike threshold (3) but still
    # uncomfortable above this; ~2 m/s² = passenger lurch threshold.
    lat_acc_comfort_max_mps2: float = 2.0
    lat_acc_comfort_source: str = 'ADS lit: passenger comfort lat'

    # Speed-change rate (= longitudinal acceleration absolute) comfort cap.
    speed_change_comfort_max_mps2: float = 3.0
    speed_change_comfort_source: str = 'ADS lit: brake/accelerate comfort'

    # =================================================================
    # Scenario classification thresholds (no severity, just bucketing)
    # =================================================================
    scenario_cold_start_v0_max_mps: float = 0.5
    scenario_curve_lat_acc_min_mps2: float = 1.0
    scenario_object_interaction_neighbor_range_m: float = 30.0


# ---------------------------------------------------------------------------
# Signal extraction (SG-filtered)
# ---------------------------------------------------------------------------

def _sg_derivatives(xy: np.ndarray, dt: float, window: int = 11, polyorder: int = 3):
    """Return (v_xy, a_xy, j_xy) via Savitzky-Golay smoothed derivatives.
    Uses mode='interp' which polynomial-extrapolates at boundaries (no
    nearest-padding artifacts)."""
    n = len(xy)
    if n < window:
        # fall back for very short sequences
        v = np.diff(xy, axis=0, prepend=xy[:1]) / dt
        a = np.diff(v, axis=0, prepend=v[:1]) / dt
        j = np.diff(a, axis=0, prepend=a[:1]) / dt
        return v, a, j
    v = savgol_filter(xy, window, polyorder, deriv=1, axis=0, mode='interp') / dt
    a = savgol_filter(xy, window, polyorder, deriv=2, axis=0, mode='interp') / (dt ** 2)
    j = savgol_filter(xy, window, polyorder, deriv=3, axis=0, mode='interp') / (dt ** 3)
    return v, a, j


def _signed_long_acc(v_xy: np.ndarray, a_xy: np.ndarray) -> np.ndarray:
    """Longitudinal acceleration projected onto velocity direction.
    Positive = accelerating, negative = decelerating."""
    speed = np.linalg.norm(v_xy, axis=1) + 1e-6
    return np.einsum('ij,ij->i', a_xy, v_xy) / speed


def _lat_acc_curvature(v_xy: np.ndarray, a_xy: np.ndarray) -> np.ndarray:
    """Lateral acceleration = (v × a) / |v| (2D cross product)."""
    cross = v_xy[:, 0] * a_xy[:, 1] - v_xy[:, 1] * a_xy[:, 0]
    speed = np.linalg.norm(v_xy, axis=1) + 1e-6
    return np.abs(cross / speed)


def _heading_vs_velocity(headings: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    """Per-step deviation between predicted heading and velocity-vector heading.
    Only meaningful when speed > 0.3 m/s."""
    v_heading = np.arctan2(v_xy[:, 1], v_xy[:, 0])
    err = np.arctan2(np.sin(headings - v_heading), np.cos(headings - v_heading))
    mask = np.linalg.norm(v_xy, axis=1) > 0.3
    return np.abs(err) * mask  # zero out low-speed where heading is ill-defined


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
        ahead_proj = dx * math.cos(ego_heading) + dy * math.sin(ego_heading)
        if ahead_proj <= 0:
            continue
        closing = ego_speed - nv
        if closing <= 0:
            continue
        best = min(best, ahead_proj / closing)
    return best


def _min_neighbor_distance(ego_xy: np.ndarray, neighbors: list[dict] | None) -> float:
    if not neighbors:
        return math.inf
    best = math.inf
    for nb in neighbors:
        nx, ny = nb['xy']
        for px, py in ego_xy:
            d = math.hypot(nx - px, ny - py)
            best = min(best, d)
    return best


def _route_max_deviation(ego_xy: np.ndarray, route_xy: list[tuple] | None) -> float:
    if not route_xy:
        return 0.0
    return float(max(
        min(math.hypot(rx - px, ry - py) for rx, ry in route_xy)
        for px, py in ego_xy
    ))


# ---------------------------------------------------------------------------
# Violation construction
# ---------------------------------------------------------------------------

def _make_violation(name: str, threshold: float, measured: float, *,
                    explanation: str, layer: str) -> dict:
    """Build a per-criterion violation record."""
    if not math.isfinite(threshold):
        # criterion disabled in this config (e.g. infinity)
        return {'name': name, 'layer': layer, 'threshold': None,
                'measured': float(measured), 'severity': None, 'triggered': False,
                'explanation': explanation + ' (criterion disabled in this config)'}
    severity = float(measured) / float(threshold) if threshold > 0 else float(measured)
    return {'name': name, 'layer': layer, 'threshold': float(threshold),
            'measured': float(measured), 'severity': float(severity),
            'triggered': bool(measured > threshold), 'explanation': explanation}


def _scenario_bucket(ego_v0: float | None, lat_acc_max: float,
                     neighbors_within_range: bool, route_xy_provided: bool,
                     cfg: CCProxyConfig) -> str:
    if ego_v0 is not None and ego_v0 < cfg.scenario_cold_start_v0_max_mps:
        return 'cold_start'
    if neighbors_within_range:
        return 'object_interaction'
    if lat_acc_max >= cfg.scenario_curve_lat_acc_min_mps2:
        return 'curve'
    if route_xy_provided:
        return 'route_following'
    return 'straight'


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

def score_trajectory_v3(
    ego_xy: list[tuple],
    ego_v_provided: list[float] | None = None,
    route_xy: list[tuple] | None = None,
    neighbors: list[dict] | None = None,
    ego_v0: float | None = None,
    dt: float = 0.1,
    cfg: CCProxyConfig | None = None,
) -> dict:
    """Score a planned trajectory against the JAMA-inspired Preventability PROXY.

    Args:
        ego_xy: list of (x, y, heading, t).
        ego_v_provided: optional planner-output speed sequence (else derived
                        from SG-filtered xy positions).
        route_xy: optional planned route centerline.
        neighbors: optional list of {'xy': (x,y), 'v': speed} at t=0.
        ego_v0: ego speed at t=0 (for cold-start / stuck checks).
        dt: time step (default 0.1).
        cfg: CCProxyConfig (default = factory defaults).

    Returns:
        {
          'strict_safety':       {'any_triggered': bool, 'violations': [...]},
          'c_and_c_driverlike':  {'any_triggered': bool, 'violations': [...]},
          'comfort':             {'any_triggered': bool, 'violations': [...]},
          'scenario_bucket':     str,
          'details':             {...},  # raw measured signals
          'cfg':                 asdict(cfg),
          'mode':                'jama_inspired_proxy_v3',
        }
    """
    if cfg is None:
        cfg = CCProxyConfig()

    xy = np.array([(p[0], p[1]) for p in ego_xy], dtype=np.float64)
    headings = np.array([p[2] if len(p) > 2 else 0.0 for p in ego_xy])

    # SG-filtered kinematics
    v_xy, a_xy, j_xy = _sg_derivatives(xy, dt=dt, window=11, polyorder=3)
    speed = np.linalg.norm(v_xy, axis=1)
    long_acc_signed = _signed_long_acc(v_xy, a_xy)
    lat_acc = _lat_acc_curvature(v_xy, a_xy)
    long_jerk_mag = np.linalg.norm(j_xy, axis=1)
    heading_err = _heading_vs_velocity(headings, v_xy)

    if ego_v_provided is not None and len(ego_v_provided) == len(ego_xy):
        max_speed = float(max(ego_v_provided))
    else:
        max_speed = float(speed.max())

    min_neighbor_dist = _min_neighbor_distance(xy, neighbors)
    ttc = _min_ttc((xy[0, 0], xy[0, 1]), float(headings[0]),
                   float(speed[0]) if len(speed) else 0.0, neighbors)
    max_route_dev = _route_max_deviation(xy, route_xy)
    max_decel = float(-long_acc_signed.min()) if len(long_acc_signed) else 0.0
    max_accel = float(long_acc_signed.max()) if len(long_acc_signed) else 0.0
    max_lat_acc = float(lat_acc.max()) if len(lat_acc) else 0.0
    max_jerk = float(long_jerk_mag.max()) if len(long_jerk_mag) else 0.0
    max_heading_err = float(heading_err.max()) if len(heading_err) else 0.0

    # ===== Layer 1: strict_safety =====
    safety_violations = [
        _make_violation('collision_proxy_min_distance',
                        cfg.collision_min_distance_m, -min_neighbor_dist if math.isfinite(min_neighbor_dist) else -math.inf,
                        layer='strict_safety',
                        explanation=f'Min trajectory-to-neighbor distance is {min_neighbor_dist:.2f} m (threshold {cfg.collision_min_distance_m} m). Inverted sign: triggers when distance < threshold.'),
        _make_violation('ttc_below_min',
                        -cfg.ttc_min_sec, -ttc if math.isfinite(ttc) else -math.inf,
                        layer='strict_safety',
                        explanation=(f'Min TTC to a leading neighbor is {ttc:.2f} s' if math.isfinite(ttc) else 'No leading neighbor in TTC range')
                                    + f' (threshold {cfg.ttc_min_sec} s, JAMA Fig 108).'),
        _make_violation('route_corridor_departure',
                        cfg.route_corridor_max_m, max_route_dev,
                        layer='strict_safety',
                        explanation=f'Max lateral deviation from planned route is {max_route_dev:.2f} m (hard corridor {cfg.route_corridor_max_m} m).'),
        _make_violation('extreme_decel',
                        cfg.extreme_decel_mps2, max_decel,
                        layer='strict_safety',
                        explanation=f'Peak longitudinal deceleration {max_decel:.2f} m/s² (hard cap {cfg.extreme_decel_mps2} m/s² ≈ 1 G).'),
        _make_violation('extreme_lat_acc',
                        cfg.extreme_lat_acc_mps2, max_lat_acc,
                        layer='strict_safety',
                        explanation=f'Peak lateral acceleration {max_lat_acc:.2f} m/s² (hard cap {cfg.extreme_lat_acc_mps2} m/s² ≈ 0.6 G).'),
    ]

    # Fix sign convention: collision_proxy and ttc use negative values for "triggered when below threshold"
    # — simplest is to handle explicitly:
    safety_violations[0]['triggered'] = (min_neighbor_dist < cfg.collision_min_distance_m)
    safety_violations[0]['measured'] = float(min_neighbor_dist) if math.isfinite(min_neighbor_dist) else None
    safety_violations[0]['threshold'] = cfg.collision_min_distance_m
    safety_violations[0]['severity'] = (cfg.collision_min_distance_m / min_neighbor_dist) if min_neighbor_dist > 0 else None
    safety_violations[1]['triggered'] = (ttc < cfg.ttc_min_sec) if math.isfinite(ttc) else False
    safety_violations[1]['measured'] = (float(ttc) if math.isfinite(ttc) else None)
    safety_violations[1]['threshold'] = cfg.ttc_min_sec
    safety_violations[1]['severity'] = (cfg.ttc_min_sec / ttc) if math.isfinite(ttc) and ttc > 0 else None

    # ===== Layer 2: c_and_c_driverlike =====
    # Cold-start check
    cs_speed = 0.0
    cold_start_triggered = False
    if ego_v0 is not None and ego_v0 <= cfg.cold_start_v0_thresh_mps:
        n = min(cfg.cold_start_window_n, len(speed))
        if n > 0:
            cs_speed = float(speed[:n].max())
            cold_start_triggered = (cs_speed > cfg.cold_start_max_predicted_speed_mps)
    # Stuck check
    stuck_triggered = False
    stuck_progress = 0.0
    stuck_peak_v = 0.0
    if ego_v0 is not None and ego_v0 > cfg.stuck_max_speed_mps:
        n_h = min(int(cfg.stuck_check_horizon_sec / dt), len(xy))
        if n_h >= 2:
            stuck_progress = math.hypot(xy[n_h-1, 0] - xy[0, 0], xy[n_h-1, 1] - xy[0, 1])
            stuck_peak_v = float(speed[:n_h].max())
            stuck_triggered = (stuck_progress < cfg.stuck_progress_max_m and stuck_peak_v < cfg.stuck_max_speed_mps)

    driverlike_violations = [
        _make_violation('cold_start_accel_burst',
                        cfg.cold_start_max_predicted_speed_mps, cs_speed,
                        layer='c_and_c_driverlike',
                        explanation=f'Ego stationary at t=0 (v0≈{ego_v0 if ego_v0 is not None else 0:.2f}); '
                                    f'planner predicts {cs_speed:.2f} m/s within first {cfg.cold_start_window_n}×{dt}s.'),
        _make_violation('driverlike_lat_acc',
                        cfg.driverlike_lat_acc_max_mps2, max_lat_acc,
                        layer='c_and_c_driverlike',
                        explanation=f'Max curve-induced lat acc {max_lat_acc:.2f} m/s² (C&C-driver-like cap {cfg.driverlike_lat_acc_max_mps2} m/s²).'),
        _make_violation('heading_velocity_inconsistency',
                        cfg.heading_velocity_max_rad, max_heading_err,
                        layer='c_and_c_driverlike',
                        explanation=f'Max predicted-heading vs velocity-direction deviation {math.degrees(max_heading_err):.1f}° '
                                    f'(C&C consistency cap {math.degrees(cfg.heading_velocity_max_rad):.0f}°).'),
        _make_violation('driverlike_decel',
                        cfg.driverlike_decel_max_mps2, max_decel,
                        layer='c_and_c_driverlike',
                        explanation=f'Peak deceleration {max_decel:.2f} m/s² (C&C-driver-like cap {cfg.driverlike_decel_max_mps2} m/s² = JAMA Fig 94).'),
        _make_violation('driverlike_accel',
                        cfg.driverlike_accel_max_mps2, max_accel,
                        layer='c_and_c_driverlike',
                        explanation=f'Peak acceleration {max_accel:.2f} m/s² (C&C-driver-like cap {cfg.driverlike_accel_max_mps2} m/s²).'),
    ]
    # cold_start has special trigger logic
    driverlike_violations[0]['triggered'] = bool(cold_start_triggered)
    # Stuck check appended separately
    stuck_violation = {
        'name': 'planner_stuck',
        'layer': 'c_and_c_driverlike',
        'threshold': cfg.stuck_progress_max_m,
        'measured': float(stuck_progress),
        'severity': (cfg.stuck_progress_max_m / max(stuck_progress, 0.01)) if stuck_triggered else None,
        'triggered': bool(stuck_triggered),
        'explanation': f'Ego was moving (v0={ego_v0 if ego_v0 is not None else 0:.2f} m/s) but planner predicts '
                       f'progress {stuck_progress:.2f} m and peak speed {stuck_peak_v:.2f} m/s over '
                       f'first {cfg.stuck_check_horizon_sec}s; below {cfg.stuck_progress_max_m} m + '
                       f'{cfg.stuck_max_speed_mps} m/s thresholds indicates planner-stuck failure.'
    }
    driverlike_violations.append(stuck_violation)

    # ===== Layer 3: comfort =====
    comfort_violations = [
        _make_violation('jerk_comfort',
                        cfg.jerk_comfort_max_mps3, max_jerk,
                        layer='comfort',
                        explanation=f'Peak longitudinal jerk {max_jerk:.2f} m/s³ (ISO 2631 comfort cap {cfg.jerk_comfort_max_mps3}).'),
        _make_violation('lat_acc_comfort',
                        cfg.lat_acc_comfort_max_mps2, max_lat_acc,
                        layer='comfort',
                        explanation=f'Peak lat acc {max_lat_acc:.2f} m/s² exceeds passenger-comfort cap {cfg.lat_acc_comfort_max_mps2} (still safe).'),
        _make_violation('speed_change_comfort',
                        cfg.speed_change_comfort_max_mps2, max(max_accel, max_decel),
                        layer='comfort',
                        explanation=f'Peak speed-change magnitude {max(max_accel, max_decel):.2f} m/s² exceeds comfort cap {cfg.speed_change_comfort_max_mps2}.'),
    ]

    # Scenario bucket
    nbr_in_range = False
    if neighbors:
        ego_x, ego_y = xy[0, 0], xy[0, 1]
        nbr_in_range = any(math.hypot(nb['xy'][0] - ego_x, nb['xy'][1] - ego_y)
                           < cfg.scenario_object_interaction_neighbor_range_m
                           for nb in neighbors)
    bucket = _scenario_bucket(ego_v0, max_lat_acc, nbr_in_range, route_xy is not None, cfg)

    return {
        'strict_safety': {
            'any_triggered': any(v['triggered'] for v in safety_violations),
            'violations': safety_violations,
        },
        'c_and_c_driverlike': {
            'any_triggered': any(v['triggered'] for v in driverlike_violations),
            'violations': driverlike_violations,
        },
        'comfort': {
            'any_triggered': any(v['triggered'] for v in comfort_violations),
            'violations': comfort_violations,
        },
        'scenario_bucket': bucket,
        'details': {
            'max_speed_mps':       round(max_speed, 3),
            'max_accel_mps2':      round(max_accel, 3),
            'max_decel_mps2':      round(max_decel, 3),
            'max_lat_acc_mps2':    round(max_lat_acc, 3),
            'max_jerk_mps3':       round(max_jerk, 3),
            'max_heading_err_rad': round(max_heading_err, 4),
            'max_heading_err_deg': round(math.degrees(max_heading_err), 2),
            'max_route_dev_m':     round(max_route_dev, 3),
            'min_ttc_sec':         (round(float(ttc), 3) if math.isfinite(ttc) else None),
            'min_nbr_dist_m':      (round(float(min_neighbor_dist), 3) if math.isfinite(min_neighbor_dist) else None),
            'cold_start_peak_v_mps': round(cs_speed, 3),
            'stuck_progress_m':    round(stuck_progress, 3),
            'stuck_peak_v_mps':    round(stuck_peak_v, 3),
        },
        'mode': 'jama_inspired_proxy_v3',
        'cfg': asdict(cfg),
    }


def aggregate_bucket_results(per_trajectory_results: list[dict]) -> dict:
    """Aggregate a list of per-trajectory score_trajectory_v3 outputs into a
    per-scenario-bucket summary table.

    Returns:
      {bucket_name: {
         'n': int,
         'strict_safety_violation_rate': float,
         'driverlike_violation_rate': float,
         'comfort_violation_rate': float,
         'per_criterion_rate': {criterion_name: rate_of_trigger},
      }, ...,
       '_overall': {...same shape, across all trajectories...},
      }
    """
    out = {}
    overall_safety = 0
    overall_driverlike = 0
    overall_comfort = 0
    overall_n = len(per_trajectory_results)
    per_crit_all = {}
    bucket_pool: dict[str, list[dict]] = {}
    for r in per_trajectory_results:
        b = r['scenario_bucket']
        bucket_pool.setdefault(b, []).append(r)
        if r['strict_safety']['any_triggered']:    overall_safety += 1
        if r['c_and_c_driverlike']['any_triggered']: overall_driverlike += 1
        if r['comfort']['any_triggered']:           overall_comfort += 1
        for layer in ('strict_safety', 'c_and_c_driverlike', 'comfort'):
            for v in r[layer]['violations']:
                per_crit_all.setdefault(v['name'], 0)
                if v['triggered']:
                    per_crit_all[v['name']] += 1

    out['_overall'] = {
        'n': overall_n,
        'strict_safety_violation_rate':   overall_safety / overall_n if overall_n else 0.0,
        'driverlike_violation_rate':      overall_driverlike / overall_n if overall_n else 0.0,
        'comfort_violation_rate':         overall_comfort / overall_n if overall_n else 0.0,
        'per_criterion_rate':             {k: v / overall_n if overall_n else 0.0 for k, v in per_crit_all.items()},
    }
    for b, rs in bucket_pool.items():
        n_b = len(rs)
        per_crit = {}
        n_safety = sum(1 for r in rs if r['strict_safety']['any_triggered'])
        n_drv = sum(1 for r in rs if r['c_and_c_driverlike']['any_triggered'])
        n_cmf = sum(1 for r in rs if r['comfort']['any_triggered'])
        for r in rs:
            for layer in ('strict_safety', 'c_and_c_driverlike', 'comfort'):
                for v in r[layer]['violations']:
                    per_crit.setdefault(v['name'], 0)
                    if v['triggered']: per_crit[v['name']] += 1
        out[b] = {
            'n': n_b,
            'strict_safety_violation_rate': n_safety / n_b,
            'driverlike_violation_rate':    n_drv / n_b,
            'comfort_violation_rate':       n_cmf / n_b,
            'per_criterion_rate':           {k: v / n_b for k, v in per_crit.items()},
        }
    return out
