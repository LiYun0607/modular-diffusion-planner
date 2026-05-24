"""JAMA-inspired C&C-style preventability / safety-violation scorer.

NOT a re-implementation of JAMA Ver.4.0 §6 — this is an *inspired* operationalization:
we treat a trajectory as a C&C-violation if it crosses any of a small set of
heuristic comfort/safety thresholds that a careful and competent human driver
would not produce. The intent is to give us a deployment-safety metric that is
INDEPENDENT of the training reward function (which has been shown to over-reward
unsafe behaviors like v8 accelerating through turns despite high reward).

Usage:
  from cc_violation_score import score_trajectory, CCConfig

  cfg = CCConfig()
  result = score_trajectory(
      ego_xy=[(x0,y0,h0,t0), (x1,y1,h1,t1), ...],   # planned trajectory waypoints
      ego_v=[v0, v1, ...],                            # ego speed sequence
      route_xy=[(rx0,ry0), ...],                      # planned route centerline (optional)
      neighbors=[{'xy': (x,y), 'v': v}, ...],         # tracked neighbors at t=0 (optional)
  )
  # result['violation'] -> bool (any threshold crossed)
  # result['violations'] -> dict of {criterion: bool}
  # result['details']    -> dict of {criterion: scalar_violating_value}

Module-level entry points:
  score_trajectory(...) -> dict
  score_batch(list of trajs) -> aggregate dict with violation_rate per criterion
  load_trajectories_jsonl(path) -> iterator over traj dicts

Convention (matches Diffusion-Planner):
  ego_xy: [(x, y, heading_rad, t_sec)] for the planner's 80-step output
  ego_v:  [v_m_per_s] same length as ego_xy
  Coordinates assumed in vehicle-fixed frame (ego start at origin) OR map frame.
  If in map frame, route_xy must be in the same frame.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Iterator


@dataclass
class CCConfig:
    """Thresholds calibrated against JAMA Ver.4.0 §6 comfort/safety guidance
    and common ADS evaluation literature. All values are PER-FRAME maxima
    over the planning horizon."""

    # Lateral comfort: a C&C driver rarely exceeds 3 m/s^2 lat acc in normal driving;
    # JAMA Annex C / RSS use 4 m/s^2 as outer comfort. We use 3.0 as default.
    lat_acc_max_mps2: float = 3.0

    # Longitudinal jerk: comfort threshold ~5 m/s^3 widely cited (ISO 2631, JAMA).
    jerk_max_mps3: float = 5.0

    # Speed limit relative to route speed: planner should not exceed 1.25x of cmd
    # (we use absolute cap when route speed unknown).
    speed_max_mps: float = 18.0  # ~65 km/h, urban kashiwa upper

    # Route deviation: lateral distance from planned route centerline.
    # JAMA §6 careful driver stays within ~1 m of intended path in normal flow.
    route_dev_max_m: float = 1.5

    # Heading error vs route tangent at the closest route point.
    heading_err_max_rad: float = math.radians(30.0)

    # Time-to-collision with nearest neighbor (only counted if neighbor is ahead).
    ttc_min_sec: float = 2.0
    ttc_min_distance_for_check_m: float = 1.0  # ignore if neighbor literally on top

    # v=0 cold-start fault: if ego is stationary AND first 8 steps predict
    # >5 m/s velocity, that's a cold-start drift (our paper's §6.1 finding).
    cold_start_v0_speed_thresh_mps: float = 0.5     # ego must be ≤ this
    cold_start_predicted_speed_mps: float = 5.0     # but predicts >this
    cold_start_within_first_n_steps: int = 8

    # Backwards motion: planner predicts negative longitudinal motion
    # (illegal for forward driving scenario).
    allow_backwards: bool = False

    # Stop-required scenario flag: if goal is within stop_required_distance_m
    # and predicted terminal speed > stop_required_terminal_speed_mps, fail.
    stop_required_distance_m: float = 5.0
    stop_required_terminal_speed_mps: float = 1.0

    # Stuck-failure: planner predicts (near-)zero forward progress over horizon.
    # Distinct from intentional stop (handled by stop_required) — this catches
    # the "vehicle physically can't move" deployment failure (e.g., R1_rep2_selora,
    # where the vehicle creeps at 0.23 m/s for the entire test). C&C-driver
    # would not silently fail to make progress when path is clear.
    # NOTE: only meaningful if combined with "intent to move" — when called on
    # the real ego trajectory, set check_stuck=False if ego is at an
    # intersection-stop or in heavy traffic. Default ON for the synthetic /
    # garage / clear-path case.
    stuck_progress_max_m: float = 2.0  # horizon-end displacement from start
    stuck_max_speed_mps: float = 0.5    # AND max speed never exceeded this
    stuck_check_horizon_sec: float = 4.0  # check the first stuck_check_horizon_sec only


def _diff(a: list[float]) -> list[float]:
    return [a[i+1] - a[i] for i in range(len(a)-1)]


def _lateral_acc(ego_xy: list[tuple], ego_v: list[float]) -> list[float]:
    """Approx lat_acc = v^2 * curvature.
    curvature estimated from finite-difference of heading."""
    if len(ego_xy) < 3:
        return [0.0] * len(ego_xy)
    out = [0.0]
    for i in range(1, len(ego_xy) - 1):
        # heading change between adjacent segments
        h0 = ego_xy[i-1][2] if len(ego_xy[i-1]) > 2 else 0.0
        h1 = ego_xy[i+1][2] if len(ego_xy[i+1]) > 2 else 0.0
        dh = math.atan2(math.sin(h1-h0), math.cos(h1-h0))
        # arc-length over the two segments
        dx = ego_xy[i+1][0] - ego_xy[i-1][0]
        dy = ego_xy[i+1][1] - ego_xy[i-1][1]
        ds = math.hypot(dx, dy) + 1e-9
        kappa = dh / ds
        out.append((ego_v[i] ** 2) * abs(kappa))
    out.append(out[-1])
    return out


def _longitudinal_jerk(ego_v: list[float], dt: float = 0.1) -> list[float]:
    a = _diff(ego_v)
    j = _diff(a)
    return [abs(x) / (dt * dt) for x in j]


def _route_deviation(ego_xy: list[tuple], route_xy: list[tuple]) -> tuple[list[float], list[float]]:
    """Per-step (lateral_dev_m, heading_err_rad) wrt nearest route point."""
    dev, hdg = [], []
    if not route_xy:
        return [0.0] * len(ego_xy), [0.0] * len(ego_xy)
    for px, py, *rest in ego_xy:
        # nearest route point + local tangent
        d2 = [(rx - px) ** 2 + (ry - py) ** 2 for rx, ry in route_xy]
        j = min(range(len(d2)), key=lambda i: d2[i])
        d = math.sqrt(d2[j])
        # tangent at j
        if j + 1 < len(route_xy):
            tx = route_xy[j+1][0] - route_xy[j][0]
            ty = route_xy[j+1][1] - route_xy[j][1]
        else:
            tx = route_xy[j][0] - route_xy[j-1][0]
            ty = route_xy[j][1] - route_xy[j-1][1]
        route_heading = math.atan2(ty, tx)
        ego_heading = rest[0] if rest else math.atan2(ty, tx)
        herr = math.atan2(math.sin(ego_heading - route_heading), math.cos(ego_heading - route_heading))
        dev.append(d)
        hdg.append(abs(herr))
    return dev, hdg


def _min_ttc(ego_xy: list[tuple], ego_v: list[float], neighbors: list[dict] | None) -> float:
    """Simplest TTC: for each neighbor, if relative position ahead in ego heading
    and ego closing speed > 0, compute TTC = distance / closing_speed."""
    if not neighbors:
        return math.inf
    px, py, *rest = ego_xy[0]
    eh = rest[0] if rest else 0.0
    ev = ego_v[0]
    best = math.inf
    for nb in neighbors:
        nx, ny = nb['xy']
        nv = nb.get('v', 0.0)
        dx, dy = nx - px, ny - py
        dist = math.hypot(dx, dy)
        ahead_proj = dx * math.cos(eh) + dy * math.sin(eh)
        if ahead_proj <= 0:
            continue
        closing = ev - nv
        if closing <= 0:
            continue
        ttc = ahead_proj / closing
        best = min(best, ttc)
    return best


def score_trajectory(
    ego_xy: list[tuple],
    ego_v: list[float],
    route_xy: list[tuple] | None = None,
    neighbors: list[dict] | None = None,
    ego_v0: float | None = None,
    goal_xy: tuple | None = None,
    dt: float = 0.1,
    cfg: CCConfig | None = None,
) -> dict:
    """Score one planned trajectory against C&C-style preventability checks.

    Returns:
        {
          'violation': bool,
          'violations': {criterion: bool},
          'details': {criterion: float (the offending value)},
          'config': asdict(cfg),
        }
    """
    if cfg is None:
        cfg = CCConfig()
    assert len(ego_xy) == len(ego_v), "ego_xy and ego_v must align"

    lat_acc = _lateral_acc(ego_xy, ego_v)
    jerk = _longitudinal_jerk(ego_v, dt=dt)
    dev, hdg = _route_deviation(ego_xy, route_xy or [])
    ttc = _min_ttc(ego_xy, ego_v, neighbors)

    # backwards motion
    backwards = False
    if not cfg.allow_backwards:
        for i in range(1, len(ego_xy)):
            dx = ego_xy[i][0] - ego_xy[i-1][0]
            dy = ego_xy[i][1] - ego_xy[i-1][1]
            h = ego_xy[i-1][2] if len(ego_xy[i-1]) > 2 else 0.0
            forward_proj = dx * math.cos(h) + dy * math.sin(h)
            if forward_proj < -0.2:
                backwards = True
                break

    # cold-start drift: ego currently ~stationary AND predicted speed surges
    cold_start = False
    if ego_v0 is not None and ego_v0 <= cfg.cold_start_v0_speed_thresh_mps:
        n = min(cfg.cold_start_within_first_n_steps, len(ego_v))
        peak_early = max(ego_v[:n]) if n > 0 else 0.0
        if peak_early > cfg.cold_start_predicted_speed_mps:
            cold_start = True

    # stop-required miss
    stop_required_miss = False
    if goal_xy is not None:
        gx, gy = goal_xy
        px0, py0 = ego_xy[0][0], ego_xy[0][1]
        dist_to_goal = math.hypot(gx - px0, gy - py0)
        if dist_to_goal <= cfg.stop_required_distance_m:
            if ego_v[-1] > cfg.stop_required_terminal_speed_mps:
                stop_required_miss = True

    # stuck-failure: low progress AND low max speed across check horizon.
    # Catches the "vehicle physically can't move" deployment failure (rep2 case).
    # GUARDED: only fires when ego was *already moving* (ego_v0 above threshold) —
    # otherwise legitimate intersection stops (real ego at v=0 waiting for signal)
    # would all be flagged. Cannot detect rep2-style failures per-frame from this
    # check alone; use bag_level_progress_summary() for fleet-level detection.
    stuck = False
    if ego_v0 is not None and ego_v0 > cfg.stuck_max_speed_mps:
        n_check = min(int(cfg.stuck_check_horizon_sec / dt), len(ego_xy))
        if n_check >= 2:
            x0, y0 = ego_xy[0][0], ego_xy[0][1]
            x_end, y_end = ego_xy[n_check - 1][0], ego_xy[n_check - 1][1]
            progress = math.hypot(x_end - x0, y_end - y0)
            peak_v = max(ego_v[:n_check])
            if progress < cfg.stuck_progress_max_m and peak_v < cfg.stuck_max_speed_mps:
                stuck = True

    max_lat = max(lat_acc) if lat_acc else 0.0
    max_jerk = max(jerk) if jerk else 0.0
    max_speed = max(ego_v) if ego_v else 0.0
    max_dev = max(dev) if dev else 0.0
    max_hdg = max(hdg) if hdg else 0.0

    violations = {
        'lat_acc':       max_lat > cfg.lat_acc_max_mps2,
        'jerk':          max_jerk > cfg.jerk_max_mps3,
        'speed_cap':     max_speed > cfg.speed_max_mps,
        'route_dev':     max_dev > cfg.route_dev_max_m,
        'heading_err':   max_hdg > cfg.heading_err_max_rad,
        'ttc':           ttc < cfg.ttc_min_sec,
        'cold_start':    cold_start,
        'backwards':     backwards,
        'stop_required': stop_required_miss,
        'stuck':         stuck,
    }
    details = {
        'lat_acc_max':       round(max_lat, 3),
        'jerk_max':          round(max_jerk, 3),
        'speed_max':         round(max_speed, 3),
        'route_dev_max':     round(max_dev, 3),
        'heading_err_max':   round(max_hdg, 3),
        'min_ttc':           (round(ttc, 3) if math.isfinite(ttc) else None),
    }
    return {
        'violation': any(violations.values()),
        'violations': violations,
        'details': details,
        'config': asdict(cfg),
    }


def score_batch(trajectories: Iterable[dict], cfg: CCConfig | None = None) -> dict:
    """Aggregate per-criterion violation rate across many trajectories.

    Each trajectory should be a dict with keys matching score_trajectory args:
    ego_xy, ego_v, optional route_xy, neighbors, ego_v0, goal_xy.
    """
    if cfg is None:
        cfg = CCConfig()
    per_criterion = {}
    n = 0
    n_any_violation = 0
    detail_sums = {}
    for traj in trajectories:
        r = score_trajectory(cfg=cfg, **traj)
        n += 1
        if r['violation']:
            n_any_violation += 1
        for k, v in r['violations'].items():
            per_criterion.setdefault(k, 0)
            if v:
                per_criterion[k] += 1
        for k, v in r['details'].items():
            if v is None:
                continue
            detail_sums.setdefault(k, []).append(v)
    out = {
        'n_trajectories': n,
        'any_violation_rate': (n_any_violation / n) if n else 0.0,
        'per_criterion_rate': {k: (v / n if n else 0.0) for k, v in per_criterion.items()},
        'detail_mean': {k: (sum(vs) / len(vs) if vs else None) for k, vs in detail_sums.items()},
        'detail_p95': {
            k: (sorted(vs)[int(len(vs) * 0.95)] if vs else None) for k, vs in detail_sums.items()
        },
        'config': asdict(cfg),
    }
    return out


def bag_level_progress_summary(records: list[dict]) -> dict:
    """Aggregate ego-trajectory records (from extract_real_ego_trajectories.py
    output) into a bag-level deployment-progress signature.

    Catches the rep2-style failure where every per-frame metric is "safe" but
    the bag-level behavior is "vehicle never moved meaningfully".

    Each record should have keys: ego_v_now, ego_xy_now_map (x, y, yaw).

    Returns:
      n_frames, mean_ego_v, max_ego_v, fraction_stationary (v<0.5),
      xy_span_m, deployment_progress_failure (bool: span<10m AND mean_v<0.5
      over ≥30s of recording)
    """
    if not records:
        return {'n_frames': 0, 'deployment_progress_failure': False}
    speeds = [r.get('ego_v_now', 0.0) for r in records]
    xs = [r['ego_xy_now_map'][0] for r in records if 'ego_xy_now_map' in r]
    ys = [r['ego_xy_now_map'][1] for r in records if 'ego_xy_now_map' in r]
    n = len(records)
    mean_v = sum(speeds) / n
    max_v = max(speeds)
    frac_stat = sum(1 for v in speeds if v < 0.5) / n
    span = (math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            if xs and len(xs) >= 2 else 0.0)
    # Deployment-progress failure: vehicle effectively didn't drive.
    # Heuristic: ≥95% of frames stationary AND mean speed < 0.5 m/s,
    # AND duration >= 30s (n >= 300 at 10 Hz to avoid false flag on short tests).
    # Tolerates slow drift (rep2 had 72m span but 100% stationary at 0.23 m/s).
    deployment_failure = (frac_stat >= 0.95 and mean_v < 0.5 and n >= 300)
    return {
        'n_frames': n,
        'mean_ego_v_mps': round(mean_v, 3),
        'max_ego_v_mps': round(max_v, 3),
        'fraction_stationary': round(frac_stat, 3),
        'xy_span_m': round(span, 2),
        'deployment_progress_failure': bool(deployment_failure),
    }


def load_trajectories_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _self_test():
    """Smoke test with synthetic trajectories. Run: python cc_violation_score.py"""
    # Synthetic: cold-start drift trajectory (ego at v=0, planner predicts 80 km/h cruise)
    n = 20
    dt = 0.1
    ego_xy = [(i * 2.2 * dt, 0.0, 0.0, i * dt) for i in range(n)]  # 22 m/s = 80 km/h
    ego_v = [22.0] * n
    r = score_trajectory(ego_xy, ego_v, ego_v0=0.0)
    print("\n=== test 1: cold-start drift (should violate cold_start + speed_cap)")
    print(json.dumps(r['violations'], indent=2))
    print(json.dumps(r['details'], indent=2))

    # Synthetic: clean slow drive
    ego_xy = [(i * 0.5 * dt, 0.0, 0.0, i * dt) for i in range(n)]
    ego_v = [5.0] * n
    r = score_trajectory(ego_xy, ego_v, ego_v0=5.0)
    print("\n=== test 2: clean slow drive (should violate nothing)")
    print(json.dumps(r['violations'], indent=2))

    # Synthetic: v8 accelerate-through-turn (curving + accelerating)
    ego_xy = []
    ego_v = []
    R = 10.0  # 10m radius turn
    for i in range(n):
        theta = i * 0.1
        v = 4.0 + 0.5 * i  # accelerating
        ego_xy.append((R * math.sin(theta), R * (1 - math.cos(theta)), theta, i * dt))
        ego_v.append(v)
    r = score_trajectory(ego_xy, ego_v, ego_v0=4.0)
    print("\n=== test 3: v8-style accelerate-through-turn (should violate lat_acc + jerk)")
    print(json.dumps(r['violations'], indent=2))
    print(json.dumps(r['details'], indent=2))


if __name__ == '__main__':
    _self_test()
