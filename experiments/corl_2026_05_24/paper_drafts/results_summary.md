# CoRL §6 Results Summary (concrete numbers from 2026-05-21 + 2026-05-01 real-vehicle data)

> All numbers below are extracted from real-vehicle ROS bags collected at
> kashiwa test site. Source paths are noted for reproducibility.

## R.1 4-Model C&C Violation Rates on Garage v=0 (2026-05-21)

**Source.** `/home/yunli/Pixkit/field_data/2026-05-21/offline_ab_4model/4model.parquet`
(11,859 frames across 4 models, ~3000 frames each). Models replayed offline
against the recorded `garage_v0_180856` LiDAR + perception + chassis bag
(40 min, 147 GB) using `pixkit/offline_ab_replay.sh`.

**Trajectory extraction.** From each model's output bag
(`offline_ab_4model/stage{1,2}/garage_v0_180856/{baseline,kashiwa_selora}_traj`)
we read the `/planning/diffusion_planner/trajectory` topic (3,021 messages,
80 waypoints each), parsed `pose.position.{x,y}`,
`longitudinal_velocity_mps`, `acceleration_mps2`, applied `cc_violation_score.py`.

**Result.**

| Model          | n_frames | any_violation | jerk>5 | lat_acc>3 | route_dev>1.5 | cold_start | p95 jerk | p95 speed |
|----------------|----------|---------------|--------|-----------|---------------|------------|----------|-----------|
| base_nolora    | 3021     | 16.45%        | 16.45% | 1.22%     | 0.00%         | 0.00%      | 6.902    | 1.962     |
| joint_replay   | 2851     | 17.54%        | 17.54% | 1.16%     | 0.00%         | 0.00%      | 7.251    | 2.361     |
| kashiwa_alone  | 3010     | 17.44%        | 17.44% | 1.13%     | 0.00%         | 0.00%      | 7.098    | 2.296     |
| shared_only    | 2977     | 17.13%        | 17.13% | 1.24%     | 0.00%         | 0.00%      | 7.237    | 2.367     |

**Interpretation.**
- All four models violate the C&C jerk threshold (>5 m/s³) ~17% of frames.
  This is universal — not LoRA-specific — and indicates the diffusion
  planner's per-waypoint output is jerkier than the comfort threshold; in
  deployment a downstream trajectory_optimizer smooths this.
- LoRA variants (joint_replay, kashiwa_alone, shared_only) all show ~1
  percentage point higher jerk violation rate than base_nolora.
- Cold-start violation rate is 0% across all models — in genuine static
  garage v=0 the trajectories don't predict fast cruise. Cold-start
  manifests in dynamic OR low-v transitions, not pure stationary.
- Route deviation is well-bounded (max 0.65m vs 1.5m threshold) for all
  models in garage.

**Honest caveat.** This is a "boring" result for the §6 narrative because no
single model stands out as catastrophically worse. The catastrophic failure
is visible in §6.2 (2026-05-01 dynamic driving), not garage v=0.

## R.2 Trajectory L2 Divergence from Baseline (2026-05-21)

| Model          | n      | mean Δ (m) | p95 Δ (m) | p99 Δ (m) | max Δ (m) |
|----------------|--------|------------|-----------|-----------|-----------|
| base_nolora→self | 2977 | 0.398      | 2.425     | ~3.5      | 7.563     |
| kashiwa_alone  | 3010   | 0.431      | 2.606     | ~3.7      | **15.775** |

The 15.775m max divergence (kashiwa_alone vs base on the SAME stationary
ego input) indicates that at some frames the LoRA delta sends the predicted
trajectory radically off. This is the v=0-onset cold-start failure pattern
captured in a static scene — it occurs in 0.7% of frames (21/3010).

## R.3 Sim vs Real Ego Velocity Distribution (2026-05-01)

**Source.** `/home/yunli/Pixkit/field_data/2026-05-01/analysis/sim_vs_real/`

| Model                | Sim μ ± σ (m/s) | Real μ ± σ (m/s) | KS test D | p-value |
|----------------------|-----------------|------------------|-----------|---------|
| base_nolora (BASELINE) | 4.11 ± 3.19   | 1.25 ± 1.78      | 0.435     | 0.0     |
| SE-LoRA kashiwa      | 4.10 ± 3.19     | 3.48 ± 2.44 (open-loop) | 0.234 | 1.6e-218 |

**Critical observation.** In sim closed-loop, both models cruise at 4 m/s
on average. In real closed-loop, baseline only reached 1.25 m/s on average
(driver-induced stop-and-go); SE-LoRA `kashiwa_selora` rep1 reached 3.48 m/s
open-loop predicted speed but the real-vehicle did not reach this; SE-LoRA
rep2 failed to move at all beyond 70 m.

## R.4 SE-LoRA Catastrophic Real-Vehicle Failure (2026-05-01 R1 rep2)

**Source.** `/home/yunli/Pixkit/field_data/2026-05-01/analysis/fig1_xy_path.png`
and `fig2_speed_vs_time.png`.

R1_rep2_kashiwa_selora data trace:
- Total distance: ~70 m before halt
- Speed profile: constant ~0.22 m/s (idle creep / brake-not-fully-released)
  for the entire ~310 s test, then full halt
- XY trajectory: drove west ~130 m from intended route (3838, 73730) →
  (3705, 73705), perpendicular to the planned loop direction

**This is the headline §6 negative result.** Same model, same input
configuration, same map — one rep drove the loop with smooth speed control;
another rep failed to move and drifted off-route. Demonstrates fundamental
unreliability of reward-back-prop LoRA real-vehicle deployment.

## R.5 Baseline vs SE-LoRA Jerk Profile (2026-05-01)

**Source.** `/home/yunli/Pixkit/field_data/2026-05-01/analysis/fig4_accel_jerk.png`

Time-series of rolling-1s jerk RMS:
- R1_rep1_baseline (blue): MULTIPLE spikes up to 20-21 m/s³ jerk RMS over
  the 550 s test; many cycles of 5-13 m/s³ spikes
- R1_rep1_kashiwa_selora (orange): smaller spikes, max ~7 m/s³, generally
  smoother — but only drove for 250 s before stopping
- R1_rep2_kashiwa_selora (green): all zeros (vehicle never moved)

**Surprising finding (relative to our prior intuition).** SE-LoRA, on the
runs where it actually drove (rep1), was SMOOTHER than baseline. This
contradicts the simple "LoRA = worse" narrative. The true story is:
- baseline: drives loop, but jerky (downstream control compensates)
- SE-LoRA rep1: drives loop, smoother (LoRA learns smoother behavior),
  but stops 50% earlier than baseline
- SE-LoRA rep2: catastrophic failure (no movement)

This is the failure-rate vs. quality-when-it-works trade-off that
deployment safety analysis MUST address. The mean violation rate is
misleading; the failure mode is bimodal.

## R.6 Provisional Methods Result (TBD — implementation complete, training pending)

The methods of §7 are implemented at the algorithmic level
(see implementation status in §7.3). Full training + evaluation pending:
1. ROS bag → npz frame extraction
2. Preference pair generation + DPO LoRA training (GPU-overnight)
3. Latent bank build + OOD scaling deployment
4. Per-method C&C violation rate measurement vs §R.1 baselines

We will report:
- C&C violation rate per method on R1 dynamic-driving and garage v=0
- Sensor perturbation degradation curves (per §6.5)
- Bimodal-failure analysis: per-method run-to-run consistency (per R.4)
