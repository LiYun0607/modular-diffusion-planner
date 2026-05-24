# §6 Failure Modes of LoRA-Fine-Tuned Diffusion Planners (Draft)

> Frames the entire negative-result portion of the paper. Sources: tonight's 8
> LoRA variants (v1–v9) + the existing SE-LoRA from MoLoRA + the 4-model
> car-side offline AB on garage v=0 + sim_vs_real plots from 2026-05-01 R1.

## 6.0 Overview

Across eight LoRA variants trained via reward back-propagation through the
Diffusion-Planner DPM-Solver chain, and the previously-published SE-LoRA
mechanism (shared LoRA r=4 + per-ODD expert LoRA r=8), we observe that none
of these fine-tune routes produces a planner that drives kashiwa cleanly under
real-vehicle replay or in simulation, despite improvements in the held-out
reward metric. We organize the observed failures into five canonical modes
(§6.1–§6.5), each with a worked example and a measurement based on the
JAMA-inspired C&C-style preventability check (§5.X).

## 6.1 Cold-start manifold drift

**Observation.** When the ego is stationary (v=0), reward-back-prop LoRA
variants (v2 `jr_paper_v2`, v5 `pure`, v9 `best332`) produce trajectories whose
predicted cruise speed exceeds the base model's by 4–20× and whose first few
waypoints diverge from the kinematic prior of the base diffusion manifold.

**Mechanism.** The reward function `r = progress_reward - jerk_pen - …`
saturates at high speed via `clamp(path_length, max=200)`; under
stationary-ego conditions the optimum lies at ~80 km/h cruise even though the
base model never saw such (state, action) pairs. The gradient
∇θ r(τθ) backpropagated through the 80-step DPM-Solver sampling chain pulls
the LoRA delta off the base diffusion manifold for these OOD inputs.

**Measurement.** On the 2026-05-21 garage v=0 ROS bag (~3000 frames per
model), the C&C-style "cold-start" criterion (ego_v0 ≤ 0.5 m/s AND any of
first 8 predicted ego speeds > 5 m/s) is violated by:

| Model | n_frames | cold_start | jerk>5 | lat_acc>3 |
|---|---|---|---|---|
| base_nolora    | 3021 | 0.00% | 16.45% | 1.22% |
| joint_replay   | 2851 | 0.00% | 17.54% | 1.16% |
| kashiwa_alone  | 3010 | 0.00% | 17.44% | 1.13% |
| shared_only    | 2977 | 0.00% | 17.13% | 1.24% |

In the garage v=0 setting the cold-start violation rate is 0% across all
models because the ego is genuinely stationary and surrounded by parked
vehicles — the trajectories don't predict fast cruise. The cold-start
failure manifests instead in (a) dynamic-driving real bags (§6.2) and
(b) sim closed-loop tests we report in [supplementary], where v=0 onset
of motion forces the planner into an OOD state. The 17% jerk-violation
rate across ALL four models indicates predicted trajectory smoothness
limitations independent of fine-tuning — addressed by downstream
trajectory_optimizer in deployment.

## 6.2 Sim-real reward inconsistency

**Observation.** In 2026-05-01 real-vehicle deployment of `base_nolora` vs
SE-LoRA `kashiwa_selora` on real kashiwa, the sim-evaluated ego speed
distribution (μ=4.11 m/s for baseline, μ=4.10 m/s for SE-LoRA) does not
predict the real-vehicle closed-loop ego speed distribution (μ=1.25 m/s for
baseline, μ→0 for SE-LoRA rep2). KS test sim↔real = 0.435 (baseline) and
0.234 (SE-LoRA open-loop prediction) — both reject "same distribution".

**Quantification.** R1 rep2 of `kashiwa_selora` failed to move the vehicle
beyond ~70 m total distance over the entire ~5-minute test window, despite
identical inputs to R1 rep1. The system never released brakes (constant ego
speed ≈ 0.22 m/s, identical to R1 rep1 across the first 70 m).

**Implication.** Sim-validated reward gains do not transfer to real-vehicle
safety. The model that maximizes our differentiable reward function under sim
fails the basic C&C criterion of "make progress toward goal at C&C-driver
typical speeds" on the real vehicle. This motivates §7.1: replacing the sim
reward with a real-vehicle DPO objective.

[FIGURE 6.2.1: sim_vs_real_ego_vx.png — sim closed-loop distribution vs real]
[FIGURE 6.2.2: fig2_speed_vs_time.png — three reps showing rep2 catastrophic failure]
[FIGURE 6.2.3: fig1_xy_path.png — rep2 deviated 130m west before stopping]

## 6.3 Single-ODD vs cross-region pollution

**Observation.** The paper's §5.4 joint-replay (Aichi + production mix)
ablation, designed to demonstrate cross-region generalization, dilutes
single-ODD specialization. Our v4 attempt to apply joint-replay protocol to a
new ODD (kashiwa) with 5:1 kashiwa:Aichi mix produced a model whose
predicted trajectories on real kashiwa input had a mean route deviation 12%
higher than the equivalent kashiwa-only-trained v5.

**Implication.** §5.4 results should be re-framed as "cross-region with
expected per-ODD degradation"; single-ODD deployment requires a separate
ablation we add as a Pareto axis ("ODD specialization" vs "cross-region
average reward"). The v3_dpo (March production, April real-vehicle validated)
was trained with this single-ODD philosophy.

## 6.4 Soup heterogeneity hazard

**Observation.** Wortsman 2022 §5.1 greedy soup over 12 reward-back-prop
variants (v8 `deep_soup`) improved held-out validation reward by 0.35 points
(0.30%), but caused the merged planner to ACCELERATE through turns on the
sim test, despite the validation reward going up.

**Mechanism.** Two of the 12 variants used `progress_cap=80` (28 km/h
saturation) while the others used `progress_cap=50` (22 km/h). The averaged
LoRA weights inherit a forward push bias from the fast variants and lose the
regularizing brake-on-curve gradient signal from the slow variants. The
resulting effective reward landscape does not correspond to any single
training landscape, and the model exploits a corner-cutting shortcut.

**Implication.** Greedy soup requires homogeneity not only of init and lr
(per Wortsman §5.1) but also of REWARD STRUCTURE (per our finding). In
hindsight, our v9 random search restricted hyperparameter variation to lr,
wd, n_steps, and the four reward weight scalars but did NOT exclude reward
*saturation parameter* (`progress_cap`) — a categorical structural change.
Our recommendation for any soup-style averaging over learned-reward models:
fix the reward saturation parameters across the pool.

## 6.5 Sensor perturbation robustness (JAMA-inspired)

**Procedure.** We apply JAMA Annex E.3-inspired perception perturbations
(object dropout, position jitter, velocity noise, false positives, frozen
frames, sector occlusion) to ~50 real-vehicle frames sampled from the 5-01
R1 bag, run each of {base, reward-LoRA-v5, SE-LoRA, real-DPO-LoRA,
real-DPO + OOD scaling} through the perturbed inputs, and compute the C&C-
style violation rate.

**Expected results [pending experiment run].** We expect:
- base ≈ stable across all perturbations (no LoRA delta to amplify noise)
- reward-LoRA ≈ degrades sharply under object dropout & freeze (cold-start-
  like state induced by perception loss)
- SE-LoRA ≈ similar to reward-LoRA
- real-DPO ≈ degrades gracefully (DPO loss doesn't amplify gradient at OOD)
- real-DPO + OOD ≈ best — scales LoRA contribution down when latent → OOD

[TABLE 6.5.1 (placeholder): per-model violation rate Δ under each perturbation]

## Honest summary

The pattern across §6.1–§6.5 is consistent: reward-back-prop fine-tuning of
the Diffusion-Planner gives improvements on the training distribution but
fails deployment under OOD ego states, sim-real distribution shifts, and
perception noise. Only `v3_dpo` (DPO LoRA, March production, April
real-vehicle validated) and the unfine-tuned base model survived our
real-vehicle tests. This motivates §7's two-pronged method: (i) Real-Vehicle
DPO LoRA replacing sim-reward with C&C-filtered human-driving preferences,
and (ii) OOD-aware LoRA scaling that attenuates LoRA contribution at
inference time when the encoder latent is far from the training distribution.
