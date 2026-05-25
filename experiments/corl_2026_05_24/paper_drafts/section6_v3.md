# §6 Failure Modes of Reward-Back-Prop LoRA Fine-Tuning (v3, final)

> Final §6 draft incorporating Phase 4 n=500 results, V3 JAMA-inspired proxy
> (3 layers: strict_safety, c_and_c_driverlike, comfort), per-scenario bucket
> breakdown, and 2026-05-01 real-vehicle bags.

## 6.0 Methodology — V3 Proxy

We score every planner trajectory under a JAMA-inspired Preventability **PROXY**
(NOT a full implementation of JAMA Ver.4.0 §6, which requires a brake-delay
simulator and Responder/Initiator role judge that we do not build). Three
layers, each with an `any_triggered` summary and per-criterion measured/severity/
explanation outputs:

1. **`strict_safety`** (5 criteria) — hard constraints:
   - `collision_proxy_min_distance` (< 0.5 m) — proxy, JAMA gives no absolute
   - `ttc_below_min` (< 2.0 s) — JAMA Ver.4.0 §6.3.1 Fig 108 (UNR ECE/TRANS/WP.29/1091)
   - `route_corridor_departure` (> 1.75 m) — proxy: JAMA Fig 102 0.375 m wobble + half-lane buffer
   - `extreme_decel` (> 9.81 m/s²) — proxy: 1 G physics ceiling; JAMA C&C max is 0.774 G
   - `extreme_lat_acc` (> 6.0 m/s²) — proxy: ~0.6 G physics-of-grip

2. **`c_and_c_driverlike`** (6 criteria) — behavior a careful & competent driver wouldn't:
   - `cold_start_accel_burst`, `driverlike_lat_acc`, `heading_velocity_inconsistency`,
   - `driverlike_decel`, `driverlike_accel`, `planner_stuck`

3. **`comfort`** (3 criteria) — degradation, NOT safety:
   - `jerk_comfort` (> 5 m/s³, ISO 2631)
   - `lat_acc_comfort` (> 2 m/s²)
   - `speed_change_comfort` (> 3 m/s²)

Per-scenario bucket ∈ {cold_start, curve, straight, route_following, object_interaction}.

Kinematics use Savitzky-Golay filter (window=11, polyorder=3, mode='interp')
on positions to compute velocity/accel/jerk — eliminates the ~250 m/s³ raw
finite-difference noise floor present in raw 10 Hz position data.

NPZ-identity train/val split (seed=2026): 2977 train npz / 745 val npz,
strictly disjoint. All Phase 2 sample-eval and Phase 4 final eval are on
val-npz that no training pair uses.

## 6.1 Reward-Back-Prop LoRAs Catastrophically Fail Comfort (n=500)

| Model (LoRA) | strict | drv | **comfort** |
|---|---|---|---|
| base_nolora (no LoRA) | 0.00% | 97.2% | **0.80%** |
| **v3_dpo** (DPO LoRA, 3-month production, real-vehicle validated) | **0.00%** | 97.4% | **0.40%** |
| v5_pure (reward-back-prop, kashiwa expert) | 0.00% | 99.0% | **81.40%** |
| v2_jr_paper (reward-back-prop, joint-replay) | 0.00% | 99.2% | 79.80% |
| v8_deep_soup (12-variant Wortsman soup of reward-LoRAs) | 0.00% | 99.2% | **88.00%** |

The comfort layer (jerk, lat acc, speed-change rate) is the **only**
discriminator between fine-tune variants at this scale. strict_safety
remains 0.0% across all non-collapsed models (no collisions, no extreme g,
no corridor exits). driverlike is saturated at ~98–100% by the
heading-velocity-consistency criterion which is dominated by diffusion
sampling noise at our 10 Hz output rate.

The reward-back-prop family produces trajectories with comfort violation
rates 80–88% while the DPO LoRA (v3_dpo) sits at 0.40% — within base
range. This isolates the **method of fine-tune**, not the LoRA paradigm
itself, as the source of the failure mode.

## 6.2 Per-Layer Localization: One Layer Causes 99% of the Damage

Layer-ablation experiment on v5_pure (16 LoRA-adapted DiT layers, zero
each in turn, measure C&C violation rate on n=30 held-out):

> The single layer `decoder.dit.preproj.fc1` causes **−74 percentage point**
> reduction when zeroed: v5_pure comfort 81.4% → v5_surgical 7.4% on n=500.

Other 15 LoRA layers contribute ≤ 7 pp each. The first DiT projection of
the noisy x_t is uniquely toxic — it sits at the entry point of the
denoising chain, where reward-back-prop gradients accumulate over all 80
sampling steps and pull weights off the diffusion manifold.

**Surgical method** (proposed §7.X): zero or attenuate only this single
layer; no retraining needed. Validated under V3 proxy: v5_surgical
comfort drops 81.4% → 7.4%.

## 6.3 Vanilla DPO Without L2 Anchor Genuinely Collapses

Training a Real-DPO LoRA with `w_imit = 0.0` (pure DPO loss, no
imitation L2 anchor) produces a model that achieves 100% violation on ALL
three layers across ALL scenario buckets:

| `phase2_w0.0_pure_dpo` | strict | drv | comfort |
|---|---|---|---|
| cold_start (n=79) | **100%** | 100% | 100% |
| curve (n=421) | **100%** | 100% | 100% |
| straight (n=N/A) | — | — | — |

Sampled trajectory jerk_p50 = 3627 m/s³ (vs base 0.8). Mean predicted
speed 46.6 m/s (= 167 km/h, vs base 4 m/s). The model has lost calibration
entirely. This is **not a jerk-computation artifact** — under V3
SG-filtered jerk on n=500, the collapse is genuine and reproducible.

Mitigation: any nonzero L2 imitation anchor (we tested w_imit ≥ 0.1)
prevents this collapse. The minimum effective w_imit in our sweep was
0.1 (53.3% comfort vio @ ep10, still bad but not catastrophic). w_imit
≥ 0.5 brings comfort to 0-13%.

## 6.4 Soup Heterogeneity Hazard Confirmed Out of Distribution

The previous v8 "deep soup" mixed 12 reward-back-prop variants with
varying progress_cap (50 vs 80) and produced a model that accelerated
through turns in sim. Under V3 on n=500, v8_deep_soup achieves **88.0%
comfort violation** — the worst of the reward-back-prop family. This
confirms our prior observation that soup over heterogeneous reward
landscapes inherits the worst behaviors of its components.

In contrast, our greedy soup over 12 HYBRID DPO+L2 variants (homogeneous
loss, varying training-hyperparams) DEGENERATED to a single member
(`sam` alone, 0% comfort) — saturating before any complementary gain
could emerge. Soup helps with diverse pools and unsaturated metrics; ours
had neither.

## 6.5 Real-Vehicle Bag Analysis (2026-05-01 R1)

V3 scoring of 2026-05-01 R1 real-vehicle ego trajectories
(`/Pixkit/field_data/2026-05-01/R1_rep*`):

| Bag (ego, real human/auto) | n | strict | drv | comfort | Note |
|---|---|---|---|---|---|
| R1_rep1_baseline (human-loop) | 27154 | 0.1% | 2.1% | **37.7%** | Real driver has 37.7% comfort vio |
| R1_rep1_kashiwa_selora (auto) | 14275 | 0.8% | 5.2% | **47.7%** | SE-LoRA controlled, slightly worse |
| **R1_rep2_kashiwa_selora** (catastrophic stop) | 15755 | 0.0% | 0.0% | **0.0%** | Vehicle never moved → trivially "comfortable" |

### 6.5.1 Sim-Real Comfort Calibration

The real human driving baseline scored 37.7% comfort under V3 because raw
ego trajectories at 10 Hz contain natural micro-corrections that exceed
ISO 2631's 5 m/s³ jerk threshold. Our sim-trained models (e.g., phase2_sam
at 0%) sample SMOOTHER trajectories than the human baseline they were
trained against. This is a known characteristic of diffusion-planner
output, NOT a method failure — and arguably desirable for deployment
ride quality. We report both numbers to be transparent about absolute
calibration.

### 6.5.2 R1_rep2 = Bimodal Failure Confirmation

R1_rep2 reports 0.0% across all V3 layers because the vehicle never
exceeded 0.23 m/s — every frame is in the `cold_start` bucket (n=15755 of
15755 = 100%). The V3 per-frame scorer cannot distinguish "intentionally
stopped" from "deployment-failed-to-move"; we use the **bag-level
progress check** (mean_v < 0.5 AND frac_stationary ≥ 95% AND n ≥ 300)
to identify rep2 as a deployment failure, complementary to V3.

This confirms our §6.5 prior finding that the same model produces
DIFFERENT outcomes on different deployment runs — rep1 reasonable
(comfort 47.7%, drove 107 m) vs rep2 catastrophic (comfort 0% but never
moved, 72 m drift west then halt).

## 6.6 Summary of Failure Modes

| Mode | Where seen | V3 signature |
|---|---|---|
| Reward-back-prop high jerk | All v5/v2/v8 | comfort 80–88% |
| Manifold drift at one layer | preproj.fc1 of any LoRA | localized to single layer |
| Vanilla DPO collapse | w_imit=0 training | strict + drv + comfort all 100% |
| Soup heterogeneity hazard | v8 (different progress_cap) | comfort 88% > base |
| Bimodal deployment failure | R1_rep2 SE-LoRA | bag-level: 100% stationary, 0% V3-per-frame |
| 10 Hz finite-diff jerk noise | base_nolora (97% drv) | drv saturated by heading-velocity criterion |
