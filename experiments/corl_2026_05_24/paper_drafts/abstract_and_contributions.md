# Abstract + Contributions Draft

## Abstract (one paragraph, ~250 words)

Reward back-propagation through differentiable diffusion samplers has emerged
as a popular method for fine-tuning learned planners, but its real-vehicle
deployment safety has not been systematically characterized. We report on
eight reward-back-prop LoRA variants and one Shared+Expert (SE) MoLoRA, each
fine-tuned from a common Diffusion-Planner backbone (Plan-R1) for a single
operational design domain (kashiwa, Japan). Across simulation and real-vehicle
ROS-bag replay collected over four weeks of test-track deployment, none of
the fine-tuned variants matched the safety profile of a Direct Preference
Optimization (DPO) LoRA trained for the same backbone in early 2026 — despite
all variants showing improved held-out reward in simulation. We organize the
observed failure modes into five canonical categories
(cold-start manifold drift, sim-real reward inconsistency, single-ODD vs
cross-region pollution, soup heterogeneity, and sensor-perturbation
sensitivity), and propose two complementary corrections inspired by the
Japanese Automobile Manufacturers Association (JAMA)
"Automated Driving Safety Evaluation Framework v4.0" (Dec 2025): (i)
Real-Vehicle DPO LoRA where the "chosen" trajectories are real-vehicle ego
recordings filtered by Careful-and-Competent-driver-inspired feasibility
checks, and (ii) inference-time OOD-aware LoRA scaling that attenuates
the LoRA contribution based on encoder-latent distance from a training
distribution bank — without retraining. We release the open-source
implementation of the C&C-style scorer, the OOD scaler, and the
JAMA-Annex-E-inspired perturbation operators; preliminary real-vehicle
results from the 2026-05-21 garage v=0 four-model AB and the 2026-05-01
dynamic-driving R1 runs are reported in §6 and §R.

## Contributions

1. **First systematic catalog of reward-back-prop LoRA failure modes for
   diffusion planners**, organized into 5 canonical modes (§6.1–§6.5)
   with worked examples from 8 reward-back-prop variants + 1 SE-LoRA + 1
   DPO LoRA, all on the same kashiwa ODD and shared backbone.

2. **C&C-inspired preventability scorer** (§5.X, scripts/cc_violation_score.py)
   that operationalizes a subset of JAMA Ver.4.0 §6 criteria as a
   reward-function-independent safety metric. Demonstrated to distinguish
   sim-reward gains from real-vehicle safety on the 5-21 garage v=0 4-model
   AB.

3. **Real-Vehicle DPO LoRA** (§7.1, scripts/real_dpo_lora.py): preference
   pairs where chosen = real-vehicle ego trajectory passing C&C, rejected =
   reward-LoRA sample violating C&C. Sidesteps the gradient-through-sampling
   pathology that causes cold-start manifold drift in reward-back-prop LoRA.

4. **OOD-aware LoRA Scaling** (§7.2, scripts/ood_aware_lora_scaling.py):
   inference-time, training-free LoRA delta attenuation based on
   encoder-latent distance from a training-distribution bank. Composable
   with any LoRA-fine-tuned diffusion planner.

5. **JAMA-Annex-E-inspired sensor perturbation operators** (§6.5,
   scripts/sensor_perturbation.py): 8 operators + sweep runner for
   per-model robustness ablation. Validated end-to-end on a 3-model
   mock-inference demo.

6. **Public real-vehicle artifacts** (where licensing permits):
   - 4-model offline AB outputs (3000+ frames per model)
   - C&C-scored trajectories from the same
   - Sim-vs-real ego-velocity distributions, paired trajectory L2 plots
   - Localization-confound analysis from the bimodal SE-LoRA failure run

## Paper outline (proposed)

| § | Topic | Status |
|---|-------|--------|
| 1 | Introduction + Contributions | abstract drafted; contributions enumerated |
| 2 | Related Work | TBD (DPO, model soups, LoRA, JAMA, MoLoRA-NeurIPS-companion) |
| 3 | Background: Diffusion-Planner + LoRA fine-tune | TBD (cite Plan-R1, our NeurIPS MoLoRA) |
| 4 | The kashiwa real-vehicle deployment & dataset | TBD (cite Pixkit + Tier IV) |
| 5 | System & Implementation | placeholder; cite existing ROS bridge + ONNX split |
| 5.X | C&C-style preventability scorer | scorer code + thresholds documented |
| 6 | Failure modes (5 modes, real-vehicle measurements) | section6_failure_modes.md ✓ |
| 6.1 | Cold-start manifold drift | done w/ data |
| 6.1.1 | Offline vs closed-loop divergence (caveat) | done |
| 6.2 | Sim-real reward inconsistency | done w/ 5-01 plots |
| 6.2.1 | Localization confound (honest framing) | done |
| 6.3 | Single-ODD vs cross-region | done (narrative) |
| 6.4 | Soup heterogeneity (v8 negative result) | done |
| 6.5 | Sensor perturbation robustness | scaffolding done; needs real model run |
| 7 | Method: Real-Vehicle DPO + OOD scaling | section7_method.md ✓ |
| 7.1 | Real-Vehicle DPO LoRA with C&C filter | scaffold done |
| 7.2 | OOD-aware LoRA Scaling | implementation done |
| 8 | Results & Ablations | partial (R.1–R.6) |
| 8.X | Limitations & Future Work | TBD |
| 9 | Conclusion | TBD |
| A | Appendix: full perturbation grid + thresholds + ONNX equivalence | TBD |

## Honest framing (for §1 + §8.X)

- We do NOT claim to implement the full JAMA Ver.4.0 §6 preventability
  framework. We use a JAMA-INSPIRED subset of comfort + safety thresholds.
- We do NOT claim Real-DPO LoRA uses a "C&C driver model"; it uses real
  driver trajectories FILTERED by C&C-inspired checks.
- We do NOT claim the OOD detector is novel; the novelty is its application
  to diffusion-planner LoRA gating.
- Bimodal real-vehicle failures (one rep succeeds, one rep catastrophic)
  partially confound planner attribution; we report the localization-side
  evidence openly and discuss attribution limits in §6.2.1.

## Companion / dual-submission policy (per memory)

- NeurIPS submission: MoLoRA mechanism (shared LoRA + per-ODD experts,
  hierarchical training, evaluation in sim).
- CoRL submission (this paper): real-vehicle deployment study, failure-
  mode taxonomy, Real-DPO + OOD scaling methods.
- The two papers share the SE-LoRA artifact but address different research
  questions and use different evaluation regimes. We will explicitly note
  the NeurIPS companion paper in §3 to prevent duplicate-publication
  concerns.
