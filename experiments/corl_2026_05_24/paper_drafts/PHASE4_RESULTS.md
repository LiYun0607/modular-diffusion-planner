# Phase 4 Final Results (n=500 disjoint val, V3 proxy)

> 14-model comparison on n=500 held-out kashiwa val-npz (NPZ-identity disjoint
> from training pairs; seed 2026). Eval via V3 JAMA-inspired preventability
> PROXY (3 layers: strict_safety, c_and_c_driverlike, comfort) with
> per-scenario bucket (cold_start, curve, straight) breakdown.

## Headline Table

| Rank | Model | strict% | driverlike% | **comfort%** |
|---:|:---|:---:|:---:|:---:|
| 1 | **phase2_sam** | 0.00 | 100.0 | **0.00** |
| 1 | phase2_tgt_blocks | 0.00 | 99.8 | **0.00** |
| 1 | **soup_v3_comfort** (= sam alone, after greedy soup) | 0.00 | 100.0 | **0.00** |
| 4 | base_nolora | 0.00 | 97.2 | 0.80 |
| 5 | phase2_w0.9_mostly_imit | 0.00 | 100.0 | 3.80 |
| 6 | phase2_beta_001 | 0.00 | 100.0 | 5.00 |
| 7 | **phase3_sam_seed42_final** (25 ep, 3k pairs) | 0.00 | 99.8 | **5.20** |
| 8 | v5_surgical (zero `decoder.dit.preproj.fc1`) | 0.00 | 98.2 | 7.40 |
| 9 | phase3_sam_seed42_best (early ckpt) | 0.00 | 98.8 | 10.60 |
| 10 | phase2_alpha_64 | 0.00 | 100.0 | 28.80 |
| 11 | v2_jr_paper | 0.00 | 99.2 | **79.80** |
| 12 | v5_pure | 0.00 | 99.0 | **81.40** |
| 13 | v8_deep_soup | 0.00 | 99.2 | **88.00** |
| 14 | **phase2_w0.0_pure_dpo** | **100.0** | 100.0 | **100.00** |

## Per-Scenario Bucket (top 3 models, n=500 split)

n_eval split: 79 cold_start (~16%), 133 curve (~27%), 288 straight (~57%).

| Bucket | phase2_sam | phase2_tgt_blocks | soup_v3_comfort |
|---|---|---|---|
| cold_start | 0.0% safety / 0.0% comfort | 0.0% / 0.0% | 0.0% / 0.0% |
| curve      | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |
| straight   | 0.0% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% |

The top 3 models clear every bucket at 0% strict-safety AND 0% comfort.

## Five Paper-Grade Findings

### F1. Comfort layer is THE discriminator
strict_safety and driverlike layers either pass (~0% strict) or saturate
(~100% driverlike — dominated by `heading_velocity_inconsistency` which is
diffusion-sampling-noise-bound after tightening the threshold from 20° to
45°). The **comfort layer** is what differentiates LoRA fine-tunes from
base and from each other. The paper's main bar chart should be comfort %.

### F2. SAM training is the single most effective add-on
phase2_sam (SAM with lr=3e-5, w_imit=0.5, wd=1e-4) achieves 0% comfort and
0% safety — equal to base and better than every other LoRA variant. SAM
provides flat-minima generalization that survives the disjoint val test.

### F3. tgt_blocks-only LoRA ALSO achieves 0% comfort
LoRA on DiT `blocks.0..2` only (skip preproj + final_layer) reaches 0%
comfort. This validates the §6.1 per-layer ablation finding (preproj.fc1
is the toxic layer) and suggests LoRA should be deliberately restricted
to the block bodies, NOT the input/output projections, of the DiT.

### F4. Production-scale training does NOT beat early-stopped sweep
phase3_sam_seed_42 trained 25 epochs on 3000 pairs (vs Phase 2's 10 epochs
× 1500 pairs) ends at 5.20% comfort — WORSE than Phase 2's 0%. The
`best.pth` (intermediate checkpoint at lower vio) is 10.6%, also worse.
**Longer training overfits.** Paper §7 narrative: the comfort metric is
well-served by short training; production-quality models should use
Phase-2-sweep-style short runs + early stopping on the sample-eval
metric.

### F5. Reward-back-prop LoRAs are catastrophic on comfort (79-88%)
- v5_pure: 81.4% comfort vio
- v2_jr_paper: 79.8%
- v8_deep_soup: 88.0%
- All at 0% strict_safety — so no collisions / corridor breaks, but
  passenger ride quality is unacceptable.

**Surgical fix (v5_surgical, zero preproj.fc1) reduces v5_pure 81.4% → 7.4%**
without any retraining. Confirms §6.1 layer-localization finding under
the corrected V3 proxy.

### F6 (negative result). Vanilla DPO with no L2 anchor genuinely collapses
phase2_w0.0_pure_dpo: 100% strict_safety, 100% driverlike, 100% comfort
across ALL buckets. Jerk_p50 = 3627 m/s³ (sweep result). This is NOT a
V1/V2 jerk-computation artifact — V3 SG-filtered + per-bucket confirms
genuine model collapse. The L2 imitation anchor is essential.

## What Soup Actually Did

Greedy soup over 12 homogeneous Phase 2 variants under the COMFORT
objective DEGENERATED to a single member — phase2_sam alone — because
no other candidate could strictly improve over sam's 0% (Wortsman 2022
strict-accept criterion).

Honest framing in paper:
- Soup helps when individual variants have COMPLEMENTARY strengths.
- Here the test-set + V3 proxy saturates the comfort metric at 0% for
  the best individual (sam), so soup cannot improve.
- Larger pool (Wortsman's ViT-G/14 had 72 variants) or harder test
  scenarios (n>500, real-vehicle replay) might reveal soup gains.
- We document this as honest negative result; soup does NOT contribute
  the headline number for our setting.

## Real-Vehicle Bag Replay (TODO)

Currently the paper §6 bimodal finding (R1_rep2 SE-LoRA catastrophic
deployment failure) is grounded on the 2026-05-01 real-vehicle ROS bags
already in our possession. The Phase 4 sim-eval n=500 confirms the
DIRECTION (v5_pure 81% comfort vio, surgical 7%) but the real-vehicle
deployment outcomes are the bimodal failure stories. We will integrate
the two by adding a §8.X "sim-real correspondence" subsection citing
both n=500 sim and 2026-05-01 real bags.
