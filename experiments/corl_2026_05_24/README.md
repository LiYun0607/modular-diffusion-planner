# CoRL 2026 — Real-Vehicle Anchored Fine-Tuning Experiments (2026-05-24)

This directory contains the implementation + analysis artifacts produced
during a focused work session on 2026-05-24, building on:
- Tonight's 8 LoRA fine-tune variants (v1–v9) and their failure-mode analysis
- The existing SE-LoRA (NeurIPS submission mechanism)
- Real-vehicle ROS bags collected at kashiwa test site (2026-05-01 R1
  dynamic-driving runs + 2026-05-21 garage v=0 + 2026-05-19 v3_dpo bag)
- The JAMA "自動運転の安全性評価フレームワーク Ver.4.0" (Dec 2025) safety
  evaluation framework (for §6 evaluation criteria and §6.5 robustness)

## Layout

```
experiments/corl_2026_05_24/
├── README.md                              # this file
├── scripts/
│   ├── cc_violation_score.py             # JAMA-inspired C&C-style safety scorer
│   ├── real_dpo_lora.py                  # Real-Vehicle DPO LoRA pipeline (§7.1)
│   ├── ood_aware_lora_scaling.py         # OOD-aware inference-time gating (§7.2)
│   └── sensor_perturbation.py            # JAMA Annex-E-inspired ops for §6.5
├── paper_drafts/
│   ├── section6_failure_modes.md         # §6 narrative + measurements
│   ├── section7_method.md                # §7 method (Real-DPO + OOD scaling)
│   └── results_summary.md                # all numbers from real-vehicle bags
└── outputs/
    ├── 4model_per_model_stats.json       # per-model route_dev + traj_L2 stats
    ├── 4model_cc_violation_rates.json    # per-model C&C violation rates (parquet-level)
    ├── 4model_full_cc_scoring.json       # per-model C&C from full 80-pt trajectories
    ├── 4model_route_dev_hist.png         # route deviation distribution per model
    ├── 4model_traj_L2_hist.png           # trajectory L2 divergence from baseline per model
    └── 4model_p95_bars.png               # per-model p95 deviation bars
```

## Provenance: where each artifact came from

| Artifact | Source | Note |
|---|---|---|
| `cc_violation_score.py` | Written 2026-05-24 from JAMA §6 inspiration + GPT review | self-tested |
| `real_dpo_lora.py` | Written 2026-05-24, wraps existing `train_dpo.py` (which produced v3_dpo) | self-tested filter logic |
| `ood_aware_lora_scaling.py` | Written 2026-05-24 from inference-time-gating brainstorm | self-tested |
| `sensor_perturbation.py` | Written 2026-05-24 from JAMA Annex E inspiration | self-tested ops |
| `4model_*.json/.png` | Computed from `/Pixkit/field_data/2026-05-21/offline_ab_4model/4model.parquet` | source data on car-side machine |
| `paper_drafts/*` | Written 2026-05-24, integrating GPT review feedback | use `JAMA-inspired` framing, not "JAMA-certified" |

## Replicate the 4-model C&C results

You need:
1. The car-side `4model.parquet` (`/Pixkit/field_data/2026-05-21/offline_ab_4model/4model.parquet`)
2. The four planner output bags (`stage{1,2}/garage_v0_180856/{baseline,kashiwa_selora}_traj`)
3. A ROS 2 environment with `autoware_planning_msgs` available

Then run:

```bash
# parquet-level analysis (light)
python3 -c "
import sys; sys.path.insert(0, 'experiments/corl_2026_05_24/scripts')
# adapt path to your local copy of 4model.parquet
"

# full trajectory C&C (requires reading the db3 trajectory bags via rclpy)
# see the runner in experiments/corl_2026_05_24/paper_drafts/results_summary.md §R.1
```

## Honest framing (per peer review)

- We do NOT claim to implement the full JAMA Ver.4.0 §6 preventability
  framework. We use a JAMA-INSPIRED subset of comfort + safety thresholds.
- Real-DPO LoRA `chosen` is the real-vehicle ego trajectory FILTERED by
  our C&C-inspired feasibility checks; it is NOT a C&C-driver simulator
  output.
- The OOD-aware scaling is inference-time and training-free; we make
  no claim that it solves all manifold-drift failures.

## Status

| Item | Status |
|---|---|
| C&C scorer | ✓ implemented + tested |
| 5-21 4-model garage v=0 analysis | ✓ done (numbers in `paper_drafts/results_summary.md` R.1, R.2) |
| 5-01 dynamic-driving bags | ⏳ pulling via SFTP (~16GB, slow) |
| OOD scaler | ✓ implemented + tested |
| Real-DPO scaffold | ✓ filter + extractor + train wrapper |
| Sensor perturbation ops | ✓ all 8 operators + sweep runner |
| §6 paper draft | ✓ |
| §7 paper draft | ✓ |
| Real-DPO actual training | ⏳ blocked on bag→npz extraction |
| Latent bank build | ⏳ blocked on GPU + npz dataset |
| Full perturbation sweep | ⏳ blocked on a model_inference_fn wrapper |

## Next concrete steps (in priority order for CoRL)

1. Wire `_extract_real_traj_from_bag` against the 2026-05-01 R1 bags once
   SFTP completes (currently 2.1 GB / 16 GB)
2. Build latent bank: pick 200 kashiwa training npz, run base encoder, save
3. Run a small (5 model × 50 frame × 3 trial) sensor perturbation sweep
4. If GPU available: launch Real-DPO LoRA training overnight
5. Update §7 draft with actual numbers
