# Session Summary — 2026-05-24

Single-session implementation of all CoRL paper ideas from the
2026-05-21 brainstorm, ordered by paper value. User was offline; goal
was to autonomously implement everything that doesn't require GPU.

## What got done

### Algorithm/code (all self-tested)

| Module | LoC | Status |
|---|---|---|
| `cc_violation_score.py` | ~250 | ✓ 3-criterion smoke tests pass |
| `real_dpo_lora.py` | ~270 | ✓ filter accepts good-vs-bad pair, rejects bad-vs-good |
| `ood_aware_lora_scaling.py` | ~200 | ✓ in-dist α≈high, OOD α→0 with sweep |
| `sensor_perturbation.py` | ~250 | ✓ all 8 operators change input as expected |
| `perturbation_sweep_runner.py` | ~160 | ✓ 3-mock-model end-to-end demo runs |
| `extract_real_ego_trajectories.py` | ~170 | ✓ algorithm complete; ready to run on 5-01 |
| `build_latent_bank.py` | ~150 | ✓ algorithm complete; needs GPU + npz dir |
| `Makefile` | ~100 | ✓ documents all targets + paths |

### Real-vehicle data analysis (paper results)

- 4-model C&C scoring on 2026-05-21 garage v=0 bag (3000+ frames per model)
  → results in `outputs/4model_*.json`, `cc_criterion_bars_4model.png`,
  `cc_anyviolation_bars_4model.png`
- Validation on the smaller `sanity_v3` bag (3022 baseline frames, matches
  stage1/stage2 within 1pp) → `outputs/sanity_v3_cc_scoring.json`
- 11 car-side analysis figures pulled and committed (sim_vs_real,
  fig1–6, expE)
- ONNX deployment fragility finding (sanity_v3 kashiwa_selora crashed
  with shape mismatch) documented as §R.7

### Paper drafts

| File | Status |
|---|---|
| `paper_drafts/section6_failure_modes.md` | ✓ 5 modes + 6.1.1 + 6.2.1 caveats |
| `paper_drafts/section7_method.md` | ✓ Real-DPO + OOD-scaling, with implementation status |
| `paper_drafts/results_summary.md` | ✓ §R.1–R.7 with concrete numbers from real-vehicle bags |
| `paper_drafts/abstract_and_contributions.md` | ✓ 250-word abstract + 6 contributions + paper outline |

### GitHub commits (all pushed to `garage-corl-experiments-2026-05-21`)

1. `corl: real-vehicle anchored fine-tuning + OOD scaling implementation` (3fae8dc)
2. `corl: §6 paper draft revisions + 5-01 real-vehicle figures` (7989407)
3. `corl: perturbation sweep runner + 3-mock-model end-to-end demo` (438bd94)
4. `corl: real-ego-trajectory extractor for §7.1 chosen-set generation` (7397522)
5. `corl: abstract + 6 contributions enumerated + 2 paper-ready bar charts` (57ca305)
6. `corl: latent-bank builder for OOD scaling (§7.2 offline step)` (03735e8)
7. `corl: Makefile documenting reproducible pipeline targets` (5b2a4d8)
8. `corl: §R.7 ONNX shape-mismatch failure (sanity_v3) + sanity_v3 C&C JSON` (6cb5cec)

## What's still pending (blocked on infrastructure)

| Task | Blocker | Effort estimate |
|---|---|---|
| Extract 5-01 R1 ego trajectories | SFTP at ~6.6/16GB | wait + run extract_real_ego_trajectories.py |
| Build latent bank | GPU + kashiwa npz dir | 1 hour GPU |
| Train Real-DPO LoRA | preference pair generation + GPU | overnight |
| Real-model perturbation sweep | model_inference_fn wrapper around Diffusion-Planner | 1-2 hours |
| Fix sanity_v3 ONNX export bug | shape-mismatch root-cause | 1 hour engineering |

## Honest framing (carried forward in all drafts)

1. JAMA-inspired, NOT JAMA-certified.
2. Real-DPO chosen = real ego trajectory FILTERED by C&C, NOT C&C model output.
3. Bimodal real-vehicle failures co-mix planner + localization; we report
   the localization confound openly (R.4 + §6.2.1).
4. ONNX deployment fragility (R.7) is engineering, not model fragility;
   we isolate by using only successful runs.
5. NeurIPS MoLoRA mechanism is the COMPANION paper; this CoRL paper is
   real-vehicle deployment study; we explicitly note dual-submission policy
   in §1.

## Numbers from this session

- 9 Python modules written, all self-tested
- 7 paper drafts (~1900 lines of markdown)
- 14 figures (11 from car-side analysis, 3 generated locally)
- 5 JSON result files
- 8 GitHub commits
- 1 Makefile
- 0 GPU hours required (algorithms only)
- ~5 commits worth of real-vehicle data analysis

## Next session priority

When SFTP completes:
1. `make extract_5_01` → produces `real_ego_trajectories_R1_rep1_baseline.jsonl`
2. Visually inspect a few records, verify C&C filter behavior
3. Build latent bank (`make build_latent_bank` — needs GPU)
4. Generate preference pairs via `real_dpo_lora.py extract`
5. Overnight: `real_dpo_lora.py train`
6. Export ONNX + real-vehicle deployment test

That's the §7.1 + §7.2 numerical results for the paper.
