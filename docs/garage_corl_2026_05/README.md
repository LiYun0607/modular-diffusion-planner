# Garage CoRL Experiments (2026-05-21)

Documentation drop produced during a stationary-vehicle window at the kashiwa
test site, ahead of the 2026 CoRL submission. Use these two documents to plan
real-vehicle CoRL §5/§6/§7 experiments that exploit the in-garage time when
CAN bus + LiDAR + chassis telemetry are live but the vehicle cannot drive.

## Files

- **[garage_corl_experiments.md](./garage_corl_experiments.md)** —
  Actionable brief. Five experiments A–E (latency, v=0 dump, ONNX equivalence,
  OOD scatter, rosbag), each with procedure + acceptance criteria + paper
  section mapping. Self-contained — another Claude Code instance can pick this
  up cold and execute.
- **[garage_brainstorm.md](./garage_brainstorm.md)** — Broader idea space (25
  numbered ideas + 4 second-order combinations), each scored on
  effort/impact/novelty/defensibility. Pick from here to refine into the brief.

## Quick orientation

If you have ~2 hours:
1. Read `garage_corl_experiments.md` section A and B.
2. Run experiment A (latency profile, ~1 hour).
3. Run experiment B (v=0 trajectory dump, ~1 hour).

If you have a full day:
- Do A → B → C → D in the brief.
- Start E (rosbag) immediately in background.
- Then pick top-1 from `garage_brainstorm.md` "Top-3 picks" section.

## Key constraints (read before starting)

- The vehicle is stationary and cannot drive — design around static-ego inputs.
- Sim-trained models drift on real-vehicle perception input (confirmed
  2026-05-19 sim test). Only `v3_dpo` (DPO LoRA, April real-vehicle validated)
  and `v3_base_nolora` were sim-clean.
- Tonight's reward-back-prop LoRA variants (v2/v5/v7/v8/v9, plus the existing
  SE-LoRA) all drifted in sim. They are the negative-result material for §6.
- Use the 3-pass DDS-cleaning kill (`kill_autoware.sh` from
  kashiwa_deployment_package) — single-pass kill leaves shared memory leftovers
  that cause the AUTO button to grey on the next launch.
