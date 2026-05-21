# Garage-Time CoRL Brainstorm (2026-05-21)

> Brainstorm to maximize CoRL 2026 paper value while stranded in the kashiwa
> garage with live CAN bus + LiDAR + chassis telemetry. Companion to
> [`garage_corl_experiments.md`](./garage_corl_experiments.md) (the actionable
> brief). This file is the broader idea space — pick from here, then refine
> into the brief format if executing.

## Scoring axes

Each idea scored on:
- **E** = effort (1=easy, 5=can't do in-garage)
- **I** = impact on CoRL paper (1=tiny, 5=section-defining)
- **N** = novelty for reviewers (1=table-stakes, 5="never seen this")
- **D** = defensibility against reviewer attack (1=weak, 5=ironclad)

## §5/§6 main-track (already in brief)

| # | Idea | E | I | N | D |
|---|------|---|---|---|---|
| 1 | Inference latency + n_steps sweep | 2 | 4 | 2 | 5 |
| 2 | v=0 real-sensor trajectory dump (5+ models) | 2 | 5 | 4 | 5 |
| 3 | ONNX ↔ torch numerical equivalence | 1 | 2 | 1 | 5 |
| 4 | OOD-latent vs LoRA-delta scatter | 3 | 4 | 4 | 4 |
| 5 | rosbag recording (background, free) | 1 | 3 | 1 | 3 |

## Extended ideas (new)

### A. Measurement / data (in-garage feasible)

| # | Idea | E | I | N | D | Notes |
|---|------|---|---|---|---|-------|
| 6 | Multi-seed diffusion uncertainty quantification | 2 | 4 | 4 | 4 | 8 noise seeds per model → trajectory variance. v3_dpo should be tight, reward-back-prop LoRA should be wide. Direct evidence of manifold corruption. |
| 7 | Per-layer LoRA ablation | 3 | 4 | 5 | 4 | Zero out one LoRA layer at a time on v5_pure, watch v=0 trajectory. Localize which DiT block is the manifold-drift source. §6 lesion-localization figure. |
| 8 | Diffusion-timestep intermediate dump | 2 | 3 | 3 | 3 | Save trajectory at each denoising step. Does LoRA break early (cold-start drift) or late (refinement)? |
| 9 | Linear interpolation between LoRAs | 3 | 4 | 5 | 4 | α ∈ [0,1] interpolate v3_dpo ↔ v5_pure. Find α where drift starts. LMC-style figure reviewers love. |
| 10 | Per-head attention map for each LoRA | 4 | 3 | 4 | 3 | Compare reward-LoRA vs DPO-LoRA attention on lane queries. §A figure. |
| 11 | Sim2real input-distribution gap quantification | 2 | 5 | 5 | 5 | Same scene in sim AND on car. Compare context_encoder latents. "Our experiments are sim-only, but the latent-space gap to real is ε" — CoRL reviewers eat this up. |
| 12 | Per-class object response | 3 | 3 | 3 | 3 | Person walks in front of car, observe avoidance amplitude per model. |
| 13 | Real-vehicle vs sim latency comparison | 2 | 3 | 2 | 4 | Same ONNX on dev laptop vs Tier IV. Supports "deployable on edge". |

### B. Paper-argument construction (think in-garage, prose later)

| # | Idea | E | I | N | D | Notes |
|---|------|---|---|---|---|-------|
| 14 | Build "LoRA failure-mode taxonomy" | 1 | 4 | 5 | 5 | Catalog tonight's 8 LoRA failures into 5 modes (cold-start drift, reward hacking, manifold off, soup heterogeneity, Aichi pollution), one paragraph + one example + one mini-figure each. The §6 scaffold. Most-citable section of the paper. |
| 15 | "OOD-aware LoRA gating" concept proof | 2 | 5 | 5 | 3 | Use idea-4 latent bank to train a 1-NN OOD detector. At v=0 garage input, gate LoRA delta → 0 (fall back to base). Measure clean trajectory. **This is the §7 method contribution** — converts tonight's negative result into a positive paper. |
| 16 | "Why DPO doesn't drift" mathematical argument | 1 | 5 | 5 | 5 | DPO gradient ignores the sampling chain → no accumulated backprop error → manifold intact. Reward-back-prop ∇θ r(τ(θ)) backpropagates through the entire chain → at v=0 cold-start the gradient explodes against the conditioning. Most academically substantive section. |
| 17 | Reanalysis of §5.4 Pareto with single-ODD distinction | 2 | 4 | 2 | 5 | Add tonight's "single-ODD vs cross-region" axis to the existing Aichi figure. |
| 18 | CoRL/NeurIPS contribution-cut diagram | 1 | 3 | 2 | 5 | One-page block diagram: NeurIPS = MoLoRA mechanism, CoRL = real-vehicle failure modes + OOD-aware method. Prevents dual-submit overlap (per memory). |

### C. Ambitious / creative (paper-flagship potential)

| # | Idea | E | I | N | D | Notes |
|---|------|---|---|---|---|-------|
| 19 | "Manifold projection" new method | 4 | 5 | 5 | 3 | Project LoRA delta onto column-space of base-model Jacobian during training → force stay-on-manifold. Math is SGD-with-projection. If garage time allows, run a tiny prototype. Plan-B for §7. |
| 20 | "Reverse trajectory" reward re-evaluation | 3 | 4 | 5 | 4 | Feed v3_dpo's real-car trajectory back into our reward function. If real reward < training reward but humans say it's good → label as "our reward function is biased" → explains §6.4 soup failure. |
| 21 | Closed-loop ablation on rosbag | 3 | 5 | 4 | 5 | Replay garage+slow-exit bag through each planner offline, compute reactive-agent NDS score. Standard CoRL closed-loop result, stronger than sim. |
| 22 | OOD detector → adaptive n_steps | 3 | 4 | 5 | 3 | OOD → more denoise steps (stable); IID → fewer (fast). Extension of idea 15; OOD detector doubles as budget scheduler. Nice §7 ablation. |
| 23 | Per-ODD expert replacement experiment | 2 | 3 | 3 | 4 | Feed v=0 garage input to merged_kashiwa vs merged_shalun vs merged_hongo. Show wrong-expert → drift. Reinforces §6. |
| 24 | Real-vehicle → sim → real-vehicle round-trip | 4 | 5 | 5 | 4 | Replay garage bag in sim, feed sim planner output to real car control. See how sim-trained model survives real actuator delay. Major work, could carry §7. |
| 25 | One-page replication recipe | 1 | 3 | 3 | 5 | A README that says "to reproduce, run scripts A/B/D in this order." CoRL increasingly grades on reproducibility. |

## Second-order combinations (idea × idea)

Non-obvious but high-value:

### Combo α: **15 + 4 + 6**
OOD detector trained on latent bank, gates LoRA delta. Then run multi-seed uncertainty on gated vs ungated. Expected outcome: gated LoRA variance shrinks to near-base levels. **This is the empirical proof that OOD is the variance/manifold-off root cause** — a complete §7 contribution.

### Combo β: **2 + 7 + 16**
v=0 real input × per-layer LoRA ablation × DPO non-drift math. Localize a single DiT layer as the cold-start manifold-drift source. Show the math says "this layer's gradient blows up at v=0 under reward-back-prop." Show that disabling it (or applying the gating from idea 15) fixes the trajectory. **Strongest §6 + §7 combined argument.**

### Combo γ: **11 + 4**
Sim2real gap measured in encoder-latent space + LoRA-delta vs OOD-distance correlation → one figure with x-axis = sim→real latent distance, y-axis = LoRA delta magnitude. Tells reviewers "sim2real gap × LoRA = real-vehicle failure" in one image. Highly compact, highly defensible.

### Combo δ: **14 + 25**
Failure-mode taxonomy + replication recipe → an independently-citable "Kashiwa LoRA Failure Benchmark" that future work has to engage with. Long-tail citation play.

## Top-3 picks if garage time is tight

1. **Idea 2** (v=0 real-sensor trajectory dump). 30–60 min. Almost-guaranteed paper-figure.
2. **Idea 15** (OOD-gating concept proof). 1–2 h. Converts tonight's negative result into a §7 positive method contribution.
3. **Idea 21** (closed-loop ablation on rosbag). 0 garage time (just record), analysis later in lab. Brings CoRL-standard real-vehicle closed-loop result.

Bonus: Idea 16 (DPO math) is essentially zero-effort once thought through — pure prose during downtime.

## Constraints / dependencies

- Ideas 4, 11, 15, 22 all need a **latent bank from training data** — build it once, reuse.
- Ideas 6, 7, 8, 9 all need a **fixed real-input batch** — capture once from the garage, reuse.
- Idea 21 needs **a recorded bag with control_mode + planner_output topics** — set the recording right NOW so the garage→exit transition is captured.
- Ideas 19, 24 are stretch — probably don't finish in garage but valuable to scope.

## Section-by-section CoRL paper integration

| Paper section | Which ideas feed it |
|---|---|
| §5 System / Implementation | 1, 3, 13, 25 |
| §6 Failure-mode analysis | 2, 6, 7, 8, 14, 17, 23, combo β |
| §6.1 specifically (cold-start) | 2, 7, 8, 16, combo β |
| §6.2 (reward hacking) | 14, 20 |
| §6.4 (soup heterogeneity) | 14, 17 |
| §7 New method (OOD-aware gating) | 4, 15, 19, 22, combo α |
| §A Appendix (defensive) | 3, 10, 18 |
| Datasets/Benchmarks card | 21, 25, combo δ |
| Discussion / future work | 11, 24, combo γ |

## Open questions worth thinking about while stuck

1. Is there a way to measure "how off-manifold" a trajectory is, in a model-agnostic way? (For idea 15 detector training signal.)
2. Could we ship the OOD detector as a separate ROS node so other Autoware users can drop it in front of any LoRA-tuned planner?
3. If real-vehicle closed-loop NDS beats sim NDS for v3_dpo (unlikely but possible), what does that say about our sim metric? (Counterintuitive but defensible negative result for §6.)
4. The 1.3GB pth bundle is too big for GitHub LFS free — should the OOD detector + a representative subset of LoRAs be the actual public release artifact?

## End-of-day deliverable (single command)

```bash
mkdir -p ~/Desktop/corl_garage_results
tar czf ~/Desktop/corl_garage_results.tar.gz -C ~/Desktop corl_garage_results/
sha256sum ~/Desktop/corl_garage_results.tar.gz
```

Hand back to main Claude instance / paper draft.
