# In-Garage CoRL Experiments Brief (2026-05-21)

> **Context for the Claude Code instance receiving this brief:** The vehicle is
> currently stationary in the kashiwa test-site garage. It cannot drive yet but
> has live CAN bus, LiDAR (`/sensing/lidar/concatenated/pointcloud`), perception
> tracking output, and chassis (steering, wheel speed, IMU) topics published.
> Diffusion-Planner ONNX models are deployed under `~/autoware_data/diffusion_planner/`.
> Goal: maximize CoRL 2026 paper value during this idle window by producing
> real-vehicle quantitative artifacts that sim cannot provide.

---

## TL;DR — priority order (do A→D, run E in background)

| # | Experiment | Effort | CoRL paper deliverable |
|---|---|---|---|
| **A** | Inference latency profile (3 models × n_steps sweep) | 1–2 h | §5 latency table, "deployable on Tier IV" claim |
| **B** | Real-vehicle v=0 trajectory dump (5+ models) | ~1 h | §6.1 figure replacing sim figure with real-sensor data |
| **C** | ONNX ↔ torch numerical equivalence | 30 min | §A defensive table for reviewers |
| **D** | OOD distance measurement (real-input encoder latent vs training set) | 1–2 h | §6 causal claim: LoRA pulls manifold off under OOD ego state |
| **E** | rosbag record of garage + slow exit | 0 effort | offline replay dataset, for unlimited later analysis |

---

## Repository state you need to know

### Models available on this machine

ONNX (deployable):
```
~/autoware_data/diffusion_planner/
├── v3_dpo/                       PRIMARY (DPO LoRA, March prod, April real-veh validated, May 19 sim re-validated)
├── v3_base_nolora/               base Diffusion-Planner, no fine-tune (sim-clean)
├── v3_kashiwa_v9_best332/        v9 best single (drives but drifts slightly in sim)
├── v3_kashiwa_v8_deep_soup/      DANGEROUS (accelerates through turns in sim)
├── v3_kashiwa_v7_greedy_soup/    drifts in sim
├── v3_kashiwa_v5_pure/           drifts in sim
├── v3_kashiwa_jr_paper_v2/       drifts in sim
├── v3_kashiwa_selora/            SE-LoRA shared+kashiwa expert merged; drifts in sim
└── v3.0/                         args.json + ONNX (older format)
```

PTH (pre-ONNX, for torch-side experiments):
```
/tmp/kashiwa_pth_bundle/           (also at ~/Desktop/kashiwa_pth_bundle.tar.gz)
├── base/best_model.pth                  pretrained Diffusion-Planner backbone
├── dpo/merged_dpo.pth                   = v3_dpo ONNX source
├── molora/molora_full.pth               shared+4-expert MoLoRA stack
├── molora/merged_{hongo,kashiwa,nishishinjuku,shalun,unagoya}.pth
├── kashiwa_v{1-8}_*/lora_*.pth          tonight's reward-back-prop LoRAs
└── kashiwa_v9_grand_soup/
    ├── configs.json                     500 random-search configs
    ├── val_rewards.json                 495 successful val results
    └── top10/variant_{332,213,379,256,270,340,300,324,317,186}.pth
```

### Relevant code locations

- ROS planner node:
  `/root/autoware_ws/src/universe/autoware_universe/planning/autoware_diffusion_planner_onnx_split_cpp/`
- Planner config:
  `/root/autoware_ws/install/autoware_diffusion_planner_onnx_split_cpp/share/autoware_diffusion_planner_onnx_split_cpp/config/diffusion_planner.param.yaml`
- Training utilities:
  `/tmp/kashiwa_deployment_package/utility_scripts/`
  - `train_molora.py` — MoLoRA arch (`apply_molora`, `set_active_expert`, `merge_molora_into_dit`)
  - `train_reward_backprop.py` — differentiable DPM-Solver
  - `export_dpo_split.py`, `export_dit_only.py` — ONNX export utilities
- Base model class:
  `/root/autoware_ws/scripts/train/Diffusion-Planner/diffusion_planner/`
- Kashiwa training data (full npz):
  `/root/autoware_ws/grpo_data/npz/kashiwa_*` (1865 scenes)

### How to launch Autoware (use this verbatim)

```bash
source /root/autoware_ws/setup_onnx_env.sh
source /opt/autoware/setup.bash
source /root/autoware_ws/install/setup.bash
ros2 launch autoware_launch planning_simulator.launch.xml \
  map_path:=/autoware_map/kashiwa \
  vehicle_model:=sample_vehicle \
  sensor_model:=sample_sensor_kit
```
(For real-vehicle launch substitute the real launch file; the user knows it.)

### How to kill Autoware cleanly (mandatory — DDS leftover causes AUTO grey)

```bash
bash /tmp/kashiwa_deployment_package/utility_scripts/kill_autoware.sh
sleep 2
bash /tmp/kashiwa_deployment_package/utility_scripts/kill_autoware.sh
ps aux | grep -E 'autoware|ros2|rviz' | grep -v grep | grep -v defunct | wc -l    # MUST be 0
```

### How to swap which ONNX model is loaded

```bash
YAML=/root/autoware_ws/install/autoware_diffusion_planner_onnx_split_cpp/share/autoware_diffusion_planner_onnx_split_cpp/config/diffusion_planner.param.yaml
TARGET=v3_dpo   # or v3_base_nolora, v3_kashiwa_v9_best332, etc.
sed -i "s|v3_[a-z0-9_]*\/|${TARGET}/|g" "$YAML"
grep "_path:" "$YAML" | head -3   # verify
```

---

## A. Inference latency profile — REAL Tier IV vehicle compute

**Goal:** Produce one table for the CoRL paper §5: "End-to-end planner inference
latency on the deployed system."

**Why this matters:** Reviewers will ask "is this real-time on the target
compute?" Sim numbers don't answer this — real vehicle GPU under realistic load
does.

**Procedure:**

1. Write a standalone Python script `latency_profile.py` that:
   - Loads `context_encoder.onnx` + `dit_core_dynamic.onnx` via `onnxruntime`
     CUDA EP (same provider as the deployed ROS node uses).
   - Constructs ONE fixed dummy input batch matching the planner's input shape
     (use the captured real ROS message at v=0 if possible — see below).
   - Warms up 50 iters, then times 1000 iters of:
     - `context_encoder.run()` alone → record p50/p99
     - one DiT solver step → record p50/p99
     - full sampling loop (n_steps ∈ {5, 10, 20, 50}) → record p50/p99 and FPS
   - Outputs a CSV: `model, n_steps, encoder_ms, dit_per_step_ms, full_ms, fps`
2. Run for each model in `{v3_dpo, v3_base_nolora, v3_kashiwa_v9_best332}` —
   skip the obviously-broken ones, but include the SE-LoRA for completeness.
3. Stretch goal: also measure peak GPU memory via `nvidia-smi pmon` during run.

**Acceptance criteria:**
- CSV at `~/Desktop/corl_garage_results/A_latency.csv` with ≥4 rows per model.
- Each model × n_steps tested ≥1000 forward passes for stable p99.
- p50 and p99 both reported. Standard deviation included.

**Bonus:** Capture an actual ROS planner input message via
`ros2 topic echo /planning/trajectory_generator/diffusion_planner_node/input ...`
(or by adding a tap inside the node) and use it as the timing input — that
makes the "real workload" claim defensible.

---

## B. Real-vehicle v=0 trajectory dump — paper §6.1 figure

**Goal:** Produce one paper figure that shows every LoRA variant's predicted
trajectory when fed the actual stationary-ego sensor input. This is the figure
that turns a sim-only negative result into a real-vehicle finding.

**Why this matters:** Tonight's whole §6 story rests on "LoRA fails at v=0
cold-start." We've shown it in sim. Showing the same 5 trajectories diverging
on REAL LiDAR/perception input is a definitive figure.

**Procedure:**

1. Tap the planner input: capture a real-system input message at v=0 (vehicle
   stationary in garage, perception running). One frame is enough. Save as
   `garage_v0_input.npz` (matching the format the planner consumes —
   `ego_history`, `agent_states`, `static_objects`, `lanes`, `route`,
   `turn_indicator`, etc.).
2. Write `dump_trajectories.py` that:
   - Loads `garage_v0_input.npz`.
   - For each model in this list, runs the encoder+sampler and saves the
     predicted 80-step trajectory (xy + heading) to JSON:
     ```
     models_to_test = [
       'v3_dpo',
       'v3_base_nolora',
       'v3_kashiwa_v9_best332',
       'v3_kashiwa_v5_pure',
       'v3_kashiwa_v8_deep_soup',
       'v3_kashiwa_selora',
       'v3_kashiwa_jr_paper_v2',
     ]
     ```
   - Optionally also load each `.pth` variant directly via torch and dump (for
     models without ONNX, or for the v9 top10).
3. Make `plot_garage_v0.py` — single overlay figure, kashiwa map background,
   ego dot at origin, all model trajectories color-coded with legend. Save as
   `fig_garage_v0_trajectories.{pdf,png}`.

**Acceptance criteria:**
- Output dir: `~/Desktop/corl_garage_results/B_v0_dump/`
- `garage_v0_input.npz` saved (so the experiment is reproducible offline).
- ≥5 model trajectories dumped as `traj_<modelname>.json`.
- One overlay PDF figure produced.

**Critical:** Verify ego_state in the input npz is genuinely v=0 (check
`ego_history[-1, 0:2]` should be near origin AND `ego_history` past 5 frames
should be identical — true stationary, not just current frame v≈0).

---

## C. ONNX ↔ torch numerical equivalence — defensive

**Goal:** One number for the §A appendix: "Max trajectory deviation between
deployed ONNX and reference torch is X mm over 80-step horizon."

**Why:** Reviewers will ask. Better to have an answer.

**Procedure:**

1. Reuse `garage_v0_input.npz` from B.
2. Load `merged_dpo.pth` in torch eval mode, run a deterministic sampler
   (same n_steps as ONNX export — check `export_dpo_split.py` for the value).
3. Load `v3_dpo/*.onnx` via ORT, run the same input.
4. Compute `max(|xy_onnx - xy_torch|)` and `mean(|xy_onnx - xy_torch|)` over
   the 80-step trajectory.
5. Also for one v9 variant (use variant_332).

**Acceptance criteria:**
- Numbers saved at `~/Desktop/corl_garage_results/C_onnx_torch_equiv.txt`.
- If max deviation > 1 cm somewhere, investigate (could indicate ONNX export
  bug, fp16 cast, etc.).

---

## D. OOD distance measurement — paper §6 causal claim

**Goal:** Empirical evidence that LoRA delta magnitude grows under OOD ego
states, supporting the "LoRA pulls trajectory off manifold for OOD inputs"
narrative.

**Why:** Currently §6 just narrates this; with this experiment you can plot
"LoRA delta L2 norm vs. encoder-latent distance from training distribution."

**Procedure:**

1. Build a training-distribution latent bank:
   - Load ~200 random kashiwa training npz scenes from
     `/root/autoware_ws/grpo_data/npz/kashiwa_*`.
   - Run each through the **base** model's `context_encoder` (frozen) to get a
     latent `z_train ∈ R^256` per scene.
   - Save as `latent_bank.npy` shape `[200, 256]`.
2. Capture or synthesize a small set of "in-garage" inputs:
   - The real garage v=0 input from B.
   - Optionally: slow-roll input if you can move car a couple meters.
3. For each garage input `z_in`:
   - Compute distance to nearest training latent (cosine or L2).
   - For each LoRA variant in `{v5_pure, v9_best332, jr_paper_v2, selora}`:
     - Load LoRA delta.
     - Compute `||W_lora @ z_in - W_base @ z_in||` for one or two DiT layers,
       or simply the L2 norm of the LoRA delta applied to the latent.
4. Scatter plot: x-axis = OOD distance to training set, y-axis = LoRA effect
   magnitude. Expect a positive correlation (the larger the OOD, the more
   off-manifold the LoRA pulls — the paper's §6 claim).

**Acceptance criteria:**
- `latent_bank.npy` saved.
- One scatter PDF at `~/Desktop/corl_garage_results/D_ood_scatter.pdf`.
- Pearson correlation coefficient reported in caption.

---

## E. rosbag record — background, free

**Goal:** Capture a real-vehicle dataset for unlimited offline replay.

**Procedure:** Run in background continuously while doing A–D:

```bash
mkdir -p ~/Desktop/corl_garage_results/E_rosbag
cd ~/Desktop/corl_garage_results/E_rosbag
ros2 bag record \
  /sensing/lidar/concatenated/pointcloud \
  /perception/object_recognition/tracking/objects \
  /perception/object_recognition/detection/objects \
  /vehicle/status/velocity_status \
  /vehicle/status/steering_status \
  /vehicle/status/control_mode \
  /tf /tf_static \
  /localization/kinematic_state \
  /map/vector_map \
  /planning/mission_planning/route \
  -o garage_session_$(date +%Y%m%d_%H%M%S)
```

If/when the car can finally crawl out of the garage, keep the recording
running — capture the cold-start moment when ego transitions from v=0 to
v>0. That's the exact moment our v=0 finding predicts a deployed LoRA would
have failed.

**Acceptance criteria:**
- ≥30 min of bag with the topics above.
- Bag file size noted (will likely be tens of GB — copy to NAS or external
  drive if local disk fills).

---

## Notes / gotchas

- **Memory recall**: there is a feedback memory `feedback_kill_autoware_command`
  with the correct 3-pass kill — use it. There is also
  `feedback_lora_v0_cold_start_failure` and `feedback_kashiwa_base_clean` that
  give the prior context.
- **Do NOT modify Autoware system nodes** (perception, control, simulator) —
  there's a feedback memory on that. Only touch the diffusion_planner package.
- **Tier IV compute**: the on-board GPU is whatever Tier IV ships for this
  vehicle (likely Jetson AGX Orin or RTX A4000-class — confirm `nvidia-smi`
  before drawing latency conclusions).
- **Don't trust sim equivalence**: real-vehicle compute may have different
  ONNX Runtime version / different EP optimizations. The numbers from this
  garage session are the ground truth for the paper.

## Deliverable summary (single command at end)

```bash
mkdir -p ~/Desktop/corl_garage_results
tar czf ~/Desktop/corl_garage_results.tar.gz -C ~/Desktop corl_garage_results/
sha256sum ~/Desktop/corl_garage_results.tar.gz
```

Hand this back to the user (or to the main Claude Code instance) — these are
the in-garage real-vehicle CoRL §5/§6/§A artifacts.

## Where the paper draft lives

`/tmp/kashiwa_deployment_package/paper_draft/corl2026_v2/` — when each
experiment finishes, paste the resulting number/figure into the corresponding
section. Specifically:
- A → `§5 System` Table 1
- B → `§6.1` Figure (replace existing sim figure if any)
- C → `§A Appendix` defensive table
- D → `§6 Discussion` causal-claim scatter

Good luck.
