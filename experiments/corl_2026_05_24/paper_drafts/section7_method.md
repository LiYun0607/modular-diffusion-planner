# §7 Method: Real-Vehicle Anchored Fine-Tuning + OOD-Aware LoRA Scaling (Draft)

> Two complementary methods that together address the failure modes of §6.
> Both retain the Diffusion-Planner backbone and the LoRA fine-tune paradigm
> (compute-cheap, deploys as a delta), and add (a) a real-vehicle anchored
> training objective and (b) an inference-time gating mechanism.

## 7.1 Real-Vehicle DPO LoRA (with C&C-filtered preferences)

### 7.1.1 Motivation

§6.2 demonstrated that sim-evaluated reward gains do not transfer to real-
vehicle safety. We replace the sim reward objective with a DPO (Direct
Preference Optimization) objective over (chosen, rejected) trajectory pairs
where:
- **chosen** = the actual real-vehicle ego trajectory recorded during human
  driving over the next 8 s, IFF that trajectory passes a JAMA-inspired
  Careful-and-Competent driver feasibility check.
- **rejected** = a sample from the reward-back-prop LoRA model at the same
  scene input, IFF that sample VIOLATES the C&C check.

Both filters are essential: chosen-filtering removes recordings of marginal
or erroneous human driving (e.g., near-collision recoveries); rejected-
filtering ensures we are training the model to avoid clearly identifiable
violations rather than ambient sampling noise.

### 7.1.2 C&C-inspired filter

Following GPT-style review of our initial draft (we credit reviewer guidance
in operationalizing), we implement the filter as:

```
A trajectory is C&C-flagged if any of:
  max lat_acc       > 3.0 m/s²    (comfort)
  max longitudinal jerk > 5.0 m/s³  (comfort)
  max speed         > 18.0 m/s     (urban cap)
  max route deviation > 1.5 m       (lane discipline)
  max heading error  > 30°          (lane discipline)
  min TTC to leader < 2.0 s          (safety margin)
  v=0 cold-start drift              (v0≤0.5 AND any v≤8 steps > 5 m/s)
  any backwards motion              (forward driving)
  stops_within_5m AND end_speed > 1 m/s  (goal-reach)
```

This is INSPIRED BY but does NOT claim to implement JAMA Ver.4.0 §6
preventability boundary. Specifically, we use a conservative subset of
comfort thresholds that a C&C driver would not exceed in normal operation;
we make no claim that violating these thresholds is equivalent to an
unpreventable collision in the JAMA sense.

### 7.1.3 Training objective

Standard DPO (Rafailov 2023) adapted for MSE-based diffusion training (per
our existing `train_dpo.py` validated for v3_dpo):

  L_DPO = -log σ(-β × ((l_w - l_ref_w) - (l_l - l_ref_l)))

where:
- `l_w`, `l_l` are policy model MSE losses on chosen and rejected
  trajectories
- `l_ref_w`, `l_ref_l` are reference (base, frozen) model MSE losses on the
  same
- β = 0.1 (matching v3_dpo training config)
- LoRA layers (rank=8, α=32) are inserted into the DiT and trained; base
  params are frozen

### 7.1.4 Pair generation pipeline

```
extract_pref_pairs_from_bag(real_bag, reward_lora_ckpt, base_ckpt):
  for each frame ts in real_bag:
    npz = build_planner_input_npz(scene_at_ts)
    real_traj = extract_next_8s_ego_trajectory(real_bag, ts)
    if not C&C_pass(real_traj): continue       # filter chosen
    sampled = reward_lora_sample(base_ckpt + reward_lora_ckpt, npz)
    if not C&C_violation(sampled): continue    # filter rejected
    yield {npz, chosen: real_traj, rejected: sampled}
```

We bound the pair pool at 10k pairs per real bag to control training cost.

### 7.1.5 Expected results

- Improvement in real-vehicle C&C violation rate vs the reward-LoRA baseline
  (lower is better)
- Trade-off: per-frame sim reward may be lower (we are no longer optimizing
  for it directly)
- We anticipate the model behaves more like a smooth, conservative human
  driver under all conditions including cold-start.

## 7.2 OOD-Aware LoRA Scaling (inference-time, training-free)

### 7.2.1 Motivation

§6.1 showed that LoRA deltas corrupt predictions on OOD inputs that the
base model handles correctly. Instead of retraining, we attenuate the LoRA
contribution at inference time based on the encoder-latent distance from a
training-distribution latent bank.

### 7.2.2 Mechanism

Three offline + one online step:

```
OFFLINE:
  1. Build latent bank: for N=200 random training-distribution scenes,
     run base_encoder(scene) → z ∈ R^D. Save Z_bank ∈ R^[N, D].
  2. Auto-calibrate (τ, σ): for each z_i in bank, compute mean L2 distance
     to its 5 nearest neighbors → τ = 90th percentile, σ = 0.3 × τ.

ONLINE (each frame):
  3. z = base_encoder(scene)
  4. d = min L2 distance from z to Z_bank
  5. α(d) = sigmoid(-(d - τ) / σ)        # α≈1 in-dist, α→0 OOD
  6. effective_LoRA_delta = α × LoRA_delta
```

α is a single scalar multiplier on the full LoRA delta. No retraining is
required. Implementation cost: one extra encoder forward pass + one L2-NN
query (200 × 256 vector compare = trivial).

### 7.2.3 Implementation note

In the existing Diffusion-Planner codebase, `apply_molora(model, scale=1.0)`
already accepts a scale arg. We add a wrapper `apply_molora_ood(model, z)`
that computes α(z) and forwards `scale=α(z)`.

For SE-LoRA (shared + expert), we apply OOD scaling to the *expert* delta
only, leaving the shared delta at α=1 (the shared part is trained on all
ODDs so isn't particularly OOD-sensitive).

### 7.2.4 Pairing with §7.1

The two methods compose naturally: real-DPO LoRA gives a *trained-correctly*
delta; OOD-aware scaling gives a *deployed-correctly* mechanism that
guarantees the delta doesn't make things worse on OOD inputs. We expect the
combined system to dominate both ablations and the existing v3_dpo baseline
on the real-vehicle benchmark.

## 7.3 Implementation status

- C&C scorer module: `/root/corl_work/scripts/cc_violation_score.py`
  — implemented, self-tested.
- Preference pair filter + extractor scaffold:
  `/root/corl_work/scripts/real_dpo_lora.py` — filter impl complete and
  self-tested; bag-extraction skeleton in place (real implementation needs
  ROS env to convert /localization/kinematic_state + /tf to ego-frame
  trajectory).
- OOD-aware LoRA scaling: `/root/corl_work/scripts/ood_aware_lora_scaling.py`
  — `LatentBank` + `OODScaler` classes implemented and self-tested.
  Integration with `apply_molora` is a 5-line wrapper.
- Sensor perturbation ops:
  `/root/corl_work/scripts/sensor_perturbation.py` — all 8 operators
  implemented and tested for shape correctness.

The remaining engineering for paper results is:
1. Wire `_extract_real_traj_from_bag` against ROS bag (1 evening of work).
2. Patch `train_dpo.py` to apply LoRA at line ~75 (one-liner using
   `train_molora.apply_molora`).
3. Build latent bank from 200 kashiwa npz scenes (1 hour, GPU).
4. Run the per-method ablation: {base, reward-LoRA, SE-LoRA, real-DPO,
   real-DPO + OOD} × {clean, perturbed} × ~50 real frames → C&C table.
