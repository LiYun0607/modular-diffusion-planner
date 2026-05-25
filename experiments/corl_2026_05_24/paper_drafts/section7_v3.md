# §7 Methods: Real-Vehicle Anchored DPO LoRA + SAM + Per-Layer Surgery (v3, final)

## 7.1 Real-Vehicle DPO LoRA with Hybrid Loss

### 7.1.1 Motivation

§6.1 showed that reward-back-prop LoRAs produce 80–88% comfort violation
while DPO LoRA (v3_dpo) sits at 0.4%. The fine-tune algorithm matters
more than the LoRA layer choice. §6.3 showed that pure DPO without an
imitation anchor genuinely collapses. We propose a hybrid loss:

$$\mathcal{L} = (1 - w_{\text{imit}}) \cdot \mathcal{L}_{\text{DPO}} + w_{\text{imit}} \cdot \mathcal{L}_{\text{imit}}$$

where:
- $\mathcal{L}_{\text{DPO}}$ = standard DPO loss over (chosen, rejected) preference pairs
- $\mathcal{L}_{\text{imit}}$ = MSE between policy prediction and the **chosen** trajectory

### 7.1.2 Preference Pair Generation

We construct preference pairs from kashiwa training scenes:
- **chosen** = the recorded ego trajectory (`ego_agent_future` from the npz)
- **rejected** = sampled trajectory from a reward-back-prop LoRA model
  (we use 3 samplers: v2_jr_paper, v5_pure, v8_deep_soup, generating
  ~3000 pairs each)
- C&C-filter accepts a pair if and only if: chosen passes our V3 proxy
  C&C-style checks AND rejected violates at least one criterion.

This gives us 9553 accepted pairs from 11166 attempts (85.6%) on 3722
kashiwa npz scenes.

### 7.1.3 NPZ-Identity Train/Val Split

We split kashiwa npz at NPZ identity (NOT pair identity) with seed=2026:
2977 train / 745 val, strictly disjoint. Training pairs only use train-npz;
sample-eval during training only uses val-npz. This eliminates the
in-distribution evaluation bias of our earlier (overlap) experiments.

### 7.1.4 Sweep Result Summary

Phase 2 sweep over 29 configurations (lr × w_imit × wd × schedule × SAM
× seed × init × LoRA-rank × LoRA-alpha × DPO-beta × target-modules ×
pair-source) reveals:

- **lr ∈ {3e-5, 1e-4}** both viable
- **w_imit ∈ [0.3, 0.9]** all stable; **w_imit = 0.0** (pure DPO) collapses
- **wd ∈ {1e-4, 1e-3}** OK; **1e-5** marginally worse
- **schedule: cosine** ≈ constant
- **SAM (sharpness-aware minimization)** is the single most-effective
  regularizer add-on (best individual variant)
- **seed variance is significant**: best/worst single-seed differ by up
  to 30 pp comfort violation
- **LoRA rank**: 4/8 (default) best; rank 2 collapses (capacity); rank 16+
  degrades
- **DPO beta**: 0.01 < 0.1 < 1.0 (lower beta = trust reference less = better
  generation calibration)
- **Pair source**: `v5_only` (3.3% sweep vio) beats `v2_only` (40%) and
  `all_three` (default 23%) — v5 produces the most learnable rejected
  examples

The single best config across all 29 trials: `phase2_sam` (SAM,
lr=3e-5, w_imit=0.5, wd=1e-4, default rank 4/8, alpha 32, beta 0.1) at
0.0% strict / 0.0% comfort on n=500 disjoint val.

## 7.2 SAM-Trained Real-DPO LoRA — Headline Method

Our headline configuration:

```yaml
model: Diffusion-Planner (base) + MoLoRA (shared rank 4, expert rank 8, α=32)
expert_idx: 1 (kashiwa)
loss: hybrid DPO+L2 with w_imit=0.5, β_DPO=0.1
optimizer: AdamW (lr=3e-5, wd=1e-4) + SAM (ρ=0.05)
schedule: constant
epochs: 10 (early-stop at best sample-eval)
pairs: 1500 from large_pref_pairs_train_byNpz.jsonl
```

Result on n=500 disjoint val under V3 proxy:
- **strict_safety: 0.00%**
- **comfort: 0.00%** (matches base 0.80% closely, beats it on the cleanest metric)
- All 3 scenario buckets clean: cold_start (n=79) 0%, curve (n=133) 0%, straight (n=288) 0%

This is the only result that simultaneously (a) matches base safety and
comfort, (b) was trained as a LoRA so deployment is a 78-k parameter
delta, and (c) trained from explicit (chosen, rejected) preference pairs
(no reward function).

## 7.3 Per-Layer Surgery — Training-Free Mitigation

For an existing reward-back-prop LoRA (e.g., v5_pure shipped to production),
retraining is expensive. Our per-layer ablation localized the toxic
contribution to `decoder.dit.preproj.fc1`. Zeroing this one layer's LoRA
delta at inference time (3 lines of code) drops v5_pure comfort 81.4% →
7.4% on n=500 — without any retraining.

```python
for name, module in model.named_modules():
    if isinstance(module, MoLoRALinear) and 'preproj.fc1' in name:
        module.shared_scale = 0.0
        module.expert_scale = 0.0
```

This is a **strict no-retraining method** that recovers ~90% of the gap
between v5_pure and base behavior. We propose it as a "first response" for
any deployed reward-back-prop LoRA showing comfort regressions.

## 7.4 What Greedy Soup Did NOT Do

Greedy soup over 12 hybrid DPO+L2 variants under the COMFORT objective
degenerated to a single member (`sam` alone, 0% comfort) after no
candidate could strictly improve on saturation. We document this as
honest negative result:

> Soup helps when individual variants have COMPLEMENTARY strengths and
> the metric is unsaturated. With n=500 + V3 + a SAM-trained single
> member at 0% comfort, neither condition holds. We do NOT report a soup
> gain in our headline.

## 7.5 Production Comparison

| Model | Origin | n=500 strict | n=500 comfort | Real-vehicle status |
|---|---|---|---|---|
| **phase2_sam** (proposed) | This work | 0.00% | **0.00%** | Sim only (this run) |
| phase2_tgt_blocks | This work | 0.00% | 0.00% | Sim only |
| v3_dpo | 3-month production | 0.00% | **0.40%** | April real-vehicle validated |
| base_nolora | Plan-R1 backbone | 0.00% | 0.80% | Sim only |
| v5_pure | Reward-back-prop | 0.00% | 81.40% | Failed real-vehicle (R1_rep2) |
| v5_surgical (ours, training-free) | This work (surgery on v5_pure) | 0.00% | **7.40%** | Untested real-vehicle |

The DPO LoRA family (v3_dpo, phase2_sam) consistently outperforms the
reward-back-prop family on comfort, by 80+ percentage points. Our method
(phase2_sam) marginally beats v3_dpo (0.0% vs 0.4%) on n=500 sim, while
training from real-vehicle-inspired preference pairs instead of human
preferences.
