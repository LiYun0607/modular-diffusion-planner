#!/bin/bash
# Phase 2b (revised): AD-specific axes that matter for LoRA-fine-tuned diffusion planner
# (NOT a blind copy of Wortsman 2022 ViT-G/14 image-classification axes)
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs_train.jsonl
SWEEP=/root/corl_work/outputs/sweep

# Format: "lr w_imit wd sched sam seed init lora_rank inf_steps subset tag"
# subset: full | v5_only | v2_only | v8_only (filter pref pairs by sampler)
# (lora_rank and inf_steps not yet wired — placeholder; treated as comments for now)
CONFIGS=(
  # ===== Universal regularizers (carry over from Soup paper) =====
  "3e-5 0.5 1e-3 constant 0 42 base wd1e-3"
  "3e-5 0.5 1e-5 constant 0 42 base wd1e-5"
  "3e-5 0.5 1e-4 cosine   0 42 base sched_cosine"
  "3e-5 0.5 1e-4 constant 1 42 base sam"
  # ===== Seed variance (matters for soup pool diversity) =====
  "3e-5 0.5 1e-4 constant 0 123 base seed_123"
  "3e-5 0.5 1e-4 constant 0 7 base seed_7"
  # ===== Init checkpoint (warm-start from existing LoRA vs from base) =====
  "3e-5 0.5 1e-4 constant 0 42 v5 init_v5"
  # ===== Loss-function family (DPO↔imitation spectrum is THE AD-relevant axis) =====
  "3e-5 1.0 1e-4 constant 0 42 base w1.0_pure_imitation"
  "3e-5 0.0 1e-4 constant 0 42 base w0.0_pure_dpo"
  "3e-5 0.9 1e-4 constant 0 42 base w0.9_mostly_imit"
  "3e-5 0.1 1e-4 constant 0 42 base w0.1_mostly_dpo"
  # ===== AD-specific: lower lr + higher imitation weight (more conservative) =====
  "1e-5 0.7 1e-4 constant 0 42 base lr1e-5_w0.7_conservative"
)

for cfg in "${CONFIGS[@]}"; do
  read lr w_imit wd sched sam seed init tag <<< "$cfg"
  OUT=$SWEEP/$tag
  if [ -f "$OUT/history.json" ]; then echo "skip $OUT (done)"; continue; fi
  INIT=""
  [ "$init" = "v5" ] && INIT="--init-lora /tmp/lora_kashiwa_v5_pure.pth"
  SAM=""
  [ "$sam" = "1" ] && SAM="--sam"
  echo "=== $tag (lr=$lr w_imit=$w_imit wd=$wd sched=$sched sam=$sam seed=$seed init=$init)"
  python3 /root/corl_work/scripts/train_with_sample_eval.py \
    --pairs $PAIRS --out-dir $OUT \
    --lr $lr --w-imit $w_imit --wd $wd --schedule $sched \
    --epochs 10 --eval-every 2 --n-eval 30 --seed $seed --max-pairs 1500 \
    $INIT $SAM
done
echo "=== Phase 2b (AD-specific) done"
