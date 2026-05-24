#!/bin/bash
# Phase 2c: AD-LoRA-specific axes (rank, alpha, dpo beta, target modules, pair-source ablation)
# These are NOT in Wortsman 2022 because they did full FT, not LoRA — but they matter for our PEFT setting.
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs_train.jsonl
SWEEP=/root/corl_work/outputs/sweep

# baseline config to vary AROUND: lr=3e-5, w_imit=0.3 (Phase 2a winner so far)
# format: "tag rank alpha beta target_modules pair_subset"
CONFIGS=(
  # ===== LoRA rank (PEFT-paper key axis) =====
  "rank_2_4    2  4  16.0 all      none"
  "rank_8_16   8  16 32.0 all      none"
  "rank_16_32  16 32 32.0 all      none"
  # ===== LoRA alpha (scaling) =====
  "alpha_16    4  8  16.0 all      none"
  "alpha_64    4  8  64.0 all      none"
  # ===== DPO beta =====
  "beta_001    4  8  32.0 all      none --dpo-beta 0.01"
  "beta_1.0    4  8  32.0 all      none --dpo-beta 1.0"
  # ===== Target modules (surgical-inspired) =====
  "tgt_preproj 4  8  32.0 preproj  none"
  "tgt_blocks  4  8  32.0 blocks   none"
  # ===== Pair-source ablation (which reward-LoRA's rejected samples) =====
  "src_v5_only  4  8  32.0 all     v5"
  "src_v2_only  4  8  32.0 all     v2"
)

for cfg in "${CONFIGS[@]}"; do
  arr=($cfg); tag=${arr[0]}; shr=${arr[1]}; expr_=${arr[2]}; alpha=${arr[3]}; tgt=${arr[4]}; src=${arr[5]}; extra="${arr[@]:6}"
  OUT=$SWEEP/$tag
  if [ -f "$OUT/history.json" ]; then echo "skip $OUT"; continue; fi
  # Optionally subset pairs by sampler
  if [ "$src" != "none" ]; then
    SUBSET_FILE=/tmp/pref_${src}.jsonl
    grep "\"sampler\": \"$src\"" $PAIRS > $SUBSET_FILE
    USE_PAIRS=$SUBSET_FILE
    echo "  pair subset: $src ($(wc -l < $SUBSET_FILE) pairs)"
  else
    USE_PAIRS=$PAIRS
  fi
  echo "=== $tag (shared_rank=$shr expert_rank=$expr_ alpha=$alpha target=$tgt extra=$extra)"
  python3 /root/corl_work/scripts/train_with_sample_eval.py \
    --pairs $USE_PAIRS --out-dir $OUT \
    --lr 3e-5 --w-imit 0.3 \
    --shared-rank $shr --expert-rank $expr_ --lora-alpha $alpha \
    --target-modules $tgt \
    --epochs 10 --eval-every 2 --n-eval 30 --seed 42 --max-pairs 1500 \
    $extra
done
echo "=== Phase 2c done"
