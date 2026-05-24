#!/bin/bash
# Phase 2a coarse sweep: 6 configs (2 lr × 3 w_imit), 10 epochs, 1500 pairs each
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs_train.jsonl
SWEEP=/root/corl_work/outputs/sweep
mkdir -p $SWEEP
for lr in 3e-5 1e-4; do
  for w_imit in 0.3 0.5 0.7; do
    OUT=$SWEEP/lr${lr}_w${w_imit}
    if [ -f "$OUT/history.json" ]; then echo "skip $OUT (done)"; continue; fi
    echo "=== running lr=$lr w_imit=$w_imit"
    python3 /root/corl_work/scripts/train_with_sample_eval.py \
      --pairs $PAIRS --out-dir $OUT \
      --lr $lr --w-imit $w_imit \
      --epochs 10 --eval-every 2 --n-eval 30 --seed 42 --max-pairs 1500
  done
done
echo "=== sweep done"
