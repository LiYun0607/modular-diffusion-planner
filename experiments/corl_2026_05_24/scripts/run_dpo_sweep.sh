#!/bin/bash
# Phase 2: sweep (lr × w_imit) with sample-based eval
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs.jsonl
SWEEP=/root/corl_work/outputs/sweep
mkdir -p $SWEEP

for lr in 1e-5 3e-5 1e-4; do
  for w_imit in 0.3 0.5 0.7 0.9; do
    OUT=$SWEEP/lr${lr}_w${w_imit}
    if [ -f "$OUT/history.json" ]; then echo "skip $OUT (done)"; continue; fi
    echo "=== running lr=$lr w_imit=$w_imit"
    python3 /root/corl_work/scripts/train_with_sample_eval.py \
      --pairs $PAIRS --out-dir $OUT \
      --lr $lr --w-imit $w_imit \
      --epochs 15 --eval-every 5 --n-eval 30 --seed 42 \
      2>&1 | tee $OUT.log | grep -E "ep|init|pairs"
  done
done
echo "=== sweep done"
ls -la $SWEEP/*/history.json 2>&1 | head -20
