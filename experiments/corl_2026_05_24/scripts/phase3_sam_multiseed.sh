#!/bin/bash
# Phase 3: best Phase 2 config (sam) trained at production scale with multi-seed
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs_train_byNpz.jsonl
OUT=/root/corl_work/outputs/phase3_sam
mkdir -p $OUT
for SEED in 42 123 7; do
  D=$OUT/seed_$SEED
  if [ -f "$D/history.json" ]; then echo "skip $D (done)"; continue; fi
  echo "=== Phase 3 sam seed=$SEED ($(date))"
  python3 /root/corl_work/scripts/train_with_sample_eval.py \
    --pairs $PAIRS --out-dir $D \
    --lr 3e-5 --w-imit 0.5 --wd 1e-4 --schedule constant \
    --sam \
    --epochs 25 --eval-every 2 --n-eval 50 --seed $SEED \
    --max-pairs 3000
done
echo "=== Phase 3 done"
