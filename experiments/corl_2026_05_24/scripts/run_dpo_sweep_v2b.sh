#!/bin/bash
# Phase 2b: cover the missing Soup-paper axes (wd, schedule, seed, SAM, init, pure-loss variants)
set -e
PAIRS=/root/corl_work/outputs/large_pref_pairs_train.jsonl
SWEEP=/root/corl_work/outputs/sweep

# Format: "lr w_imit wd schedule sam seed init tag"
# (sam=1 means use --sam, init=v5 means warm start)
CONFIGS=(
  "3e-5 0.5 1e-3 constant 0 42 base wd1e-3"        # wd axis
  "3e-5 0.5 1e-5 constant 0 42 base wd1e-5"        # wd axis
  "3e-5 0.5 1e-4 cosine   0 42 base sched_cosine"  # schedule axis
  "3e-5 0.5 1e-4 constant 1 42 base sam"            # SAM axis (Soup paper used 4 SAM)
  "3e-5 0.5 1e-4 constant 0 123 base seed_123"     # seed
  "3e-5 0.5 1e-4 constant 0 7 base seed_7"         # seed
  "3e-5 0.5 1e-4 constant 0 42 v5 init_v5"         # init=v5 warm
  "1e-5 0.7 1e-4 constant 0 42 base lr1e-5_w0.7"   # very slow lr + high imit
  "1e-4 0.9 1e-4 constant 0 42 base lr1e-4_w0.9"   # pure-imitation-leaning
  "3e-5 0.1 1e-4 constant 0 42 base w0.1"          # DPO-leaning
  "3e-5 1.0 1e-4 constant 0 42 base w1.0_pure_imit" # pure imitation (no DPO)
  "3e-5 0.0 1e-4 constant 0 42 base w0.0_pure_dpo"  # pure DPO (no L2)
)

for cfg in "${CONFIGS[@]}"; do
  read lr w_imit wd sched sam seed init tag <<< "$cfg"
  TAG="${tag}"
  OUT=$SWEEP/$TAG
  if [ -f "$OUT/history.json" ]; then echo "skip $OUT"; continue; fi
  INIT=""
  [ "$init" = "v5" ] && INIT="--init-lora /tmp/lora_kashiwa_v5_pure.pth"
  SAM=""
  [ "$sam" = "1" ] && SAM="--sam"
  echo "=== running TAG=$TAG (lr=$lr w_imit=$w_imit wd=$wd sched=$sched sam=$sam seed=$seed init=$init)"
  python3 /root/corl_work/scripts/train_with_sample_eval.py \
    --pairs $PAIRS --out-dir $OUT \
    --lr $lr --w-imit $w_imit --wd $wd --schedule $sched \
    --epochs 10 --eval-every 2 --n-eval 30 --seed $seed --max-pairs 1500 \
    $INIT $SAM
done
echo "=== Phase 2b done"
