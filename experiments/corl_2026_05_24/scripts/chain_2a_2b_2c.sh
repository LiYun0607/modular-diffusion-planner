#!/bin/bash
while pgrep -f "train_with_sample_eval.py" > /dev/null; do sleep 60; done
echo "=== Phase 2a exited, launching 2b at $(date)"; bash /root/corl_work/scripts/run_dpo_sweep_2b_ad.sh 2>&1 | tee /tmp/sweep_2b.log
while pgrep -f "train_with_sample_eval.py" > /dev/null; do sleep 60; done
echo "=== Phase 2b exited, launching 2c at $(date)"; bash /root/corl_work/scripts/run_dpo_sweep_2c_lora.sh 2>&1 | tee /tmp/sweep_2c.log
echo "=== ALL sweeps done at $(date)"
