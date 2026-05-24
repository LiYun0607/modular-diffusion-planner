#!/bin/bash
# Wait for Phase 2a sweep process to exit, then launch Phase 2b
while pgrep -f "train_with_sample_eval.py" > /dev/null; do
  sleep 60
done
echo "=== Phase 2a process exited, launching Phase 2b at $(date)"
bash /root/corl_work/scripts/run_dpo_sweep_v2b.sh 2>&1 | tee /tmp/sweep_2b.log
