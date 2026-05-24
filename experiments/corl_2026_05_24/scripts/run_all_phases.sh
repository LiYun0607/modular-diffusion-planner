#!/bin/bash
# Sequential: 2a → 2b → 2c, skip configs with history.json already present
set -e
LOG=/tmp/run_all_phases.log
echo "=== START $(date)" | tee -a $LOG
echo "=== Phase 2a at $(date)" | tee -a $LOG
bash /root/corl_work/scripts/run_dpo_sweep_v2.sh 2>&1 | tee -a $LOG
echo "=== Phase 2b at $(date)" | tee -a $LOG
bash /root/corl_work/scripts/run_dpo_sweep_2b_ad.sh 2>&1 | tee -a $LOG
echo "=== Phase 2c at $(date)" | tee -a $LOG
bash /root/corl_work/scripts/run_dpo_sweep_2c_lora.sh 2>&1 | tee -a $LOG
echo "=== ALL DONE at $(date)" | tee -a $LOG
