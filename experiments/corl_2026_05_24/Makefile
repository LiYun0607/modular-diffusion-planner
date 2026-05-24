# Makefile for CoRL 2026 — Real-Vehicle Anchored Fine-Tuning Experiments
#
# Targets:
#   all                       — run the full pipeline (most stages blocked on GPU/data)
#   analyze_4model            — re-analyze the 5-21 garage v=0 4-model parquet
#   score_5_21_full           — extract trajectories + full C&C scoring on 5-21 bags
#   score_sanity_v3           — same for the smaller sanity_v3 bags
#   extract_5_01              — extract real ego trajectories from 5-01 bag
#   build_latent_bank         — build OOD scaling latent bank (needs GPU)
#   sweep_perturbation_mock   — end-to-end demo with mock model (no GPU)
#   sweep_perturbation_real   — REAL model perturbation sweep (needs GPU+model)
#   train_real_dpo            — Real-Vehicle DPO LoRA training (needs GPU + pairs)
#   test                      — run self-tests of all scripts
#   clean                     — remove derived outputs

WORK_ROOT      ?= /root/corl_work
SCRIPTS_DIR    ?= /root/corl_work/scripts
OUT_DIR        ?= /root/corl_work/outputs
FROM_CAR_DIR   ?= /root/corl_work/from_car
PYTHON         ?= python3

# Data paths (set these per environment)
PARQUET_4M     ?= $(FROM_CAR_DIR)/field_2026-05-21_offline_ab_4model/4model.parquet
BAG_4M_ROOT    ?= $(FROM_CAR_DIR)/field_2026-05-21_offline_ab_4model
BAG_SANITY_V3  ?= $(FROM_CAR_DIR)/field_2026-05-21_offline_ab_sanity_v3
BAG_5_01_BASE  ?= $(FROM_CAR_DIR)/field_2026-05-01_R1_rep1_baseline
BAG_5_01_SE_R1 ?= $(FROM_CAR_DIR)/field_2026-05-01_R1_rep1_kashiwa_selora
BAG_5_01_SE_R2 ?= $(FROM_CAR_DIR)/field_2026-05-01_R1_rep2_kashiwa_selora

# Training paths (GPU side)
BASE_CKPT      ?= /root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth
ARGS_JSON      ?= /root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json
NPZ_KASHIWA_DIR ?= /root/autoware_ws/grpo_data/npz/kashiwa

.PHONY: all test test_cc test_ood test_pert test_real_dpo \
        analyze_4model score_5_21_full score_sanity_v3 \
        extract_5_01 build_latent_bank sweep_perturbation_mock \
        sweep_perturbation_real train_real_dpo clean help

help:
	@echo "Targets:"
	@grep -E '^[a-zA-Z_]+:.*?##' Makefile | sed 's/##/ —/'

all: test analyze_4model score_5_21_full score_sanity_v3 sweep_perturbation_mock
	@echo "[DONE] all (non-GPU) targets complete"

# === Self-tests (no data required) ===
test: test_cc test_ood test_pert test_real_dpo  ## Run all module self-tests
test_cc:    ## C&C scorer self-test
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SCRIPTS_DIR)'); from cc_violation_score import _self_test; _self_test()"
test_ood:   ## OOD scaler self-test
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SCRIPTS_DIR)'); from ood_aware_lora_scaling import _self_test; _self_test()"
test_pert:  ## Sensor perturbation self-test
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SCRIPTS_DIR)'); from sensor_perturbation import _self_test; _self_test()"
test_real_dpo:  ## Real-DPO filter self-test
	$(PYTHON) -c "import sys; sys.path.insert(0,'$(SCRIPTS_DIR)'); from real_dpo_lora import _self_test; _self_test()"

# === Analysis on car-side data already pulled ===
analyze_4model:  ## Analyze 5-21 garage 4-model parquet
	$(PYTHON) $(SCRIPTS_DIR)/analyze_4model_parquet.py

score_5_21_full:  ## Full trajectory C&C on 5-21 4-bag set (rclpy req)
	source /opt/autoware/setup.bash && source /root/autoware_ws/install/setup.bash && \
	$(PYTHON) -c "exec(open('$(SCRIPTS_DIR)/score_5_21_full.py').read())"

score_sanity_v3:  ## C&C on the smaller sanity_v3 bags
	source /opt/autoware/setup.bash && source /root/autoware_ws/install/setup.bash && \
	$(PYTHON) -c "exec(open('$(SCRIPTS_DIR)/score_sanity_v3.py').read())"

extract_5_01:  ## Extract real ego trajectories from 5-01 bag (needs full SFTP done)
	source /opt/autoware/setup.bash && source /root/autoware_ws/install/setup.bash && \
	$(PYTHON) $(SCRIPTS_DIR)/extract_real_ego_trajectories.py \
		--bag $(BAG_5_01_BASE) \
		--out $(OUT_DIR)/real_ego_trajectories_R1_rep1_baseline.jsonl

# === Mock perturbation sweep (no GPU) ===
sweep_perturbation_mock:  ## 3-model mock perturbation sweep demo
	$(PYTHON) $(SCRIPTS_DIR)/perturbation_sweep_runner.py

# === GPU-required (placeholders, document params for handoff) ===
build_latent_bank:  ## Build OOD scaling latent bank (needs GPU + base model + npz)
	$(PYTHON) $(SCRIPTS_DIR)/build_latent_bank.py \
		--base-ckpt $(BASE_CKPT) \
		--args-json $(ARGS_JSON) \
		--npz-dir   $(NPZ_KASHIWA_DIR) \
		--n-samples 200 \
		--out-npz   $(OUT_DIR)/latent_bank_kashiwa.npz \
		--out-json  $(OUT_DIR)/ood_scaler_calib.json

sweep_perturbation_real:  ## REAL model perturbation sweep (needs GPU + model wrapper)
	@echo "TODO: wrap actual model_inference_fn in perturbation_sweep_runner.py"

train_real_dpo:  ## Real-Vehicle DPO LoRA training (needs GPU + extracted pairs)
	@echo "Step 1: $(MAKE) extract_5_01     (real trajectories)"
	@echo "Step 2: $(PYTHON) $(SCRIPTS_DIR)/real_dpo_lora.py extract \\"
	@echo "          --real-bag $(BAG_5_01_BASE) --reward-lora-ckpt <path> \\"
	@echo "          --base-ckpt $(BASE_CKPT) --npz-dir <dir> --out <pairs.jsonl>"
	@echo "Step 3: $(PYTHON) $(SCRIPTS_DIR)/real_dpo_lora.py train \\"
	@echo "          --pairs <pairs.jsonl> --base-ckpt $(BASE_CKPT) --out-dir <dir>"

# === Cleanup ===
clean:  ## Remove derived outputs (NOT real-vehicle bags)
	rm -f $(OUT_DIR)/*.json $(OUT_DIR)/*.png $(OUT_DIR)/*.npz
