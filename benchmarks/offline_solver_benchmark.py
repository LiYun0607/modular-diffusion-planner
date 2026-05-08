#!/usr/bin/env python3
"""Offline Solver Benchmark: DPM-Solver++ (order 1, 2) and DDIM.

Replays captured AWSIM frames through different solver configurations
using the split ONNX models (context_encoder + dit_core) via onnxruntime.

Outputs: per-config FDE/ADE statistics for paper Table + Figure.
"""
import os
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ─── Paths ───────────────────────────────────────────────────────────
MODEL_DIR = os.environ.get(
    "DIFFUSION_PLANNER_MODEL_DIR",
    os.path.expanduser("~/autoware_data/diffusion_planner/v2.0"),
)
LOG_DIR = os.environ.get(
    "DIFFUSION_PLANNER_LOG_DIR",
    os.path.expanduser("~/autoware_data/diffusion_planner/sample_frames"),
)

ENCODER_PATH = os.path.join(MODEL_DIR, "context_encoder_v2.onnx")
DIT_PATH = os.path.join(MODEL_DIR, "dit_core.onnx")

# ─── Constants ───────────────────────────────────────────────────────
NUM_AGENTS = 33
SEQ_LEN = 81        # 0..80 (0=current, 80=8s ahead at 10Hz)
STATE_DIM = 4       # x, y, cos_theta, sin_theta
BETA_0 = 0.1
BETA_1 = 20.0


# ====================================================================
# VP Noise Schedule (matches C++ VPNoiseSchedule)
# ====================================================================
class VPNoiseSchedule:
    def __init__(self, num_steps: int, beta_0=BETA_0, beta_1=BETA_1):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.num_steps = num_steps
        self.timesteps = self._compute_timesteps(num_steps)

    def log_alpha(self, t: float) -> float:
        return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

    def alpha(self, t: float) -> float:
        return np.exp(self.log_alpha(t))

    def sigma(self, t: float) -> float:
        a = self.alpha(t)
        return np.sqrt(max(1.0 - a**2, 1e-12))

    def lam(self, t: float) -> float:
        """log-SNR: lambda(t) = log(alpha/sigma)"""
        return self.log_alpha(t) - 0.5 * np.log(max(1.0 - np.exp(2.0 * self.log_alpha(t)), 1e-12))

    def inverse_lambda(self, lam_val: float) -> float:
        """Binary search: find t such that lambda(t) = lam_val"""
        lo, hi = 1e-6, 1.0 - 1e-6
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if self.lam(mid) < lam_val:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2.0

    def _compute_timesteps(self, N: int) -> List[float]:
        """logSNR-uniform timestep schedule with N+1 points."""
        t_max = 1.0 - 1e-3   # avoid t=1 singularity
        t_min = 1e-3
        lam_max = self.lam(t_min)  # high SNR (small t)
        lam_min = self.lam(t_max)  # low SNR (large t)

        # Uniform spacing in lambda from lam_min to lam_max
        lams = np.linspace(lam_min, lam_max, N + 1)
        ts = [self.inverse_lambda(l) for l in lams]
        return ts


# ====================================================================
# Solvers
# ====================================================================
def dpm_solver_first_order_update(
    x_t: np.ndarray, x0_pred: np.ndarray,
    sched: VPNoiseSchedule, t_s: float, t: float
) -> np.ndarray:
    sigma_s = sched.sigma(t_s)
    alpha_t = sched.alpha(t)
    sigma_t = sched.sigma(t)
    h = sched.lam(t) - sched.lam(t_s)
    coef_x = sigma_t / sigma_s
    coef_m = alpha_t * (1.0 - np.exp(-h))
    return coef_x * x_t + coef_m * x0_pred


def dpm_solver_second_order_update(
    x_t: np.ndarray, x0_curr: np.ndarray, x0_prev: np.ndarray,
    sched: VPNoiseSchedule, t_prev: float, t_s: float, t: float
) -> np.ndarray:
    sigma_s = sched.sigma(t_s)
    alpha_t = sched.alpha(t)
    sigma_t = sched.sigma(t)
    h = sched.lam(t) - sched.lam(t_s)
    h_prev = sched.lam(t_s) - sched.lam(t_prev)
    r = h_prev / h
    coef_x = sigma_t / sigma_s
    coef_m = alpha_t * (1.0 - np.exp(-h))
    D1 = (x0_curr - x0_prev) / r
    return coef_x * x_t + coef_m * x0_curr + 0.5 * coef_m * D1


def ddim_update(
    x_t: np.ndarray, x0_pred: np.ndarray,
    sched: VPNoiseSchedule, t_s: float, t: float
) -> np.ndarray:
    """DDIM deterministic update (eta=0)."""
    alpha_s = sched.alpha(t_s)
    sigma_s = sched.sigma(t_s)
    alpha_t = sched.alpha(t)
    sigma_t = sched.sigma(t)
    # epsilon prediction from data prediction:  eps = (x_t - alpha_s * x0) / sigma_s
    eps = (x_t - alpha_s * x0_pred) / sigma_s
    return alpha_t * x0_pred + sigma_t * eps


# ====================================================================
# Initial state constraint
# ====================================================================
def apply_initial_state_constraint(x_t: np.ndarray, current_states: np.ndarray):
    """Set t=0 slice of each agent to current_states. In-place."""
    # x_t: [33, 81, 4], current_states: [33, 4]
    x_t[:, 0, :] = current_states


# ====================================================================
# Load frame data
# ====================================================================
def load_frame(frame_id: int) -> Dict[str, np.ndarray]:
    """Load saved tensors for a single frame."""
    prefix = os.path.join(LOG_DIR, f"frame_{frame_id}")
    data = {}
    data['ego_current_state'] = np.fromfile(f"{prefix}_ego_current_state.bin", dtype=np.float32).reshape(1, 10)
    data['ego_agent_past'] = np.fromfile(f"{prefix}_ego_agent_past.bin", dtype=np.float32).reshape(1, 31, 4)
    data['neighbor_agents_past'] = np.fromfile(f"{prefix}_neighbor_agents_past.bin", dtype=np.float32).reshape(1, 32, 31, 11)
    data['goal_pose'] = np.fromfile(f"{prefix}_goal_pose.bin", dtype=np.float32).reshape(1, 4)
    data['lanes'] = np.fromfile(f"{prefix}_lanes.bin", dtype=np.float32).reshape(1, 140, 20, 33)
    data['route_lanes'] = np.fromfile(f"{prefix}_route_lanes.bin", dtype=np.float32).reshape(1, 25, 20, 33)
    data['context_embedding'] = np.fromfile(f"{prefix}_context_embedding.bin", dtype=np.float32).reshape(226, 1, 256)
    data['output_ref'] = np.fromfile(f"{prefix}_output.bin", dtype=np.float32).reshape(NUM_AGENTS, SEQ_LEN - 1, STATE_DIM)
    return data


def build_current_states(ego_state: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """Build [33, 4] current_states array."""
    cs = np.zeros((NUM_AGENTS, STATE_DIM), dtype=np.float32)
    cs[0] = ego_state[0, :STATE_DIM]
    for n in range(32):
        cs[n + 1] = neighbors[0, n, 30, :STATE_DIM]  # last timestep
    return cs


# ====================================================================
# Run solver
# ====================================================================
def run_solver(
    dit_session: ort.InferenceSession,
    context_embedding: np.ndarray,
    ego_current_state: np.ndarray,
    neighbor_agents_past: np.ndarray,
    current_states: np.ndarray,
    N: int,
    solver: str,  # "dpm1", "dpm2", "ddim"
) -> np.ndarray:
    """Run denoising with specified solver and return final x0 prediction [33, 81, 4]."""
    sched = VPNoiseSchedule(N)

    # Initial noise: zeros (temperature=0)
    x_t = np.zeros((1, NUM_AGENTS, SEQ_LEN, STATE_DIM), dtype=np.float32)
    # Apply initial state constraint
    x_t[0, :, 0, :] = current_states

    model_prev = None
    t_prev = None

    for step in range(N):
        t_current = sched.timesteps[step]
        t_next = sched.timesteps[step + 1]

        # Run DiT core
        dit_inputs = {
            'sampled_trajectories': x_t,
            'ego_current_state': ego_current_state,
            'neighbor_agents_past': neighbor_agents_past,
            'context_embedding': context_embedding,
            'timestep': np.array([t_current], dtype=np.float32),
        }
        raw_output = dit_session.run(None, dit_inputs)[0]  # [1, 33, 324] or similar
        # Reshape to [1, 33, 81, 4]
        x0_pred = raw_output.reshape(1, NUM_AGENTS, SEQ_LEN, STATE_DIM)

        # Solver update
        if solver == "dpm1":
            x_t = dpm_solver_first_order_update(x_t, x0_pred, sched, t_current, t_next)
        elif solver == "dpm2":
            if step == 0 or model_prev is None:
                x_t = dpm_solver_first_order_update(x_t, x0_pred, sched, t_current, t_next)
            else:
                x_t = dpm_solver_second_order_update(
                    x_t, x0_pred, model_prev, sched, t_prev, t_current, t_next)
        elif solver == "ddim":
            x_t = ddim_update(x_t, x0_pred, sched, t_current, t_next)

        # Apply initial state constraint
        x_t[0, :, 0, :] = current_states

        t_prev = t_current
        model_prev = x0_pred.copy()

    # Final denoise-to-zero
    t_0 = sched.timesteps[N]
    dit_inputs['sampled_trajectories'] = x_t
    dit_inputs['timestep'] = np.array([t_0], dtype=np.float32)
    raw_output = dit_session.run(None, dit_inputs)[0]
    x0_final = raw_output.reshape(1, NUM_AGENTS, SEQ_LEN, STATE_DIM)

    return x0_final[0]  # [33, 81, 4]


# ====================================================================
# Metrics
# ====================================================================
def compute_ego_metrics(pred: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """Compare ego trajectory (agent 0). pred: [33,81,4], ref: [33,80,4]."""
    # Use timesteps 1..80 (pred has 81 steps: 0=current, 1..80=future)
    # ref has 80 steps (1..80)
    pred_ego = pred[0, 1:, :2]    # [80, 2] (x, y)
    ref_ego = ref[0, :, :2]       # [80, 2] (x, y)

    displacements = np.sqrt(np.sum((pred_ego - ref_ego)**2, axis=1))  # [80]
    ade = float(np.mean(displacements))
    fde = float(displacements[-1])  # t=80 = 8s horizon

    return {'ade': ade, 'fde': fde}


# ====================================================================
# Main
# ====================================================================
def main():
    print("=" * 70)
    print("Offline Solver Benchmark")
    print("=" * 70)

    # Load DiT model
    print("Loading DiT core model...")
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    dit_session = ort.InferenceSession(DIT_PATH, opts, providers=['CPUExecutionProvider'])

    # Find available frames
    frame_ids = sorted(set(
        int(f.split('_')[1])
        for f in os.listdir(LOG_DIR)
        if f.startswith("frame_") and f.endswith("_context_embedding.bin")
    ))
    print(f"Found {len(frame_ids)} frames")

    # Use up to 50 frames for benchmark (balance speed vs statistics)
    max_frames = 50
    if len(frame_ids) > max_frames:
        # Sample evenly
        indices = np.linspace(0, len(frame_ids) - 1, max_frames, dtype=int)
        frame_ids = [frame_ids[i] for i in indices]
    print(f"Using {len(frame_ids)} frames for benchmark")

    # Solver configurations
    N_values = [3, 5, 7, 10, 15, 20]
    solvers = [
        ("DPM++ (p=1)", "dpm1"),
        ("DPM++ (p=2)", "dpm2"),
        ("DDIM", "ddim"),
    ]

    # First, get baseline (N=10, p=2) for each frame
    print("\nRunning baseline (N=10, DPM++ p=2) for all frames...")
    baselines = {}
    for i, fid in enumerate(frame_ids):
        frame = load_frame(fid)
        cs = build_current_states(frame['ego_current_state'], frame['neighbor_agents_past'])
        x0 = run_solver(
            dit_session, frame['context_embedding'],
            frame['ego_current_state'], frame['neighbor_agents_past'],
            cs, N=10, solver="dpm2"
        )
        baselines[fid] = x0
        if (i + 1) % 10 == 0:
            print(f"  Baseline: {i + 1}/{len(frame_ids)} frames")

    # Verify baseline matches saved output
    frame0 = load_frame(frame_ids[0])
    ref_out = frame0['output_ref']  # [33, 80, 4]
    base0 = baselines[frame_ids[0]]  # [33, 81, 4]
    max_err = np.max(np.abs(base0[0, 1:, :] - ref_out[0, :, :]))
    print(f"Baseline vs saved output max error (ego): {max_err:.6f}")

    # Run all configurations
    results = {}  # (solver_name, N) -> {'ade': [...], 'fde': [...]}

    for solver_name, solver_key in solvers:
        for N in N_values:
            config_key = (solver_name, N)
            ade_list = []
            fde_list = []

            print(f"\nRunning {solver_name}, N={N}...")
            for i, fid in enumerate(frame_ids):
                frame = load_frame(fid)
                cs = build_current_states(frame['ego_current_state'], frame['neighbor_agents_past'])
                x0 = run_solver(
                    dit_session, frame['context_embedding'],
                    frame['ego_current_state'], frame['neighbor_agents_past'],
                    cs, N=N, solver=solver_key
                )
                # Compare to baseline (N=10, p=2)
                ref = baselines[fid]
                pred_ego = x0[0, 1:, :2]
                ref_ego = ref[0, 1:, :2]
                displacements = np.sqrt(np.sum((pred_ego - ref_ego)**2, axis=1))
                ade = float(np.mean(displacements))
                fde = float(displacements[-1])

                ade_list.append(ade)
                fde_list.append(fde)

                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{len(frame_ids)} frames")

            results[config_key] = {
                'ade': np.array(ade_list),
                'fde': np.array(fde_list),
            }

    # ─── Print Results ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS: FDE at 8s Horizon (vs N=10 DPM++ p=2 baseline)")
    print("=" * 70)
    header = f"{'Solver':>15} {'N':>4} {'DiT':>5} {'FDE mean':>10} {'FDE std':>9} {'FDE max':>9} {'ADE mean':>10}"
    print(header)
    print("-" * len(header))
    for solver_name, _ in solvers:
        for N in N_values:
            key = (solver_name, N)
            r = results[key]
            dit_calls = N + 1
            print(f"{solver_name:>15} {N:>4} {dit_calls:>5} "
                  f"{r['fde'].mean():>10.3f} {r['fde'].std():>9.3f} {r['fde'].max():>9.3f} "
                  f"{r['ade'].mean():>10.3f}")

    # Save results for figure generation
    out_path = os.environ.get(
        "DIFFUSION_PLANNER_SOLVER_OUT",
        os.path.join(os.path.dirname(__file__), "data", "solver_benchmark_results.npz"),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        N_values=np.array(N_values),
        solver_names=[s[0] for s in solvers],
        **{f"{s[0]}_{N}_fde": results[(s[0], N)]['fde'] for s in solvers for N in N_values},
        **{f"{s[0]}_{N}_ade": results[(s[0], N)]['ade'] for s in solvers for N in N_values},
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
