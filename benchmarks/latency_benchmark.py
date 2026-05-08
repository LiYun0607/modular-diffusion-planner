#!/usr/bin/env python3
"""Latency Benchmark: Measure wall-clock time for each component.

Measures T_enc, T_dit, T_solver on CPU via ONNX Runtime.
Uses captured frame data for realistic input sizes.
"""
import os
import time
import numpy as np
import onnxruntime as ort

MODEL_DIR = os.environ.get(
    "DIFFUSION_PLANNER_MODEL_DIR",
    os.path.expanduser("~/autoware_data/diffusion_planner/v2.0"),
)
LOG_DIR = os.environ.get(
    "DIFFUSION_PLANNER_LOG_DIR",
    os.path.expanduser("~/autoware_data/diffusion_planner/sample_frames"),
)

NUM_WARMUP = 10
NUM_RUNS = 100


def load_frame_tensors(frame_id=0):
    """Load real tensors from debug log + create dummy tensors for missing encoder inputs."""
    prefix = os.path.join(LOG_DIR, f"frame_{frame_id}")
    d = {}
    d['ego_current_state'] = np.fromfile(f"{prefix}_ego_current_state.bin", dtype=np.float32).reshape(1, 10)
    d['ego_agent_past'] = np.fromfile(f"{prefix}_ego_agent_past.bin", dtype=np.float32).reshape(1, 31, 4)
    d['neighbor_agents_past'] = np.fromfile(f"{prefix}_neighbor_agents_past.bin", dtype=np.float32).reshape(1, 32, 31, 11)
    d['goal_pose'] = np.fromfile(f"{prefix}_goal_pose.bin", dtype=np.float32).reshape(1, 4)
    d['lanes'] = np.fromfile(f"{prefix}_lanes.bin", dtype=np.float32).reshape(1, 140, 20, 33)
    d['route_lanes'] = np.fromfile(f"{prefix}_route_lanes.bin", dtype=np.float32).reshape(1, 25, 20, 33)
    d['context_embedding'] = np.fromfile(f"{prefix}_context_embedding.bin", dtype=np.float32).reshape(226, 1, 256)

    # Encoder needs additional inputs not saved in debug log
    # Create zero tensors with correct shapes (latency is input-independent)
    d['sampled_trajectories'] = np.zeros((1, 33, 81, 4), dtype=np.float32)
    d['static_objects'] = np.zeros((1, 5, 10), dtype=np.float32)
    d['lanes_speed_limit'] = np.zeros((1, 140, 1), dtype=np.float32)
    d['lanes_has_speed_limit'] = np.zeros((1, 140, 1), dtype=np.bool_)
    d['route_lanes_speed_limit'] = np.zeros((1, 25, 1), dtype=np.float32)
    d['route_lanes_has_speed_limit'] = np.zeros((1, 25, 1), dtype=np.bool_)
    d['polygons'] = np.zeros((1, 10, 40, 2), dtype=np.float32)
    d['line_strings'] = np.zeros((1, 10, 20, 2), dtype=np.float32)
    d['ego_shape'] = np.array([[4.89, 1.84, 1.5]], dtype=np.float32)  # typical car dimensions
    d['turn_indicators'] = np.ones((1, 31), dtype=np.float32)
    return d


def benchmark_encoder(session, tensors):
    """Benchmark context encoder."""
    inputs = {inp.name: tensors[inp.name] for inp in session.get_inputs()}
    # Warmup
    for _ in range(NUM_WARMUP):
        session.run(None, inputs)
    # Measure
    times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)  # ms
    return np.array(times)


def benchmark_dit(session, tensors):
    """Benchmark single DiT core step."""
    inputs = {
        'sampled_trajectories': tensors['sampled_trajectories'],
        'ego_current_state': tensors['ego_current_state'],
        'neighbor_agents_past': tensors['neighbor_agents_past'],
        'context_embedding': tensors['context_embedding'],
        'timestep': np.array([0.5], dtype=np.float32),
    }
    # Warmup
    for _ in range(NUM_WARMUP):
        session.run(None, inputs)
    # Measure
    times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        session.run(None, inputs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.array(times)


def benchmark_solver_overhead():
    """Benchmark pure C++ solver overhead (Python equivalent).
    This measures: VP schedule computation + coefficient computation + array update.
    """
    # Simulate solver overhead: compute coefficients + update x_t array
    x_t = np.random.randn(1, 33, 81, 4).astype(np.float32)
    x0_pred = np.random.randn(1, 33, 81, 4).astype(np.float32)

    def vp_coefficients(t_s, t):
        beta_0, beta_1 = 0.1, 20.0
        log_alpha_s = -0.25 * t_s**2 * (beta_1 - beta_0) - 0.5 * t_s * beta_0
        log_alpha_t = -0.25 * t**2 * (beta_1 - beta_0) - 0.5 * t * beta_0
        alpha_s = np.exp(log_alpha_s)
        sigma_s = np.sqrt(max(1 - alpha_s**2, 1e-12))
        alpha_t = np.exp(log_alpha_t)
        sigma_t = np.sqrt(max(1 - alpha_t**2, 1e-12))
        lam_s = log_alpha_s - 0.5 * np.log(max(1 - np.exp(2 * log_alpha_s), 1e-12))
        lam_t = log_alpha_t - 0.5 * np.log(max(1 - np.exp(2 * log_alpha_t), 1e-12))
        h = lam_t - lam_s
        return sigma_t / sigma_s, alpha_t * (1 - np.exp(-h))

    # Warmup
    for _ in range(NUM_WARMUP):
        c1, c2 = vp_coefficients(0.5, 0.3)
        _ = c1 * x_t + c2 * x0_pred

    # Measure N=10 solver steps (coefficient + array update, no DiT)
    times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        for step in range(10):
            t_s = 1.0 - step * 0.1
            t_next = t_s - 0.1
            c1, c2 = vp_coefficients(t_s, t_next)
            x_t = c1 * x_t + c2 * x0_pred
            x_t[:, :, 0, :] = 0  # initial state constraint
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return np.array(times)


def main():
    print("=" * 60)
    print("Latency Benchmark (ONNX Runtime, CPU)")
    print("=" * 60)

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = 4

    # Load models
    print("Loading models...")
    enc_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "context_encoder_v2.onnx"), opts,
        providers=['CPUExecutionProvider'])
    dit_session = ort.InferenceSession(
        os.path.join(MODEL_DIR, "dit_core.onnx"), opts,
        providers=['CPUExecutionProvider'])

    # Load frame data
    tensors = load_frame_tensors(0)

    # Benchmark each component
    print(f"\nBenchmarking encoder ({NUM_WARMUP} warmup + {NUM_RUNS} runs)...")
    enc_times = benchmark_encoder(enc_session, tensors)

    print(f"Benchmarking DiT core ({NUM_WARMUP} warmup + {NUM_RUNS} runs)...")
    dit_times = benchmark_dit(dit_session, tensors)

    print(f"Benchmarking solver overhead ({NUM_WARMUP} warmup + {NUM_RUNS} runs)...")
    sol_times = benchmark_solver_overhead()

    # Results
    T_enc = np.median(enc_times)
    T_dit = np.median(dit_times)
    T_sol = np.median(sol_times)

    print("\n" + "=" * 60)
    print("RESULTS (median over 100 runs)")
    print("=" * 60)
    print(f"T_enc (Encoder):     {T_enc:7.2f} ms  (std: {enc_times.std():.2f})")
    print(f"T_dit (DiT/step):    {T_dit:7.2f} ms  (std: {dit_times.std():.2f})")
    print(f"T_sol (Solver/10st): {T_sol:7.2f} ms  (std: {sol_times.std():.2f})")

    print(f"\n--- End-to-End Latency Estimates ---")
    for N in [3, 5, 7, 10, 15, 20]:
        # Modular: encoder once + N DiT + 1 final DiT + solver
        T_mod = T_enc + (N + 1) * T_dit + T_sol * N / 10
        # Monolithic: encoder at every step (fused graph)
        T_mono = (N + 1) * (T_enc + T_dit)
        speedup = T_mono / T_mod
        print(f"N={N:>2}: Modular={T_mod:7.1f}ms  Monolithic={T_mono:7.1f}ms  Speedup={speedup:.1f}x")

    # Save for figure generation
    out_path = os.environ.get(
        "DIFFUSION_PLANNER_OUT",
        os.path.join(os.path.dirname(__file__), "data", "latency_results.npz"),
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        T_enc=T_enc, T_dit=T_dit, T_sol=T_sol,
        enc_times=enc_times, dit_times=dit_times, sol_times=sol_times,
    )
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
