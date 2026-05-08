#!/usr/bin/env python3
"""Analyze step-count sensitivity from multi-frame debug log.
Parses summary.log to extract per-step model predictions across all frames,
then computes ADE/FDE at different early-stopping points.
"""
import os
import re
import numpy as np

LOG_PATH = os.environ.get(
    "DIFFUSION_PLANNER_SUMMARY_LOG",
    os.path.expanduser("~/autoware_data/diffusion_planner/sample_frames/summary.log"),
)

# Parse all frames
frames = []  # list of dicts: {step_models: [(x,y), ...], final_xy: (x,y)}

with open(LOG_PATH, 'r') as f:
    content = f.read()

# Split by frame
frame_blocks = re.split(r'=== Frame (\d+) ===', content)
# frame_blocks: ['header', '0', 'frame0_content', '1', 'frame1_content', ...]

for i in range(1, len(frame_blocks) - 1, 2):
    frame_id = int(frame_blocks[i])
    block = frame_blocks[i + 1]

    # Extract per-step model predictions (x, y) at 8s horizon
    step_pattern = r'Step (\d+)/10.*?\| model\(x,y\)=\(([\d.]+),([\d.]+)\)m'
    steps = re.findall(step_pattern, block)

    if len(steps) < 10:
        continue

    step_models = []
    for s_num, x, y in steps:
        step_models.append((float(x), float(y)))

    # Extract final trajectory endpoint (after denoise-to-zero)
    final_match = re.search(r'final_x_t\[t=80\] \(denorm\): x=([\d.]+)m, y=([\d.]+)m', block)
    if not final_match:
        continue

    final_x = float(final_match.group(1))
    final_y = float(final_match.group(2))

    # Also get model prediction at step 10 (the last step before denoise-to-zero)
    # step_models[9] is step 10

    frames.append({
        'id': frame_id,
        'step_models': step_models,  # model x0-prediction at each step (1-10)
        'final_xy': (final_x, final_y),
    })

print(f"Parsed {len(frames)} frames")

# Compute metrics: for each "early stopping" at step k,
# the model's best guess of the 8s-ahead waypoint is step_models[k-1]
# Compare to step 10 model prediction as baseline

# For each frame, compute displacement error at step k vs step 10
# DE(k) = ||model(k) - model(10)||_2  (in meters, denormalized)
# Also compute relative error in x and y separately

print("\n=== Step-Count Sensitivity Analysis (100+ frames) ===")
print(f"{'Step':>4} {'Mean DE (m)':>11} {'Std DE (m)':>10} {'Max DE (m)':>10} "
      f"{'Mean x-err%':>11} {'Mean y-err%':>11}")

for k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    des = []
    x_errs = []
    y_errs = []
    for f in frames:
        # Baseline: step 10 model prediction (before denoise-to-zero)
        base_x, base_y = f['step_models'][9]
        # Early stop: step k model prediction
        pred_x, pred_y = f['step_models'][k - 1]

        de = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
        des.append(de)

        x_err = abs(pred_x - base_x) / abs(base_x) * 100 if base_x != 0 else 0
        y_err = abs(pred_y - base_y) / abs(base_y) * 100 if base_y != 0 else 0
        x_errs.append(x_err)
        y_errs.append(y_err)

    des = np.array(des)
    x_errs = np.array(x_errs)
    y_errs = np.array(y_errs)

    print(f"{k:>4} {des.mean():>11.3f} {des.std():>10.3f} {des.max():>10.3f} "
          f"{x_errs.mean():>11.2f} {y_errs.mean():>11.2f}")

# Physical displacement (denormalized): multiply by sigma=20m
print("\n=== Physical Displacement Error (denormalized, sigma=20m) ===")
print(f"{'Step':>4} {'Mean (m)':>10} {'Std (m)':>10} {'Max (m)':>10}")
for k in [3, 5, 7, 10]:
    des = []
    for f in frames:
        base_x, base_y = f['step_models'][9]
        pred_x, pred_y = f['step_models'][k - 1]
        # These are in normalized coords; physical = norm * 20m
        de_norm = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
        # Actually, looking at the log: model(x,y)=(100.169,3.46402)m
        # The "m" suffix and the scale (~90m for x) suggests these are ALREADY denormalized
        # Let's double check: if normalized, x would be ~4.5 (89.6/20).
        # 100.169 >> 4.5, so these are denormalized values
        des.append(de_norm)
    des = np.array(des)
    print(f"{k:>4} {des.mean():>10.2f} {des.std():>10.2f} {des.max():>10.2f}")

# Compute ADE over the trajectory: we only have the 8s-ahead waypoint per step,
# but we can compute multi-horizon metrics if we had full trajectory.
# For now, compute FDE (Final Displacement Error) at 8s horizon.
print("\n=== FDE at 8s Horizon: Early-Stop vs N=10 Baseline ===")
print(f"{'Config':>15} {'DiT Calls':>10} {'Mean FDE':>10} {'Std FDE':>10} {'Max FDE':>10}")
for k in [3, 5, 7, 10]:
    des = []
    for f in frames:
        base_x, base_y = f['step_models'][9]
        pred_x, pred_y = f['step_models'][k - 1]
        de = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
        des.append(de)
    des = np.array(des)
    dit_calls = k + 1  # k solver steps + 1 final denoise-to-zero
    print(f"{'N='+str(k)+',p=2':>15} {dit_calls:>10} {des.mean():>10.2f} {des.std():>10.2f} {des.max():>10.2f}")

# Also compute: how different is the model prediction between consecutive steps?
# This shows convergence rate
print("\n=== Per-Step Convergence (model prediction change) ===")
print(f"{'Step':>4} {'Mean delta (m)':>14} {'Std delta (m)':>14}")
for k in range(2, 11):
    deltas = []
    for f in frames:
        prev_x, prev_y = f['step_models'][k - 2]
        curr_x, curr_y = f['step_models'][k - 1]
        delta = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        deltas.append(delta)
    deltas = np.array(deltas)
    print(f"{k:>4} {deltas.mean():>14.3f} {deltas.std():>14.3f}")

# Estimate order-1 vs order-2 effect
# In order-2, the D1 correction = 0.5 * alpha_t * (1-e^{-h}) * D1
# We can't compute exact order-1 trajectory without re-running the model,
# but we can estimate the magnitude of the 2nd-order correction term
print("\n=== Summary for Paper Table ===")
print("Step | DiT Calls | Encoder Calls | FDE@8s (m) | x-err% | y-err%")
for k in [3, 5, 7, 10]:
    fdes = []
    x_errs_list = []
    y_errs_list = []
    for f in frames:
        base_x, base_y = f['step_models'][9]
        pred_x, pred_y = f['step_models'][k - 1]
        fde = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
        fdes.append(fde)
        x_err = abs(pred_x - base_x) / abs(base_x) * 100
        y_err = abs(pred_y - base_y) / abs(base_y) * 100
        x_errs_list.append(x_err)
        y_errs_list.append(y_err)
    fdes = np.array(fdes)
    x_errs = np.array(x_errs_list)
    y_errs = np.array(y_errs_list)
    print(f"N={k:>2} | {k+1:>9} | {1:>13} | {fdes.mean():>10.2f} | {x_errs.mean():>6.2f} | {y_errs.mean():>6.2f}")
