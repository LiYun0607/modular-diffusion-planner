#!/usr/bin/env python3
"""Generate IEEE-style figures for the ITSC 2026 paper.

Reads benchmark .npz files from ./data/ (or $DIFFUSION_PLANNER_DATA_DIR) and
the closed-loop summary log from $DIFFUSION_PLANNER_SUMMARY_LOG, and writes
PDFs to ./figures/ (or $DIFFUSION_PLANNER_FIG_DIR).
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DIFFUSION_PLANNER_DATA_DIR", os.path.join(_HERE, "data"))
FIG_DIR = os.environ.get("DIFFUSION_PLANNER_FIG_DIR", os.path.join(_HERE, "figures"))
SUMMARY_LOG = os.environ.get(
    "DIFFUSION_PLANNER_SUMMARY_LOG",
    os.path.expanduser("~/autoware_data/diffusion_planner/sample_frames/summary.log"),
)
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 10, 'axes.titlesize': 10,
    'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'pdf.fonttype': 42,   # TrueType fonts (avoid Type 3)
    'ps.fonttype': 42,
})

# ============================================================
# Figure 1: Denoising Convergence
# (a) Dual y-axis: x and y position over steps 1-10
# (b) Relative error line chart with early-stopping markers
# ============================================================
steps = np.arange(1, 11)
x_vals = np.array([100.2, 100.2, 99.6, 97.9, 95.3, 92.8, 92.4, 91.6, 90.7, 89.6])
y_vals = np.array([3.46, 3.50, 3.67, 4.11, 4.11, 3.61, 4.34, 5.12, 5.77, 6.26])
final_x, final_y = 89.6, 6.26

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 3.3))
color_x, color_y = '#1f77b4', '#ff7f0e'

# --- (a) Dual y-axis: position convergence ---
# Shaded phases
ax_a.axvspan(0.5, 2.5, alpha=0.08, color='gray', label='_')
ax_a.axvspan(2.5, 6.5, alpha=0.06, color='cornflowerblue', label='_')
ax_a.axvspan(6.5, 10.5, alpha=0.06, color='mediumseagreen', label='_')

# Left y-axis: longitudinal (x)
lx, = ax_a.plot(steps, x_vals, color_x, marker='o', ms=5, lw=1.5,
                label='Longitudinal (x)')
ax_a.axhline(final_x, color=color_x, ls='--', lw=0.8, alpha=0.5)
ax_a.set_xlabel('Denoising Step')
ax_a.set_ylabel('Longitudinal x (norm.)', color=color_x)
ax_a.tick_params(axis='y', labelcolor=color_x)
ax_a.set_xlim(0.5, 10.5)
ax_a.set_ylim(87, 103)
ax_a.set_xticks(steps)
ax_a.text(10.3, final_x + 0.6, f'x*={final_x}', fontsize=7, color=color_x,
          ha='right', va='bottom')

# Right y-axis: lateral (y)
ax_a2 = ax_a.twinx()
ly, = ax_a2.plot(steps, y_vals, color_y, marker='^', ms=5, lw=1.5, ls='--',
                 label='Lateral (y)')
ax_a2.axhline(final_y, color=color_y, ls='--', lw=0.8, alpha=0.5)
ax_a2.set_ylabel('Lateral y (norm.)', color=color_y)
ax_a2.tick_params(axis='y', labelcolor=color_y)
ax_a2.set_ylim(2.5, 7.5)
ax_a2.text(10.3, final_y + 0.2, f'y*={final_y}', fontsize=7, color=color_y,
           ha='right', va='bottom')

# Phase labels at bottom
ax_a.text(1.5, 87.5, 'Coarse', ha='center', fontsize=6.5, color='gray', style='italic')
ax_a.text(4.5, 87.5, 'Refinement', ha='center', fontsize=6.5, color='cornflowerblue', style='italic')
ax_a.text(8.5, 87.5, 'Fine Adj.', ha='center', fontsize=6.5, color='green', style='italic')

ax_a.set_title('(a) Per-Step Waypoint at 8s Horizon')
ax_a.legend(handles=[lx, ly], loc='upper right', framealpha=0.9,
            fontsize=7, handlelength=1.5)
ax_a.grid(axis='y', alpha=0.2)

# --- (b) Relative error with early-stopping annotations ---
x_err = np.abs(x_vals - final_x) / final_x * 100
y_err = np.abs(y_vals - final_y) / final_y * 100

# Shaded phases (same as panel a)
ax_b.axvspan(0.5, 2.5, alpha=0.08, color='gray')
ax_b.axvspan(2.5, 6.5, alpha=0.06, color='cornflowerblue')
ax_b.axvspan(6.5, 10.5, alpha=0.06, color='mediumseagreen')

ax_b.plot(steps, x_err, color_x, marker='o', ms=5, lw=1.5, label='Longitudinal (x)')
ax_b.plot(steps, y_err, color_y, marker='^', ms=5, lw=1.5, ls='--', label='Lateral (y)')

# Mark early-stopping points with vertical lines
# Custom text offsets to avoid overlap: (dx, dy) for each annotation
anno_offsets = {3: (0.5, 8), 5: (1.5, 8), 7: (1.5, 4)}
for n, ls_style in [(3, ':'), (5, '-.'), (7, '--')]:
    ax_b.axvline(n, color='gray', ls=ls_style, lw=0.8, alpha=0.6)
    xe = x_err[n - 1]
    dx, dy = anno_offsets[n]
    ax_b.annotate(f'N={n}: {xe:.1f}%',
                  xy=(n, xe), xytext=(n + dx, xe + dy),
                  fontsize=6.5, color=color_x,
                  arrowprops=dict(arrowstyle='->', color=color_x, lw=0.6))

ax_b.axhline(0, color='black', lw=0.4, alpha=0.3)
ax_b.set_xlabel('Denoising Step')
ax_b.set_ylabel('Relative Error to Step 10 (%)')
ax_b.set_title('(b) Convergence Rate')
ax_b.set_xticks(steps)
ax_b.set_xlim(0.5, 10.5)
ax_b.set_ylim(-2, 50)
ax_b.legend(loc='upper right', framealpha=0.9, fontsize=7, handlelength=1.5)
ax_b.grid(axis='y', alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'denoising_convergence.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved denoising_convergence.pdf")

# ============================================================
# Figure 2: Caching Comparison
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.0))
w = 0.32
color_mono = '#e07060'
color_mod = '#2a9d8f'

# Left panel: Function Calls
cats_l = ['Encoder', 'DiT Core', 'Turn Ind.']
mono_c = [11, 11, 1]
mod_c = [1, 11, 1]
xl = np.arange(len(cats_l))

bml = ax1.bar(xl - w/2, mono_c, w, label='Monolithic', color=color_mono,
              edgecolor='white', linewidth=0.5)
bmdl = ax1.bar(xl + w/2, mod_c, w, label='Modular (ours)', color=color_mod,
               edgecolor='white', linewidth=0.5)
ax1.annotate('90.9%\u2193',
             xy=(bmdl[0].get_x() + bmdl[0].get_width()/2, bmdl[0].get_height()),
             xytext=(0, 8), textcoords='offset points',
             ha='center', va='bottom', fontsize=7, fontweight='bold', color='#1a6e5f')
for bar in bml:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
             ha='center', va='bottom', fontsize=7)
for bar in bmdl:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.2, str(int(h)),
             ha='center', va='bottom', fontsize=7)
ax1.set_xticks(xl)
ax1.set_xticklabels(cats_l)
ax1.set_ylabel('Number of Calls')
ax1.set_title('Function Calls per Cycle')
ax1.set_ylim(0, 15)
ax1.legend(loc='upper right')

# Right panel: Node Executions
cats_r = ['Encoder\nNodes', 'DiT\nNodes']
mono_n = [37587, 13607]
mod_n = [3417, 13607]
xr = np.arange(len(cats_r))

bmr = ax2.bar(xr - w/2, mono_n, w, label='Monolithic', color=color_mono,
              edgecolor='white', linewidth=0.5)
bmdr = ax2.bar(xr + w/2, mod_n, w, label='Modular (ours)', color=color_mod,
               edgecolor='white', linewidth=0.5)
ax2.annotate('90.9%\u2193',
             xy=(bmdr[0].get_x() + bmdr[0].get_width()/2, bmdr[0].get_height()),
             xytext=(0, 8), textcoords='offset points',
             ha='center', va='bottom', fontsize=7, fontweight='bold', color='#1a6e5f')
for bar in bmr:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 500, f'{int(h):,}',
             ha='center', va='bottom', fontsize=7)
for bar in bmdr:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 500, f'{int(h):,}',
             ha='center', va='bottom', fontsize=7)
ax2.set_xticks(xr)
ax2.set_xticklabels(cats_r)
ax2.set_ylabel('Node Executions per Cycle')
ax2.set_title('Node Executions per Cycle')
ax2.set_ylim(0, 46000)
ax2.legend(loc='upper right')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'caching_comparison.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved caching_comparison.pdf")

# ============================================================
# Figure 3: Step-Count Sensitivity (multi-frame analysis)
# (a) FDE distribution boxplot for N=3,5,7,10
# (b) Per-step convergence: mean delta across 131 frames
# ============================================================
import re

LOG_PATH = SUMMARY_LOG
with open(LOG_PATH, 'r') as f:
    content = f.read()

frame_blocks = re.split(r'=== Frame (\d+) ===', content)
all_step_models = []  # list of lists: each inner list has 10 (x,y) tuples

for i in range(1, len(frame_blocks) - 1, 2):
    block = frame_blocks[i + 1]
    step_pattern = r'Step (\d+)/10.*?\| model\(x,y\)=\(([\d.]+),([\d.]+)\)m'
    matches = re.findall(step_pattern, block)
    if len(matches) == 10:
        step_models = [(float(x), float(y)) for _, x, y in matches]
        all_step_models.append(step_models)

n_frames = len(all_step_models)
print(f"Figure 3: Parsed {n_frames} frames for step-count sensitivity")

# (a) Boxplot: FDE distribution at different early-stopping points
fde_data = {}
for k in [3, 5, 7]:
    fdes = []
    for sm in all_step_models:
        base_x, base_y = sm[9]  # step 10 baseline
        pred_x, pred_y = sm[k - 1]
        fde = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
        fdes.append(fde)
    fde_data[k] = np.array(fdes)

# (b) Per-step convergence: mean model prediction change
mean_deltas = []
std_deltas = []
for k in range(1, 11):
    if k == 1:
        # step 1: distance from step 1 to step 10
        deltas = []
        for sm in all_step_models:
            base_x, base_y = sm[9]
            pred_x, pred_y = sm[0]
            d = np.sqrt((pred_x - base_x)**2 + (pred_y - base_y)**2)
            deltas.append(d)
    else:
        deltas = []
        for sm in all_step_models:
            prev_x, prev_y = sm[k - 2]
            curr_x, curr_y = sm[k - 1]
            d = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            deltas.append(d)
    mean_deltas.append(np.mean(deltas))
    std_deltas.append(np.std(deltas))

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 3.0))

# --- (a) Boxplot + scatter ---
bp_data = [fde_data[3], fde_data[5], fde_data[7]]
bp_labels = ['N=3\n(4 DiT calls)', 'N=5\n(6 DiT calls)', 'N=7\n(8 DiT calls)']
bp_colors = ['#e07060', '#f4a261', '#2a9d8f']

bp = ax_a.boxplot(bp_data, positions=[1, 2, 3], widths=0.5,
                  patch_artist=True, showfliers=True,
                  flierprops=dict(marker='o', ms=3, alpha=0.3),
                  medianprops=dict(color='black', lw=1.5))
for patch, color in zip(bp['boxes'], bp_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax_a.set_xticks([1, 2, 3])
ax_a.set_xticklabels(bp_labels)
ax_a.set_ylabel('FDE at 8s Horizon (m)')
ax_a.set_title('(a) Step-Count Sensitivity (131 frames)')
ax_a.grid(axis='y', alpha=0.2)

# Add mean value annotations with custom offsets to avoid overlap
anno_cfg = {3: (0.38, 1.2), 5: (0.38, 0.8), 7: (0.38, 0.5)}
for idx, k in enumerate([3, 5, 7]):
    mean_val = fde_data[k].mean()
    dx, dy = anno_cfg[k]
    ax_a.annotate(f'{mean_val:.2f}m', xy=(idx + 1, mean_val),
                  xytext=(idx + 1 + dx, mean_val + dy),
                  fontsize=7, color=bp_colors[idx], fontweight='bold',
                  arrowprops=dict(arrowstyle='->', color=bp_colors[idx], lw=0.6))

# --- (b) Per-step convergence across frames ---
steps_b = np.arange(2, 11)
mean_d = np.array(mean_deltas[1:])  # skip step 1 (FDE to baseline, not delta)
std_d = np.array(std_deltas[1:])

ax_b.bar(steps_b, mean_d, width=0.6, color='#264653', alpha=0.8,
         edgecolor='white', linewidth=0.5)
ax_b.errorbar(steps_b, mean_d, yerr=std_d, fmt='none', color='black',
              capsize=3, lw=0.8)

# Shaded phases
ax_b.axvspan(1.5, 2.5, alpha=0.08, color='gray')
ax_b.axvspan(2.5, 6.5, alpha=0.06, color='cornflowerblue')
ax_b.axvspan(6.5, 10.5, alpha=0.06, color='mediumseagreen')

ax_b.set_xlabel('Denoising Step')
ax_b.set_ylabel('Mean Step-to-Step Change (m)')
ax_b.set_title('(b) Convergence Rate (131 frames)')
ax_b.set_xticks(steps_b)
ax_b.set_xlim(1.5, 10.5)
ax_b.grid(axis='y', alpha=0.2)

# Annotate phases
ax_b.text(2, ax_b.get_ylim()[1] * 0.92, 'Coarse', ha='center', fontsize=6.5,
          color='gray', style='italic')
ax_b.text(4.5, ax_b.get_ylim()[1] * 0.92, 'Refinement', ha='center', fontsize=6.5,
          color='cornflowerblue', style='italic')
ax_b.text(8.5, ax_b.get_ylim()[1] * 0.92, 'Fine Adj.', ha='center', fontsize=6.5,
          color='green', style='italic')

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'step_sensitivity.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved step_sensitivity.pdf")

# ============================================================
# Figure 4: Solver Comparison (offline benchmark results)
# (a) FDE@8s vs N for DPM++ p=1 (=DDIM) and DPM++ p=2
# (b) ADE vs N
# ============================================================
SIGMA = 20.0  # denormalization scale

# Load benchmark results
res = np.load(os.path.join(DATA_DIR, 'solver_benchmark_results.npz'),
              allow_pickle=True)
N_vals = res['N_values']  # [3, 5, 7, 10, 15, 20]

# Extract mean FDE and ADE for each solver × N
fde_p1 = np.array([res[f'DPM++ (p=1)_{n}_fde'].mean() * SIGMA for n in N_vals])
fde_p2 = np.array([res[f'DPM++ (p=2)_{n}_fde'].mean() * SIGMA for n in N_vals])
ade_p1 = np.array([res[f'DPM++ (p=1)_{n}_ade'].mean() * SIGMA for n in N_vals])
ade_p2 = np.array([res[f'DPM++ (p=2)_{n}_ade'].mean() * SIGMA for n in N_vals])

# Std for error bands
fde_p1_std = np.array([res[f'DPM++ (p=1)_{n}_fde'].std() * SIGMA for n in N_vals])
fde_p2_std = np.array([res[f'DPM++ (p=2)_{n}_fde'].std() * SIGMA for n in N_vals])
ade_p1_std = np.array([res[f'DPM++ (p=1)_{n}_ade'].std() * SIGMA for n in N_vals])
ade_p2_std = np.array([res[f'DPM++ (p=2)_{n}_ade'].std() * SIGMA for n in N_vals])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 3.0))

c_p1 = '#e07060'   # red-ish for p=1/DDIM
c_p2 = '#2a9d8f'   # teal for p=2

# --- (a) FDE at 8s horizon ---
ax_a.plot(N_vals, fde_p1, color=c_p1, marker='o', ms=5, lw=1.5,
          label='DPM++ (p=1) / DDIM')
ax_a.fill_between(N_vals, fde_p1 - fde_p1_std, fde_p1 + fde_p1_std,
                   color=c_p1, alpha=0.15)
ax_a.plot(N_vals, fde_p2, color=c_p2, marker='s', ms=5, lw=1.5,
          label='DPM++ (p=2)')
ax_a.fill_between(N_vals, np.maximum(fde_p2 - fde_p2_std, 0), fde_p2 + fde_p2_std,
                   color=c_p2, alpha=0.15)

ax_a.set_xlabel('Denoising Steps (N)')
ax_a.set_ylabel('FDE at 8s Horizon (m)')
ax_a.set_title('(a) Final Displacement Error')
ax_a.set_xticks(N_vals)
ax_a.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax_a.grid(axis='y', alpha=0.2)
ax_a.set_ylim(bottom=0)

# Annotate key points
ax_a.annotate(f'{fde_p2[0]:.2f}m', xy=(N_vals[0], fde_p2[0]),
              xytext=(N_vals[0] + 1.5, fde_p2[0] + 0.05),
              fontsize=7, color=c_p2,
              arrowprops=dict(arrowstyle='->', color=c_p2, lw=0.6))
ax_a.annotate(f'{fde_p1[0]:.2f}m', xy=(N_vals[0], fde_p1[0]),
              xytext=(N_vals[0] + 1.5, fde_p1[0] + 0.05),
              fontsize=7, color=c_p1,
              arrowprops=dict(arrowstyle='->', color=c_p1, lw=0.6))

# --- (b) ADE over full trajectory ---
ax_b.plot(N_vals, ade_p1, color=c_p1, marker='o', ms=5, lw=1.5,
          label='DPM++ (p=1) / DDIM')
ax_b.fill_between(N_vals, ade_p1 - ade_p1_std, ade_p1 + ade_p1_std,
                   color=c_p1, alpha=0.15)
ax_b.plot(N_vals, ade_p2, color=c_p2, marker='s', ms=5, lw=1.5,
          label='DPM++ (p=2)')
ax_b.fill_between(N_vals, np.maximum(ade_p2 - ade_p2_std, 0), ade_p2 + ade_p2_std,
                   color=c_p2, alpha=0.15)

ax_b.set_xlabel('Denoising Steps (N)')
ax_b.set_ylabel('ADE over 8s Trajectory (m)')
ax_b.set_title('(b) Average Displacement Error')
ax_b.set_xticks(N_vals)
ax_b.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax_b.grid(axis='y', alpha=0.2)
ax_b.set_ylim(bottom=0)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'solver_comparison.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved solver_comparison.pdf")

# ============================================================
# Figure 5: Pareto Frontier (Latency vs FDE)
# Three curves for fair comparison:
#   1. Monolithic (p=2): encoder re-runs, same solver order
#   2. Modular (p=1/DDIM): encoder cached, first-order
#   3. Modular (p=2): encoder cached, second-order
# This separates architecture benefit (1 vs 2/3) from solver benefit (2 vs 3).
# ============================================================
lat = np.load(os.path.join(DATA_DIR, 'latency_results.npz'))
T_enc = float(lat['T_enc'])
T_dit = float(lat['T_dit'])
T_sol = float(lat['T_sol'])

N_all = np.array([3, 5, 7, 10, 15, 20])

# Latency models
lat_mod = T_enc + (N_all + 1) * T_dit + T_sol * N_all / 10   # encoder once
lat_mono = (N_all + 1) * (T_enc + T_dit)                       # encoder every step

# Three curves (FDE already loaded from solver_benchmark_results.npz)
# Curve 1: Monolithic + p=2  (same solver, but encoder re-runs → high latency)
# Curve 2: Modular + p=1     (architecture benefit, simpler solver)
# Curve 3: Modular + p=2     (architecture + solver benefit)

fig, ax = plt.subplots(1, 1, figsize=(7, 3.0))

c_mono = '#e07060'   # red for monolithic
c_mod1 = '#f4a261'   # amber for modular p=1
c_mod2 = '#2a9d8f'   # teal for modular p=2

# 100ms budget line (draw first so curves overlay)
ax.axvline(100, color='gray', ls='--', lw=1.0, alpha=0.5, zorder=1)
ax.text(103, 0.58, '100 ms\nbudget', fontsize=7, color='gray', va='top')

# Curve 1: Monolithic (p=2) — same solver order, encoder re-runs
ax.plot(lat_mono, fde_p2, color=c_mono, marker='o', ms=6, lw=1.8,
        label='Monolithic (p=2)', zorder=3)
# Curve 2: Modular (p=1/DDIM) — encoder cached, first-order
ax.plot(lat_mod, fde_p1, color=c_mod1, marker='^', ms=6, lw=1.8,
        label='Modular (p=1 / DDIM)', zorder=3)
# Curve 3: Modular (p=2) — encoder cached, second-order
ax.plot(lat_mod, fde_p2, color=c_mod2, marker='s', ms=6, lw=1.8,
        label='Modular (p=2)', zorder=3)

# Annotate selected N values on each curve
# Monolithic: N=3 and N=10
for i, N in enumerate(N_all):
    if N == 3:
        ax.annotate(f'N={N}', xy=(lat_mono[i], fde_p2[i]),
                    xytext=(lat_mono[i] + 12, fde_p2[i] + 0.04),
                    fontsize=6.5, color=c_mono,
                    arrowprops=dict(arrowstyle='->', color=c_mono, lw=0.5))
    elif N == 10:
        ax.annotate(f'N={N}', xy=(lat_mono[i], fde_p2[i]),
                    xytext=(lat_mono[i] + 12, fde_p2[i] + 0.025),
                    fontsize=6.5, color=c_mono,
                    arrowprops=dict(arrowstyle='->', color=c_mono, lw=0.5))

# Modular p=2: N=5 and N=10 (skip N=3—benefit arrows already mark it)
for i, N in enumerate(N_all):
    if N == 5:
        ax.annotate(f'N={N}', xy=(lat_mod[i], fde_p2[i]),
                    xytext=(lat_mod[i] + 12, fde_p2[i] + 0.02),
                    fontsize=6.5, color=c_mod2,
                    arrowprops=dict(arrowstyle='->', color=c_mod2, lw=0.5))
    elif N == 10:
        ax.annotate(f'N={N}', xy=(lat_mod[i], fde_p2[i]),
                    xytext=(lat_mod[i] + 12, fde_p2[i] - 0.02),
                    fontsize=6.5, color=c_mod2, va='top',
                    arrowprops=dict(arrowstyle='->', color=c_mod2, lw=0.5))

# --- Benefit dimension arrows at N=3 ---
# Caching benefit: Mono(p=2,N=3) → Mod(p=2,N=3), same solver
ax.annotate('', xy=(lat_mod[0] + 1, fde_p2[0]),
            xytext=(lat_mono[0] - 1, fde_p2[0]),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0, ls='--'))
ax.text((lat_mono[0] + lat_mod[0]) / 2, fde_p2[0] + 0.02,
        'encoder caching', fontsize=6, color='#555555', ha='center', va='bottom')

# Solver benefit: Mod(p=1,N=3) → Mod(p=2,N=3), same latency
ax.annotate('', xy=(lat_mod[0] - 2, fde_p2[0] + 0.01),
            xytext=(lat_mod[0] - 2, fde_p1[0] - 0.01),
            arrowprops=dict(arrowstyle='<->', color='#555555', lw=1.0, ls='--'))
ax.text(lat_mod[0] - 5, (fde_p1[0] + fde_p2[0]) / 2 - 0.02,
        '2nd-order\nsolver', fontsize=6, color='#555555', ha='right', va='center')

# Label the N=3 point explicitly for p=1 curve
ax.annotate('N=3', xy=(lat_mod[0], fde_p1[0]),
            xytext=(lat_mod[0] + 10, fde_p1[0] + 0.01),
            fontsize=6.5, color=c_mod1,
            arrowprops=dict(arrowstyle='->', color=c_mod1, lw=0.5))

ax.set_xlabel('Inference Latency (ms)')
ax.set_ylabel('FDE at 8s Horizon (m)')
ax.set_title('Latency--Accuracy Pareto Frontier')
ax.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax.grid(alpha=0.2)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0, top=max(fde_p1[0], fde_p2[0]) * 1.15 + 0.05)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'pareto_frontier.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved pareto_frontier.pdf")

# ============================================================
# Figure 6: Framework Overview (System Architecture)
# NOTE: The actual framework.pdf used in the paper is ITSC2026-structure-v2.pdf
# (created externally). Do NOT overwrite it by running this section.
# To skip, comment out or set SKIP_FRAMEWORK = True
SKIP_FRAMEWORK = True
if SKIP_FRAMEWORK:
    print("Skipping framework.pdf (using externally-built figures/ITSC2026-structure-v2.pdf)")
    raise SystemExit(0)
# Top:   Autoware pipeline context (upstream → Planner → downstream)
# Left:  Monolithic — encoder inside unrolled loop, all frozen
# Right: Modular   — encoder cached, C++ solver, configurable
# ============================================================
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(7.16, 5.5))
ax.set_xlim(0, 100)
ax.set_ylim(-7, 90)
ax.axis('off')

# ---- palette ----
c_enc   = '#c8daf0'
c_dit   = '#fde8c8'
c_turn  = '#d4edda'
c_gray  = '#ececec'
c_mono  = '#e07060'
c_modC  = '#2a9d8f'
c_mono_bg = '#fef5f3'
c_mod_bg  = '#f3faf9'
c_aw    = '#e8eaf6'     # Autoware upstream/downstream modules

# ---- helpers ----
def rbox(x, y, w, h, fc, ec='#444', lw=0.8, ls='-', zorder=2):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35",
                        fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=zorder)
    ax.add_patch(b)

def txt(x, y, s, **kw):
    d = dict(ha='center', va='center', fontsize=7, zorder=5)
    d.update(kw)
    ax.text(x, y, s, **d)

def arrd(x, y1, y2, **kw):
    d = dict(arrowstyle='->', color='#444', lw=0.8)
    d.update(kw)
    ax.annotate('', xy=(x, y2), xytext=(x, y1), arrowprops=d)

# ==================== TOP: Autoware Pipeline ====================
# Upstream modules
aw_mods = [(2, 'Localization'), (24, 'Perception'),
           (48, 'HD Map'), (72, 'Mission\nPlanning')]
for mx, ml in aw_mods:
    rbox(mx, 84.5, 20, 4.5, c_aw, ec='#7986cb', lw=0.6)
    txt(mx + 10, 86.75, ml, fontsize=5.5, color='#3949ab')

# Arrows from modules to ROS 2 topic bus
for mx, _ in aw_mods:
    arrd(mx + 10, 84.5, 82, color='#999', lw=0.5)
ax.plot([2, 95], [82, 82], color='#bbb', lw=0.6, ls='--', zorder=1)
txt(97, 82, 'ROS\u20092', fontsize=5, color='#bbb', ha='left')

# Arrows down to both panels
arrd(22, 82, 79.5, color='#999', lw=0.5)
arrd(76, 82, 79.5, color='#999', lw=0.5)

# ====================  LEFT: Monolithic  ====================
LCX = 22

txt(LCX, 78, '(a) Original: Monolithic', fontsize=9.5, fontweight='bold')

# input pill
rbox(10, 72, 24, 4, c_gray, ec='#aaa')
txt(LCX, 74, 'Scene Inputs', fontsize=7)
arrd(LCX, 72, 69.5)

# big ONNX outer box
rbox(3.5, 12, 37, 57, c_mono_bg, ec=c_mono, lw=1.6, ls=(0, (4, 2)), zorder=1)
txt(LCX, 67, 'diffusion_planner.onnx', fontsize=7, color=c_mono, style='italic')
txt(LCX, 65, '18,398 nodes  |  47 MB', fontsize=5.5, color='#999')

# inner loop region
rbox(6.5, 24, 31, 39, '#ffffff', ec='#aaa', lw=0.6, ls='--', zorder=1.5)
txt(LCX, 61, 'Repeated per step  (\u00d711 unrolled)', fontsize=6,
    color='#999', style='italic')

# encoder
rbox(9, 53, 26, 5.5, c_enc, ec='#6a9fd8')
txt(LCX, 55.75, 'Context Encoder', fontsize=7, fontweight='bold')
arrd(LCX, 53, 51)

# DiT block — with math formula
rbox(9, 42, 26, 8, c_dit, ec='#d4962a')
txt(LCX, 47.5, 'DiT Block', fontsize=7, fontweight='bold')
txt(LCX, 44.5, r'$\hat{x}_0^{(i)} = f_\theta(x_{t_i},\, c,\, t_i)$',
    fontsize=6.5, color='#666')
arrd(LCX, 42, 40)

# solver math — with equation reference
rbox(9, 32, 26, 7, '#e0e0e0', ec='#999')
txt(LCX, 37, 'Solver Math', fontsize=7, fontweight='bold')
txt(LCX, 34.5, r'$x_{t_{i+1}} \leftarrow \mathrm{Eq.}\,(2)$',
    fontsize=6.5, color='#666')
txt(LCX, 32.5, '(hardcoded in graph)', fontsize=5, color='#999', style='italic')

# loop-back arrow
ax.annotate('', xy=(37.5, 58.5), xytext=(37.5, 33),
            arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0,
                            connectionstyle='arc3,rad=-0.35'))

# "×11!" badge
txt(1, 55.75, '\u00d711 !', fontsize=7, fontweight='bold', color=c_mono, ha='right',
    bbox=dict(fc='#fff0ee', ec=c_mono, lw=0.5, boxstyle='round,pad=0.15'))

# frozen label
txt(LCX, 19, 'N = 11,   p = 2,   schedule', fontsize=6.5,
    color=c_mono, fontweight='bold')
txt(LCX, 16.5, 'all frozen at export time', fontsize=6, color=c_mono, style='italic')

# output
arrd(LCX, 12, 9)
rbox(10, 4.5, 24, 4, c_gray, ec='#aaa')
txt(LCX, 6.5, 'Trajectory', fontsize=7)
txt(LCX, 2, '\u2192 TensorRT (68 MB)', fontsize=5.5, color='#888', style='italic')

# ====================  RIGHT: Modular  ====================
RCX = 76

txt(RCX, 78, '(b) Proposed: Modular', fontsize=9.5, fontweight='bold')

# input pill
rbox(64, 72, 24, 4, c_gray, ec='#aaa')
txt(RCX, 74, 'Scene Inputs', fontsize=7)
arrd(RCX, 72, 69)

# Context Encoder (separate!)
rbox(58, 62, 36, 6.5, c_enc, ec='#6a9fd8', lw=1.0)
txt(RCX, 65.5, 'Context Encoder', fontsize=7.5, fontweight='bold')
txt(RCX, 63, '3,417 nodes  |  18 MB', fontsize=5.5, color='#666')
txt(95, 65.5, '1\u00d7', fontsize=9, fontweight='bold', color='#3a7fc5', ha='left')

# cache arrow
arrd(RCX, 62, 58)
txt(RCX + 8, 60, 'cached embedding', fontsize=5.5, color='#888',
    style='italic', ha='left')

# C++ Solver loop box
rbox(55, 16, 42, 41, c_mod_bg, ec=c_modC, lw=1.6, ls=(0, (4, 2)), zorder=1)
txt(RCX, 55.5, 'C++ DPM-Solver++', fontsize=7.5, color=c_modC, fontweight='bold')

# DiT Core — with math formula
rbox(61, 40, 30, 10, c_dit, ec='#d4962a', lw=1.0)
txt(RCX, 47, 'DiT Core', fontsize=7.5, fontweight='bold')
txt(RCX, 44, '1,237 nodes  |  28 MB', fontsize=5.5, color='#666')
txt(RCX, 41.5, r'$\hat{x}_0^{(i)} = f_\theta(x_{t_i},\, c,\, t_i)$',
    fontsize=6.5, color='#555')

# loop arrow
ax.annotate('', xy=(93, 50), xytext=(93, 40),
            arrowprops=dict(arrowstyle='->', color=c_modC, lw=1.2,
                            connectionstyle='arc3,rad=-0.5'))
txt(95.5, 45, '\u00d7N', fontsize=9, fontweight='bold', color=c_modC, ha='left')

# Solver equations
txt(RCX, 35, '1st order:', fontsize=5.5, color='#555', fontweight='bold')
txt(RCX, 32.5, r'$x_t = \frac{\sigma_t}{\sigma_s} x_s + \alpha_t(1-e^{-h})\hat{x}_0$',
    fontsize=6.5, color='#555')
txt(RCX, 29, '2nd order:  + correction  ' + r'$\frac{1}{2}\alpha_t(1-e^{-h})D_1$',
    fontsize=5.5, color='#555')

# configurable params
txt(RCX, 22, 'N \u2208 [1, 50]     p \u2208 {1, 2}', fontsize=6.5, color=c_modC)
txt(RCX, 19.5, '\u03b2\u2080, \u03b2\u2081  configurable at runtime', fontsize=6,
    color=c_modC, style='italic')

# arrow to turn indicator
arrd(RCX, 16, 13)

# Turn Indicator
rbox(63.5, 6.5, 25, 6, c_turn, ec='#5a9a5a', lw=0.8)
txt(RCX, 10, 'Turn Indicator', fontsize=7, fontweight='bold')
txt(RCX, 7.5, '7 nodes', fontsize=5.5, color='#666')

# output
arrd(RCX, 6.5, 4)
txt(RCX, 1.5, 'Trajectory + Turn Signal', fontsize=7)

# ==================== CENTER: GraphSurgeon ====================
mid_y = 42
ax.annotate('', xy=(53, mid_y), xytext=(42.5, mid_y),
            arrowprops=dict(arrowstyle='->', color='#666', lw=2.0))
txt(47.7, mid_y + 4.5, 'ONNX\nGraph-\nSurgeon', fontsize=5.5, color='#666',
    style='italic', ha='center', va='bottom')

# ==================== BOTTOM: Downstream ====================
# Merge outputs to downstream
arrd(22, 4.5, 0, color='#999', lw=0.5)
arrd(76, 1.5, 0, color='#999', lw=0.5)
ax.plot([22, 76], [0, 0], color='#bbb', lw=0.6, ls='--', zorder=1)
arrd(49, 0, -2.5, color='#999', lw=0.5)
rbox(26, -6.5, 46, 3.5, c_aw, ec='#7986cb', lw=0.6)
txt(49, -4.75, 'Planning Validator  \u2192  Control Module', fontsize=6, color='#3949ab')

fig.savefig(os.path.join(FIG_DIR, 'framework.pdf'),
            bbox_inches='tight', dpi=300)
plt.close(fig)
print("Saved framework.pdf")
