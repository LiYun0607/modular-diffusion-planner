"""Plot sweep results: lr × w_imit grid, color = violation rate."""
import json, glob, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = '/root/corl_work/outputs/sweep'
out_files = sorted(glob.glob(f'{SWEEP}/lr*/history.json'))
if not out_files:
    print('no sweep results'); exit()

# Parse configs from path
data = []
for f in out_files:
    h = json.load(open(f))
    c = h['config']
    last_eval = next((x for x in reversed(h['history']) if 'vio_rate' in x), None)
    best_idx = min(range(len(h['history'])), key=lambda i: h['history'][i].get('vio_rate', 1.0))
    best = h['history'][best_idx]
    data.append({
        'lr': c['lr'], 'w_imit': c['w_imit'],
        'best_vio': best.get('vio_rate', 1.0),
        'best_jerk_p50': best.get('jerk_p50', 999),
        'best_mean_v': best.get('mean_speed', 0),
        'best_epoch': best['epoch'],
        'final_vio': last_eval['vio_rate'] if last_eval else 1.0,
        'final_dpo_acc': last_eval.get('dpo_acc', 0) if last_eval else 0,
    })

# Pivot into grids
lrs = sorted(set(d['lr'] for d in data))
ws = sorted(set(d['w_imit'] for d in data))
vio_grid = np.zeros((len(lrs), len(ws)))
jerk_grid = np.zeros((len(lrs), len(ws)))
for d in data:
    i = lrs.index(d['lr']); j = ws.index(d['w_imit'])
    vio_grid[i, j] = d['best_vio']
    jerk_grid[i, j] = d['best_jerk_p50']

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for ax, grid, title, cmap in [(axs[0], vio_grid*100, 'Best violation rate (%)', 'RdYlGn_r'),
                              (axs[1], np.log10(np.maximum(jerk_grid, 1)), 'log10(best jerk_p50 m/s³)', 'viridis')]:
    im = ax.imshow(grid, cmap=cmap, aspect='auto')
    ax.set_xticks(range(len(ws))); ax.set_xticklabels([f'{w}' for w in ws])
    ax.set_yticks(range(len(lrs))); ax.set_yticklabels([f'{lr:.0e}' for lr in lrs])
    ax.set_xlabel('w_imit'); ax.set_ylabel('lr')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    for i in range(len(lrs)):
        for j in range(len(ws)):
            ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center', color='white' if grid[i,j] > grid.mean() else 'black', fontsize=9)
fig.suptitle(f'Phase 2 sweep: (lr × w_imit), best-of-training, n_eval=30')
fig.tight_layout()
fig.savefig('/root/corl_work/outputs/sweep_heatmap.png', dpi=130)
plt.close(fig)
print(f'saved sweep_heatmap.png')

# Best config
best = min(data, key=lambda d: d['best_vio'])
print(f'\nBest config: lr={best["lr"]} w_imit={best["w_imit"]} → best_vio={best["best_vio"]*100:.1f}% at epoch {best["best_epoch"]} jerk_p50={best["best_jerk_p50"]:.2f}')
json.dump({'all':data, 'best':best}, open('/root/corl_work/outputs/sweep_summary.json','w'), indent=2, default=float)
