"""Generate paper-ready figures from all final results."""
import json, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = '/root/corl_work/outputs'

# Figure 1: per-layer LoRA ablation
data = json.load(open(f'{OUT}/per_layer_lora_ablation.json'))
layers = list(data['per_layer'].keys())
short_names = [l.replace('decoder.dit.', '').replace('.fc', '_fc').replace('.mlp', '_mlp').replace('.proj', '_proj') for l in layers]
deltas = [data['per_layer'][l]['delta'] * 100 for l in layers]
full = data['_full_lora'] * 100
all_zero = data['_all_zero'] * 100
fig, ax = plt.subplots(figsize=(11, 4.5))
colors = ['#d62728' if d < -5 else '#aaa' for d in deltas]
ax.bar(range(len(layers)), deltas, color=colors)
ax.set_xticks(range(len(layers)))
ax.set_xticklabels(short_names, rotation=80, fontsize=8)
ax.set_ylabel('Δ C&C violation rate when this LoRA layer is zeroed (pp)')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_title(f'§6.1 Per-layer v5_pure LoRA ablation (full LoRA={full:.1f}%, all-zero={all_zero:.1f}%, n=30 held-out)')
ax.grid(axis='y', alpha=0.3)
for i, d in enumerate(deltas):
    if d < -2: ax.text(i, d - 1.5, f'{d:.1f}', ha='center', fontsize=8, color='red')
fig.tight_layout()
fig.savefig(f'{OUT}/per_layer_ablation_paper_fig.png', dpi=130)
plt.close(fig)
print('Saved per_layer_ablation_paper_fig.png')

# Figure 2: 3-model perturbation sweep
data = json.load(open(f'{OUT}/perturbation_real_3model.json'))
models = list(data.keys())
all_perts = [k for k in data[models[0]] if k != '_baseline']
fig, ax = plt.subplots(figsize=(11, 5))
xs = np.arange(len(all_perts))
w = 0.8 / len(models)
colors = {'base': '#444', 'v5_pure': '#d62728', 'real_dpo': '#1f77b4'}
for i, m in enumerate(models):
    rates = [data[m][k]['mean_violation_rate'] * 100 for k in all_perts]
    offset = (i - (len(models)-1)/2) * w
    ax.bar(xs + offset, rates, width=w, label=f'{m} (baseline {data[m]["_baseline"]["mean_violation_rate"]*100:.1f}%)', color=colors.get(m, None))
ax.set_xticks(xs); ax.set_xticklabels(all_perts, rotation=70, fontsize=8)
ax.set_ylabel('C&C violation rate (%)')
ax.set_title('§6.5 Sensor perturbation: per-model violation rate per perturbation type (n=30 inputs, 2 trials each)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT}/perturbation_3model_paper_fig.png', dpi=130)
plt.close(fig)
print('Saved perturbation_3model_paper_fig.png')

# Figure 3: OOD scaling (already exists, copy)
# (done earlier as ood_scaling_paper_fig.png)
print('OOD scaling figure already exists')

# Figure 4: surgical LoRA demo (if exists)
if os.path.exists(f'{OUT}/surgical_lora_demo.json'):
    data = json.load(open(f'{OUT}/surgical_lora_demo.json'))
    names = list(data.keys())
    rates = [data[n]['any_violation']*100 for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors_list = ['#444', '#d62728', '#2ca02c', '#888']
    bars = ax.bar(range(len(names)), rates, color=colors_list[:len(names)])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=15, fontsize=9)
    ax.set_ylabel('C&C violation rate (%)')
    ax.set_title('§7.3 Surgical LoRA: zero preproj.fc1 to recover safety (n=50)')
    for i, r in enumerate(rates):
        ax.text(i, r+1, f'{r:.1f}%', ha='center')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{OUT}/surgical_lora_paper_fig.png', dpi=130)
    plt.close(fig)
    print('Saved surgical_lora_paper_fig.png')
print('done')
