"""Production-scale preference pair generation:
  - ALL kashiwa training npz (~3722)
  - Sample REJECTED from 3 different reward-LoRAs (v2_jr_paper, v5_pure, v8_deep_soup)
  - More diverse rejected = more robust DPO training"""
import sys, time, glob, random, json, math, os
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '/root/corl_work/scripts')
import torch, numpy as np
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_molora import apply_molora, set_active_expert
from train_reward_backprop import differentiable_dpm_solver_sample
from utils import load_npz_data
from cc_violation_score import score_trajectory, CCConfig
from real_dpo_lora import accept_pair, PrefFilterConfig

DEVICE = torch.device('cuda')
cfg_d = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP()
cfg_cc = PrefFilterConfig(cc=CCConfig(), min_loser_severity=1, require_winner_clean=True)

def get_v0(d): return float(math.hypot(d['ego_current_state'][0,4].cpu(), d['ego_current_state'][0,5].cpu()))
def speeds(xy, v0):
    v = [v0]
    for j in range(1, len(xy)):
        v.append(float(math.hypot(xy[j][0]-xy[j-1][0], xy[j][1]-xy[j-1][1]))*10)
    return v

# Load all kashiwa npz
all_npz = []
for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
    all_npz += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
random.seed(2026); random.shuffle(all_npz)
print(f'total kashiwa npz: {len(all_npz)}')

REJECT_LORAS = [
    ('v2', '/tmp/lora_kashiwa_jr_paper_v2.pth'),
    ('v5', '/tmp/lora_kashiwa_v5_pure.pth'),
    ('v8', '/tmp/lora_kashiwa_v8_deep_greedy_soup.pth'),
]

def load_lora_model(lora_pth):
    m = Diffusion_Planner(cfg_d).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k,v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0); m.to(DEVICE)
    lsd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
    m.load_state_dict(lsd, strict=False)
    set_active_expert(m, 1); m.eval()
    return m

out_lines = []
n_total_attempts = 0; n_accepted = 0
t0 = time.time()
for lora_name, lora_path in REJECT_LORAS:
    if not os.path.exists(lora_path):
        print(f'skip {lora_name}: missing'); continue
    print(f'\n=== sampling rejected from {lora_name}:')
    model = load_lora_model(lora_path)
    for i, p in enumerate(all_npz):
        try:
            data = load_npz_data(p, DEVICE); v0 = get_v0(data)
            data_n = cfg_d.observation_normalizer(data)
            with torch.no_grad():
                traj = differentiable_dpm_solver_sample(model, cfg_d, data_n, ns, device=DEVICE).cpu().numpy()
            rej_xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
            rej_v = speeds(rej_xy, v0)
            d = np.load(p, allow_pickle=True); fut = d['ego_agent_future']
            chosen_xy = [(float(fut[j,0]), float(fut[j,1]), float(fut[j,2]), j*0.1) for j in range(len(fut))]
            chosen_v = speeds(chosen_xy, v0)
            chosen = dict(ego_xy=chosen_xy, ego_v=chosen_v, ego_v0=v0)
            rejected = dict(ego_xy=rej_xy, ego_v=rej_v, ego_v0=v0)
            ok, meta = accept_pair(chosen, rejected, cfg_cc)
            n_total_attempts += 1
            if ok:
                n_accepted += 1
                out_lines.append(json.dumps({
                    'npz': p, 'sampler': lora_name,
                    'chosen': chosen_xy, 'rejected': rej_xy,
                    'chosen_v': chosen_v, 'rejected_v': rej_v,
                    'ego_v0': v0, 'meta': meta,
                }, default=float))
        except Exception as e:
            print(f'  skip {p}: {e}'); continue
        if (i+1) % 500 == 0:
            print(f'  {i+1}/{len(all_npz)}: total_accept={n_accepted}, elapsed={time.time()-t0:.0f}s')
    del model; torch.cuda.empty_cache()

print(f'\n=== TOTAL: {n_accepted}/{n_total_attempts} pairs accepted ({n_accepted/max(n_total_attempts,1)*100:.1f}%), elapsed {time.time()-t0:.0f}s')
out_path = '/root/corl_work/outputs/large_pref_pairs.jsonl'
with open(out_path, 'w') as f:
    f.write('\n'.join(out_lines) + '\n')
print(f'Saved {n_accepted} pairs to {out_path}')
