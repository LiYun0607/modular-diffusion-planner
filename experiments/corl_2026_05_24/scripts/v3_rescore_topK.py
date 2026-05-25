"""Re-evaluate top-K Phase 2 sweep variants with V3 proxy scorer on n=200 disjoint val-npz."""
import sys, glob, json, os, math, time
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
from cc_proxy_v3 import score_trajectory_v3, CCProxyConfig, aggregate_bucket_results

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP()

# Build val-npz set (disjoint from training pairs by NPZ identity)
with open('/root/corl_work/outputs/npz_val.txt') as f:
    val_npz = [l.strip() for l in f if l.strip()]
print(f'val-npz pool: {len(val_npz)} disjoint files')
# take first n=200 (deterministic; sorted = lexicographic)
import random as _rand
_rand.seed(2026); _rand.shuffle(val_npz)
EVAL_NPZ = val_npz[:200]
print(f'evaluating on n={len(EVAL_NPZ)} held-out val npz')

def fresh_model(lora_pth=None, lora_cfg=None):
    """lora_cfg: dict with shared_rank/expert_rank/lora_alpha/target_modules (optional)"""
    m = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    if lora_pth:
        sr = lora_cfg.get('shared_rank', 4) if lora_cfg else 4
        er = lora_cfg.get('expert_rank', 8) if lora_cfg else 8
        al = lora_cfg.get('lora_alpha', 32.0) if lora_cfg else 32.0
        apply_molora(m.decoder.dit, n_experts=4, shared_rank=sr, expert_rank=er, alpha=al); m.to(DEVICE)
        sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
        m.load_state_dict(sd, strict=False); set_active_expert(m, 1)
    m.eval(); return m

def eval_v3(model, npz_list):
    cfg_v3 = CCProxyConfig()
    per_traj = []
    for p in npz_list:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        r = score_trajectory_v3(ego_xy=xy, ego_v0=v0, dt=0.1, cfg=cfg_v3)
        per_traj.append(r)
    agg = aggregate_bucket_results(per_traj)
    return agg

# Load Phase 2 leaderboard, pick top 15 by best_vio + always include the catastrophic ones for comparison
results=[]
for h in sorted(glob.glob('/root/corl_work/outputs/sweep/*/history.json')):
    d = json.load(open(h)); hist = d.get('history', [])
    bv = min((x.get('vio_rate', 1.0) for x in hist if 'vio_rate' in x), default=1.0)
    name = os.path.basename(os.path.dirname(h))
    results.append((bv, name, os.path.dirname(h)))
results.sort()
# Pick top 15 + 2 worst for comparison + always include some baselines + w_0.0 collapse
to_eval = [(name, d) for _, name, d in results[:15]]
to_eval += [(name, d) for _, name, d in results[-2:] if name not in [n for n,_ in to_eval]]
print(f'\nEvaluating {len(to_eval)} configs:')
for name, _ in to_eval: print(f'  {name}')

# Map config name → LoRA rank/alpha for proper reconstruction
LORA_CFGS = {
    'rank_2_4':   {'shared_rank': 2, 'expert_rank': 4, 'lora_alpha': 16.0},
    'rank_8_16':  {'shared_rank': 8, 'expert_rank': 16, 'lora_alpha': 32.0},
    'rank_16_32': {'shared_rank': 16, 'expert_rank': 32, 'lora_alpha': 32.0},
    'alpha_16':   {'shared_rank': 4, 'expert_rank': 8, 'lora_alpha': 16.0},
    'alpha_64':   {'shared_rank': 4, 'expert_rank': 8, 'lora_alpha': 64.0},
}

all_v3 = {}
t0 = time.time()
for name, ddir in to_eval:
    # best.pth or final.pth?
    best_pth = os.path.join(ddir, 'best.pth')
    if not os.path.exists(best_pth):
        best_pth = os.path.join(ddir, 'final.pth')
    lc = LORA_CFGS.get(name, None)
    print(f'\n[{name}] loading {best_pth}')
    try:
        m = fresh_model(best_pth, lc)
        v3 = eval_v3(m, EVAL_NPZ)
        all_v3[name] = v3
        ov = v3['_overall']
        print(f'  {name:<30s}: strict={ov["strict_safety_violation_rate"]*100:5.1f}% drv={ov["driverlike_violation_rate"]*100:5.1f}% cmf={ov["comfort_violation_rate"]*100:5.1f}%')
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  FAIL {name}: {e}')
        all_v3[name] = {'_error': str(e)}

# Also include baselines: base_nolora, v5_pure, v3_dpo (if available)
print(f'\n=== baselines:')
for name, lp, lc in [
    ('base_nolora', None, None),
    ('v5_pure',     '/tmp/lora_kashiwa_v5_pure.pth', None),
]:
    try:
        m = fresh_model(lp, lc)
        v3 = eval_v3(m, EVAL_NPZ)
        all_v3[name] = v3
        ov = v3['_overall']
        print(f'  {name:<30s}: strict={ov["strict_safety_violation_rate"]*100:5.1f}% drv={ov["driverlike_violation_rate"]*100:5.1f}% cmf={ov["comfort_violation_rate"]*100:5.1f}%')
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  FAIL {name}: {e}')

json.dump(all_v3, open('/root/corl_work/outputs/v3_rescore_topK.json', 'w'), indent=2, default=float)
print(f'\n=== saved /root/corl_work/outputs/v3_rescore_topK.json (elapsed {time.time()-t0:.0f}s)')
