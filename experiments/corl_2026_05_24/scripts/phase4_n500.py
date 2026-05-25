"""Phase 4: n=500 final eval under V3 proxy (3 layers + scenario bucket)
across all candidate models including v3_dpo (March production baseline)."""
import sys, glob, json, os, math, time
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '/root/corl_work/scripts')
import torch, numpy as np
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_molora import apply_molora, set_active_expert, MoLoRALinear
from train_reward_backprop import differentiable_dpm_solver_sample
from utils import load_npz_data
from cc_proxy_v3 import score_trajectory_v3, CCProxyConfig, aggregate_bucket_results

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP()

LORA_CFGS = {
    'rank_2_4':   {'shared_rank': 2, 'expert_rank': 4, 'lora_alpha': 16.0},
    'rank_8_16':  {'shared_rank': 8, 'expert_rank': 16, 'lora_alpha': 32.0},
    'rank_16_32': {'shared_rank': 16, 'expert_rank': 32, 'lora_alpha': 32.0},
    'alpha_16':   {'shared_rank': 4, 'expert_rank': 8, 'lora_alpha': 16.0},
    'alpha_64':   {'shared_rank': 4, 'expert_rank': 8, 'lora_alpha': 64.0},
}

def fresh_model(lora_pth=None, lora_cfg=None, zero_preproj=False):
    m = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    if lora_pth:
        sr = lora_cfg.get('shared_rank', 4) if lora_cfg else 4
        er = lora_cfg.get('expert_rank', 8) if lora_cfg else 8
        al = lora_cfg.get('lora_alpha', 32.0) if lora_cfg else 32.0
        apply_molora(m.decoder.dit, n_experts=4, shared_rank=sr, expert_rank=er, alpha=al)
        m.to(DEVICE)
        sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
        m.load_state_dict(sd, strict=False)
        set_active_expert(m, 1)
        if zero_preproj:
            for n, mod in m.named_modules():
                if isinstance(mod, MoLoRALinear) and 'preproj.fc1' in n:
                    mod.shared_scale = 0.0; mod.expert_scale = 0.0
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
        per_traj.append(score_trajectory_v3(ego_xy=xy, ego_v0=v0, dt=0.1, cfg=cfg_v3))
    return aggregate_bucket_results(per_traj)

# Load eval set: 500 disjoint val-npz (more than Phase 2's 100)
with open('/root/corl_work/outputs/npz_val.txt') as f:
    val_npz = [l.strip() for l in f if l.strip()]
import random as _rand
_rand.seed(2026); _rand.shuffle(val_npz)
N_EVAL = min(500, len(val_npz))
EVAL_NPZ = val_npz[:N_EVAL]
print(f'n_eval = {N_EVAL} disjoint val-npz')

# Candidates: baselines + best Phase 2 + Phase 3 + soup
CANDS = [
    ('base_nolora',            None, None, False),
    ('v5_pure',                '/tmp/lora_kashiwa_v5_pure.pth', None, False),
    ('v5_surgical_zero_preproj','/tmp/lora_kashiwa_v5_pure.pth', None, True),
    ('v2_jr_paper',            '/tmp/lora_kashiwa_jr_paper_v2.pth', None, False),
    ('v8_deep_soup',           '/tmp/lora_kashiwa_v8_deep_greedy_soup.pth', None, False),
    # Phase 2 best individual variants
    ('phase2_sam',             '/root/corl_work/outputs/sweep/sam/best.pth', None, False),
    ('phase2_w0.9_mostly_imit','/root/corl_work/outputs/sweep/w0.9_mostly_imit/best.pth', None, False),
    ('phase2_beta_001',        '/root/corl_work/outputs/sweep/beta_001/best.pth', None, False),
    ('phase2_tgt_blocks',      '/root/corl_work/outputs/sweep/tgt_blocks/best.pth', None, False),
    ('phase2_alpha_64',        '/root/corl_work/outputs/sweep/alpha_64/best.pth', LORA_CFGS['alpha_64'], False),
    # Phase 2 worst (for paper §6 collapse demo)
    ('phase2_w0.0_pure_dpo',   '/root/corl_work/outputs/sweep/w0.0_pure_dpo/final.pth', None, False),
    # Phase 3
    ('phase3_sam_seed42_best', '/root/corl_work/outputs/phase3_sam/seed_42/best.pth', None, False),
    ('phase3_sam_seed42_final','/root/corl_work/outputs/phase3_sam/seed_42/final.pth', None, False),
    # Soup
    ('soup_v3_comfort',        '/root/corl_work/outputs/greedy_soup_v3_comfort/soup_final.pth', None, False),
]

results = {}
t0 = time.time()
for name, lp, lc, zp in CANDS:
    if lp and not os.path.exists(lp):
        print(f'  skip {name}: missing {lp}'); continue
    print(f'\n[{name}] loading...')
    try:
        m = fresh_model(lp, lc, zp)
        v3 = eval_v3(m, EVAL_NPZ)
        ov = v3['_overall']
        results[name] = v3
        print(f'  {name:<30s} strict={ov["strict_safety_violation_rate"]*100:5.1f}% drv={ov["driverlike_violation_rate"]*100:5.1f}% cmf={ov["comfort_violation_rate"]*100:5.1f}% (took {time.time()-t0:.0f}s)')
        for b, st in v3.items():
            if b == '_overall': continue
            print(f'    bucket={b:<22s} n={st["n"]:3d} strict={st["strict_safety_violation_rate"]*100:5.1f}% drv={st["driverlike_violation_rate"]*100:5.1f}% cmf={st["comfort_violation_rate"]*100:5.1f}%')
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'  FAIL: {e}')
        results[name] = {'_error': str(e)}

json.dump(results, open('/root/corl_work/outputs/phase4_n500_v3.json', 'w'), indent=2, default=float)
print(f'\n=== saved /root/corl_work/outputs/phase4_n500_v3.json (elapsed {time.time()-t0:.0f}s)')
