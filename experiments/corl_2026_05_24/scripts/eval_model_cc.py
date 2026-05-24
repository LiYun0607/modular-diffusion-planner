"""Quick evaluator: take a (base + optional LoRA) model, sample N npz inputs,
compute C&C violation rate. Useful for §8 results table."""
import sys, glob, random, json, math, os
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
from collections import Counter

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def load_model(lora_pth=None):
    m = Diffusion_Planner(cfg).to(DEVICE)
    base_sd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    base_sd = base_sd.get('model', base_sd.get('ema_state_dict', base_sd))
    base_sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in base_sd.items()}
    m.load_state_dict(base_sd, strict=False)
    if lora_pth:
        apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0)
        m.to(DEVICE)
        sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
        m.load_state_dict(sd, strict=False)
        set_active_expert(m, 1)
    m.eval()
    return m

def eval_model(name, lora_pth, npz_paths):
    print(f'\n=== {name}')
    m = load_model(lora_pth)
    vios = Counter()
    n_total = 0; n_violation = 0
    speed_means = []
    for p in npz_paths:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(m, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        v = [v0]
        for j in range(1, len(traj)):
            dx, dy = traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]
            v.append(float(math.hypot(dx,dy))*10)
        speed_means.append(float(np.mean(v)))
        sc = score_trajectory(ego_xy=xy, ego_v=v, ego_v0=v0, dt=0.1, cfg=cc_cfg)
        n_total += 1
        if sc['violation']: n_violation += 1
        for k, val in sc['violations'].items():
            if val: vios[k] += 1
    print(f'  n={n_total} any_violation={n_violation/n_total*100:.1f}% mean_speed={np.mean(speed_means):.2f} m/s')
    for k, c in vios.most_common():
        if c > 0: print(f'    {k:<15s} {c/n_total*100:5.1f}%')
    del m
    torch.cuda.empty_cache()
    return {'name':name, 'n':n_total, 'any_violation':n_violation/n_total, 'per_criterion':{k: v/n_total for k,v in vios.items()}, 'mean_speed': float(np.mean(speed_means))}

if __name__ == '__main__':
    all_npz = []
    for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
        all_npz += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
    random.seed(789); random.shuffle(all_npz)
    test = all_npz[:50]
    print(f'evaluating on {len(test)} held-out kashiwa npz')
    out = {}
    candidates = [
        ('base_nolora', None),
        ('v5_pure (reward-LoRA)', '/tmp/lora_kashiwa_v5_pure.pth'),
        ('v2_jr_paper', '/tmp/lora_kashiwa_jr_paper_v2.pth'),
        ('real_dpo_v1_from_v5', '/root/corl_work/outputs/real_dpo_lora_v1/real_dpo_lora_v1.pth'),
    ]
    # If from_base trained, add
    from_base = '/root/corl_work/outputs/real_dpo_lora_from_base/real_dpo_lora_v1.pth'
    if os.path.exists(from_base):
        candidates.append(('real_dpo_v2_from_base', from_base))
    for name, lp in candidates:
        if lp and not os.path.exists(lp):
            print(f'skip {name}: missing {lp}'); continue
        out[name] = eval_model(name, lp, test)
    json.dump(out, open('/root/corl_work/outputs/eval_5model_cc.json','w'), indent=2, default=float)
    print(f'\nSaved: /root/corl_work/outputs/eval_5model_cc.json')
