"""§7 surgical method: keep most of v5_pure LoRA, only suppress preproj.fc1
(the layer per-layer ablation identified as the 43pp damage source).

Compares 4 configurations on 50 held-out kashiwa npz:
  base:        no LoRA
  v5_pure:     full v5_pure LoRA
  v5_zero_preproj_fc1:  v5_pure with preproj.fc1 LoRA zeroed  ← surgical
  v5_keep_only_preproj_fc1: only preproj.fc1 active (rest zeroed) — sanity check
"""
import sys, glob, random, json, math
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
from cc_violation_score import score_trajectory, CCConfig
from collections import Counter

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def build_model():
    m = Diffusion_Planner(cfg).to(DEVICE)
    base_sd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    base_sd = base_sd.get('model', base_sd.get('ema_state_dict', base_sd))
    base_sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in base_sd.items()}
    m.load_state_dict(base_sd, strict=False)
    apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0)
    m.to(DEVICE)
    sd = torch.load('/tmp/lora_kashiwa_v5_pure.pth', map_location=DEVICE, weights_only=False)['model']
    m.load_state_dict(sd, strict=False)
    set_active_expert(m, 1); m.eval()
    return m

def set_lora_scale(model, configurations):
    """configurations: {layer_name: (shared_scale, expert_scale)}"""
    for name, mod in model.named_modules():
        if isinstance(mod, MoLoRALinear):
            if name in configurations:
                mod.shared_scale, mod.expert_scale = configurations[name]
            else:
                mod.shared_scale, mod.expert_scale = 1.0, 1.0

# Test inputs (different seed from training+ablation)
all_npz = []
for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
    all_npz += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
random.seed(789); random.shuffle(all_npz)
test = all_npz[:50]
print(f'testing on {len(test)} held-out kashiwa npz')

model = build_model()
all_lora_names = [n for n, m in model.named_modules() if isinstance(m, MoLoRALinear)]
print(f'LoRA layers: {len(all_lora_names)}')

# Build "base"-style baseline: zero all lora
def eval_config(name, configs):
    set_lora_scale(model, configs)
    vios = Counter(); n = 0; n_vio = 0
    for p in test:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        v = [v0]
        for j in range(1, len(traj)):
            dx, dy = traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]
            v.append(float(math.hypot(dx,dy))*10)
        sc = score_trajectory(ego_xy=xy, ego_v=v, ego_v0=v0, dt=0.1, cfg=cc_cfg)
        n += 1
        if sc['violation']: n_vio += 1
        for k, val in sc['violations'].items():
            if val: vios[k] += 1
    rate = n_vio / n
    print(f'  {name}: any={rate*100:.1f}% (n={n})')
    for k, c in vios.most_common():
        if c > 0: print(f'    {k}: {c/n*100:.1f}%')
    return {'name':name, 'any_violation':rate, 'per_criterion':{k:v/n for k,v in vios.items()}}

results = {}
results['base (zero all)']     = eval_config('base (zero all)',     {n: (0.0, 0.0) for n in all_lora_names})
results['full v5_pure']        = eval_config('full v5_pure',        {})
results['v5_zero_preproj_fc1'] = eval_config('v5_zero_preproj_fc1', {'decoder.dit.preproj.fc1': (0.0, 0.0)})
results['v5_keep_only_preproj_fc1'] = eval_config('v5_keep_only_preproj_fc1', {n: (0.0, 0.0) for n in all_lora_names if n != 'decoder.dit.preproj.fc1'})

json.dump(results, open('/root/corl_work/outputs/surgical_lora_demo.json','w'), indent=2, default=float)
print('\nSaved: /root/corl_work/outputs/surgical_lora_demo.json')
