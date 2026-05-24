"""Per-layer LoRA ablation: zero out each LoRA block in turn, measure C&C impact.

For each LoRA-applied DiT block (preproj, blocks.0...N, final_layer), zero
the LoRA contribution (set scale=0 for that block only), sample trajectories,
score C&C, compare to full-LoRA baseline."""
import sys, os, json, time, glob, random, math
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

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def get_v0(data):
    return float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))

def speeds(traj, v0):
    v = [v0]
    for j in range(1, len(traj)):
        dx, dy = traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]
        v.append(float(math.hypot(dx,dy))*10)
    return v

def main(lora_pth='/tmp/lora_kashiwa_v5_pure.pth',
         out_json='/root/corl_work/outputs/per_layer_lora_ablation.json',
         n_inputs=30):
    # Load model
    model = Diffusion_Planner(cfg).to(DEVICE)
    base_sd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    base_sd = base_sd.get('model', base_sd.get('ema_state_dict', base_sd))
    base_sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in base_sd.items()}
    model.load_state_dict(base_sd, strict=False)
    apply_molora(model.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0)
    model.to(DEVICE)
    sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
    model.load_state_dict(sd, strict=False)
    set_active_expert(model, 1)
    model.eval()

    # Inputs
    all_npz = []
    for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
        all_npz += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
    random.seed(456); random.shuffle(all_npz)
    inputs = all_npz[:n_inputs]

    # Catalog all MoLoRA modules
    lora_modules = []
    for name, mod in model.named_modules():
        if isinstance(mod, MoLoRALinear):
            lora_modules.append((name, mod))
    print(f'found {len(lora_modules)} MoLoRA modules')

    # Save original scales
    original_scales = {name: (mod.shared_scale, mod.expert_scale) for name, mod in lora_modules}

    def restore_all():
        for name, mod in lora_modules:
            mod.shared_scale, mod.expert_scale = original_scales[name]
    def zero_one(target_name):
        restore_all()
        for name, mod in lora_modules:
            if name == target_name:
                mod.shared_scale, mod.expert_scale = 0.0, 0.0

    def score_one_input(inp_path):
        data = load_npz_data(inp_path, DEVICE)
        v0 = get_v0(data)
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        return score_trajectory(ego_xy=xy, ego_v=speeds(traj, v0), ego_v0=v0, dt=0.1, cfg=cc_cfg)

    # Full LoRA baseline
    restore_all()
    base_vios = [1 if score_one_input(inp)['violation'] else 0 for inp in inputs]
    base_rate = float(np.mean(base_vios))
    print(f'\nfull LoRA baseline violation rate: {base_rate*100:.1f}%')

    # Zero all LoRA
    for name, mod in lora_modules:
        mod.shared_scale = 0.0; mod.expert_scale = 0.0
    all_zero_vios = [1 if score_one_input(inp)['violation'] else 0 for inp in inputs]
    restore_all()
    all_zero_rate = float(np.mean(all_zero_vios))
    print(f'all-zero LoRA (= base) violation rate: {all_zero_rate*100:.1f}%')

    # Ablate each layer
    results = {'_full_lora': base_rate, '_all_zero': all_zero_rate, 'per_layer': {}}
    print(f'\n=== ablating each layer:')
    for tname, _ in lora_modules:
        zero_one(tname)
        vios = [1 if score_one_input(inp)['violation'] else 0 for inp in inputs]
        rate = float(np.mean(vios))
        delta = rate - base_rate
        results['per_layer'][tname] = {'rate': rate, 'delta': delta}
        print(f'  {tname:<50s} rate={rate*100:5.1f}%  delta={delta*100:+.1f}pp')
    restore_all()
    json.dump(results, open(out_json, 'w'), indent=2, default=float)
    print(f'\nSaved: {out_json}')

if __name__ == '__main__':
    main()
