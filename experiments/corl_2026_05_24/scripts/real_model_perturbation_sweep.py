"""Real Diffusion-Planner perturbation sweep.
For each model in {base, v5_pure, real_dpo}, applies JAMA-inspired sensor
perturbations to N test inputs, measures C&C violation rate."""
import sys, os, json, time, glob, random, math
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
from sensor_perturbation import default_perturbation_grid, freeze_perception, occlude_sector

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def load_model(lora_pth=None):
    """Load base model, optionally apply MoLoRA + load LoRA weights."""
    m = Diffusion_Planner(cfg).to(DEVICE)
    base_sd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth',
                          map_location=DEVICE, weights_only=False)
    base_sd = base_sd.get('model', base_sd.get('ema_state_dict', base_sd))
    base_sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in base_sd.items()}
    m.load_state_dict(base_sd, strict=False)
    if lora_pth:
        apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0)
        m.to(DEVICE)
        lora_sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
        m.load_state_dict(lora_sd, strict=False)
        set_active_expert(m, 1)
    m.eval()
    return m

def get_v0(data):
    return float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))

def speeds_from_xy(traj, v0):
    v = [v0]
    for j in range(1, len(traj)):
        dx, dy = traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]
        v.append(float(math.hypot(dx, dy))*10)
    return v

def sample_and_score(model, data, v0):
    data_n = cfg.observation_normalizer(data)
    with torch.no_grad():
        traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
    xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
    v = speeds_from_xy(traj, v0)
    return score_trajectory(ego_xy=xy, ego_v=v, ego_v0=v0, dt=0.1, cfg=cc_cfg)

def apply_perturbation_to_tensor_dict(data, op_fn, params):
    """Apply perturbation op (operates on np dict) to a torch dict, returning torch dict."""
    if 'neighbor_agents_past' not in data:
        return data
    np_dict = {'neighbor_agents_past': data['neighbor_agents_past'][0].cpu().numpy()}
    out = op_fn(np_dict, **params)
    d_out = {k: v.clone() for k, v in data.items()}
    d_out['neighbor_agents_past'] = torch.from_numpy(out['neighbor_agents_past']).unsqueeze(0).to(DEVICE)
    return d_out

def main(out_json='/root/corl_work/outputs/perturbation_real_3model.json',
         n_inputs=30, n_trials=2):
    # Test inputs (kashiwa held-out)
    all_npz = []
    for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
        all_npz += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
    random.seed(123); random.shuffle(all_npz)
    inputs = all_npz[:n_inputs]
    grid = default_perturbation_grid()

    models = {
        'base':    load_model(None),
        'v5_pure': load_model('/tmp/lora_kashiwa_v5_pure.pth'),
    }
    # If real-dpo training done, add it
    dpo_path = '/root/corl_work/outputs/real_dpo_lora_v1/real_dpo_lora_v1.pth'
    if os.path.exists(dpo_path):
        models['real_dpo'] = load_model(dpo_path)
    print(f'models: {list(models.keys())}, inputs: {n_inputs}, perturbations: {len(grid)}, trials: {n_trials}')

    results = {}
    t0 = time.time()
    for mname, model in models.items():
        print(f'\n=== sweeping {mname}:')
        res = {}
        # baseline (no perturbation)
        base_vios = []
        for inp_path in inputs:
            data = load_npz_data(inp_path, DEVICE)
            v0 = get_v0(data)
            sc = sample_and_score(model, data, v0)
            base_vios.append(1 if sc['violation'] else 0)
        base_rate = np.mean(base_vios)
        res['_baseline'] = {'mean_violation_rate': float(base_rate)}
        print(f'  baseline: {base_rate*100:.1f}%')
        # perturbation sweep
        for spec in grid:
            trial_rates = []
            for trial in range(n_trials):
                v = []
                for inp_path in inputs:
                    data = load_npz_data(inp_path, DEVICE)
                    v0 = get_v0(data)
                    params = dict(spec.params)
                    if 'seed' not in params and spec.operator not in (freeze_perception, occlude_sector):
                        params['seed'] = trial
                    pdata = apply_perturbation_to_tensor_dict(data, spec.operator, params)
                    sc = sample_and_score(model, pdata, v0)
                    v.append(1 if sc['violation'] else 0)
                trial_rates.append(float(np.mean(v)))
            mean_r = float(np.mean(trial_rates))
            res[spec.name] = {'mean_violation_rate': mean_r, 'delta': mean_r - base_rate}
        worst = sorted([(k, v['mean_violation_rate']) for k, v in res.items() if k != '_baseline'], key=lambda x: -x[1])[:3]
        print(f'  worst-3: {[(k, f"{r*100:.0f}%") for k, r in worst]}')
        results[mname] = res
    json.dump(results, open(out_json, 'w'), indent=2, default=float)
    print(f'\nSaved: {out_json}  (elapsed {time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
