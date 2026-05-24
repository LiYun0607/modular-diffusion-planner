"""Phase 4: final n=500 eval across all trained variants + baselines."""
import sys, glob, random, json, math, time, os
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

DEVICE = torch.device('cuda'); cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def fresh_model(lora_pth=None, zero_preproj=False):
    m = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k,v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    if lora_pth:
        apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0); m.to(DEVICE)
        sd = torch.load(lora_pth, map_location=DEVICE, weights_only=False)['model']
        m.load_state_dict(sd, strict=False); set_active_expert(m, 1)
        if zero_preproj:
            for n, mod in m.named_modules():
                if isinstance(mod, MoLoRALinear) and 'preproj.fc1' in n:
                    mod.shared_scale = 0.0; mod.expert_scale = 0.0
    m.eval(); return m

def eval_model_n500(name, model, test_npz):
    n=0; n_vio=0; jerks=[]; speeds=[]; route_devs=[]
    cc = {}
    for p in test_npz:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        v = [v0]
        for j in range(1, len(traj)):
            v.append(float(math.hypot(traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]))*10)
        speeds.append(float(np.mean(v)))
        sc = score_trajectory(ego_xy=xy, ego_v=v, ego_v0=v0, dt=0.1, cfg=cc_cfg)
        jerks.append(sc['details']['jerk_max'])
        route_devs.append(sc['details']['route_dev_max'])
        n+=1
        if sc['violation']: n_vio+=1
        for k, val in sc['violations'].items():
            cc[k] = cc.get(k, 0) + (1 if val else 0)
    return {'name':name, 'n':n,
            'any_violation':n_vio/n,
            'per_criterion':{k:v/n for k,v in cc.items()},
            'jerk_max_p50':float(np.median(jerks)),
            'jerk_max_p95':float(np.percentile(jerks,95)),
            'mean_speed':float(np.mean(speeds)),
            'route_dev_p95':float(np.percentile(route_devs,95))}

def main(out_json='/root/corl_work/outputs/final_n500_eval.json', n=500):
    all_kashiwa = []
    for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
        all_kashiwa += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
    random.seed(2026); random.shuffle(all_kashiwa)
    test = all_kashiwa[:n]
    print(f'evaluating on n={len(test)} held-out kashiwa npz')

    cands = [
        ('base_nolora', None, False),
        ('v5_pure (full LoRA)', '/tmp/lora_kashiwa_v5_pure.pth', False),
        ('v5_surgical (zero preproj.fc1)', '/tmp/lora_kashiwa_v5_pure.pth', True),
        ('v2_jr_paper', '/tmp/lora_kashiwa_jr_paper_v2.pth', False),
        ('v8_deep_soup', '/tmp/lora_kashiwa_v8_deep_greedy_soup.pth', False),
    ]
    # add all trained models from sweep + final
    for d in sorted(glob.glob('/root/corl_work/outputs/sweep/lr*/best.pth')) + sorted(glob.glob('/root/corl_work/outputs/final_*/best.pth')):
        cands.append((os.path.basename(os.path.dirname(d)), d, False))
    # also keep the early single-config models for backwards comparison
    for old in ['/root/corl_work/outputs/real_dpo_lora_v1/real_dpo_lora_v1.pth',
                 '/root/corl_work/outputs/real_dpo_lora_from_base/real_dpo_lora_v1.pth',
                 '/root/corl_work/outputs/real_dpo_l2_hybrid/real_dpo_l2_lora.pth']:
        if os.path.exists(old):
            cands.append(('prev_' + os.path.basename(os.path.dirname(old)), old, False))

    out = {}
    for name, lp, zp in cands:
        if lp and not os.path.exists(lp): continue
        t0 = time.time()
        m = fresh_model(lp, zp)
        out[name] = eval_model_n500(name, m, test)
        out[name]['elapsed'] = time.time()-t0
        print(f'{name:<50s}: vio={out[name]["any_violation"]*100:5.1f}% jerk_p50={out[name]["jerk_max_p50"]:7.2f} v_mean={out[name]["mean_speed"]:.2f} ({out[name]["elapsed"]:.0f}s)')
        del m; torch.cuda.empty_cache()
    json.dump(out, open(out_json,'w'), indent=2, default=float)
    print(f'\nSaved: {out_json}')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='/root/corl_work/outputs/final_n500_eval.json')
    p.add_argument('--n', type=int, default=500)
    a = p.parse_args()
    main(a.out, a.n)
