"""Wortsman 2022 §5.1 greedy soup over Phase 2 sweep variants.

Procedure:
  1. Load all best.pth + final.pth files from sweep
  2. Sort variants by their best vio_rate (lower is better)
  3. Greedy add: start with the single best LoRA
     For each candidate in sorted order:
       - Average its LoRA params into the current soup (uniform mean)
       - Evaluate on n_eval held-out kashiwa npz (vio_rate, jerk_p50)
       - Accept if vio_rate < current soup vio_rate
     This is the homogeneous-reward soup (all variants trained with same
     DPO+L2 loss, different (lr, w_imit)) — avoids the §6.4 v8 failure mode
     where mixing progress_cap caused acceleration through turns.
"""
import sys, os, json, glob, math, time, random
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

DEVICE = torch.device('cuda')
cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
ns = NoiseScheduleVP(); cc_cfg = CCConfig()

def fresh_molora_model():
    m = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k,v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0); m.to(DEVICE)
    set_active_expert(m, 1); m.eval()
    return m

LORA_KEYS = ['shared_A.weight', 'shared_B.weight', 'experts_A.weight', 'experts_B.weight']

def load_lora_only(pth):
    """Extract only LoRA weights from a saved checkpoint."""
    sd = torch.load(pth, map_location=DEVICE, weights_only=False)['model']
    out = {k: v.clone() for k, v in sd.items() if any(s in k for s in LORA_KEYS)}
    return out

def avg_loras(lora_dicts):
    """Uniform average of N LoRA dicts."""
    if len(lora_dicts) == 1: return {k: v.clone() for k, v in lora_dicts[0].items()}
    out = {}
    for k in lora_dicts[0]:
        out[k] = sum(d[k] for d in lora_dicts) / len(lora_dicts)
    return out

def apply_lora_to_model(model, lora_dict):
    """Load LoRA dict into model (only the LoRA keys)."""
    sd = model.state_dict()
    for k, v in lora_dict.items():
        if k in sd: sd[k] = v
    model.load_state_dict(sd)

def eval_model(model, eval_npz):
    n_vio = 0; jerks = []; speeds = []
    for p in eval_npz:
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
        if sc['violation']: n_vio += 1
    return {'vio_rate': n_vio/len(eval_npz),
            'jerk_p50': float(np.median(jerks)),
            'mean_speed': float(np.mean(speeds))}

def main(sweep_dir='/root/corl_work/outputs/sweep',
         out_json='/root/corl_work/outputs/greedy_soup_dpo.json',
         n_eval=50):
    # Gather all variant pths from sweep
    variants = []
    for d in sorted(glob.glob(f'{sweep_dir}/lr*')):
        for tag, fname in [('best','best.pth'), ('final','final.pth')]:
            pth = os.path.join(d, fname)
            if os.path.exists(pth):
                # Also load history for initial eval
                h = json.load(open(os.path.join(d, 'history.json')))
                # find the matching history entry for this checkpoint
                if tag == 'best':
                    name = f'{os.path.basename(d)}_best'
                    eval_metric = json.load(open(pth.replace('.pth','.json'))) if os.path.exists(pth.replace('.pth','.json')) else None
                    # use the best vio from history
                    bv = min((x.get('vio_rate', 1.0) for x in h['history'] if 'vio_rate' in x), default=1.0)
                else:
                    name = f'{os.path.basename(d)}_final'
                    bv = h['history'][-1].get('vio_rate', 1.0) if h['history'] else 1.0
                variants.append({'name':name, 'pth':pth, 'sweep_vio':bv})
    if not variants:
        print('no sweep variants'); return
    print(f'found {len(variants)} variants')
    variants.sort(key=lambda v: v['sweep_vio'])

    # Held-out kashiwa npz for soup-add evaluation
    all_kashiwa = []
    for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
        all_kashiwa += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
    random.seed(8888); random.shuffle(all_kashiwa)
    eval_set = all_kashiwa[:n_eval]
    print(f'eval set: {len(eval_set)} kashiwa npz (seed=8888)')

    model = fresh_molora_model()
    # Step 1: seed soup with the best variant
    seed_v = variants[0]
    seed_lora = load_lora_only(seed_v['pth'])
    apply_lora_to_model(model, seed_lora)
    seed_eval = eval_model(model, eval_set)
    print(f'\nseed (best variant): {seed_v["name"]} → vio={seed_eval["vio_rate"]*100:.1f}% jerk_p50={seed_eval["jerk_p50"]:.2f}')

    soup_loras = [seed_lora]
    soup_eval = seed_eval
    soup_curve = [{'step':0, 'added':seed_v['name'], 'soup_vio':seed_eval['vio_rate'], 'soup_jerk':seed_eval['jerk_p50'], 'n_in_soup':1}]

    # Step 2: greedy add
    for i, cand in enumerate(variants[1:], start=1):
        cand_lora = load_lora_only(cand['pth'])
        trial = avg_loras(soup_loras + [cand_lora])
        apply_lora_to_model(model, trial)
        trial_eval = eval_model(model, eval_set)
        accept = trial_eval['vio_rate'] < soup_eval['vio_rate']
        soup_curve.append({'step':i, 'candidate':cand['name'], 'trial_vio':trial_eval['vio_rate'], 'trial_jerk':trial_eval['jerk_p50'],
                           'soup_vio_before':soup_eval['vio_rate'], 'accept':accept,
                           'soup_size_after': len(soup_loras)+1 if accept else len(soup_loras)})
        if accept:
            soup_loras.append(cand_lora)
            soup_eval = trial_eval
            print(f'  +{cand["name"]:<40s} ACCEPT vio={trial_eval["vio_rate"]*100:5.1f}% (now n={len(soup_loras)})')
        else:
            print(f'  +{cand["name"]:<40s} reject vio={trial_eval["vio_rate"]*100:5.1f}% (vs current soup {soup_eval["vio_rate"]*100:.1f}%)')

    # Save final soup
    final_soup = avg_loras(soup_loras)
    out_dir = '/root/corl_work/outputs/greedy_soup_dpo'
    os.makedirs(out_dir, exist_ok=True)
    # save as a full state dict (LoRA-only) for later eval
    apply_lora_to_model(model, final_soup)
    torch.save({'model': model.state_dict(), 'expert_idx':1, 'n_in_soup':len(soup_loras),
                'note':'greedy soup of DPO+L2 LoRAs', 'curve':soup_curve},
               os.path.join(out_dir, 'soup_final.pth'))
    json.dump({'n_variants_total':len(variants), 'n_in_soup':len(soup_loras),
               'final_soup_eval':soup_eval, 'soup_curve':soup_curve}, open(out_json,'w'), indent=2, default=float)
    print(f'\nfinal soup: {len(soup_loras)} variants merged')
    print(f'  vio={soup_eval["vio_rate"]*100:.1f}% jerk_p50={soup_eval["jerk_p50"]:.2f}')
    print(f'  saved {out_json}')

if __name__ == '__main__':
    main()
