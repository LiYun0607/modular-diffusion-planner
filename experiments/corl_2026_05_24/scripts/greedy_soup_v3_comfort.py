"""Greedy soup over Phase 2 variants using COMFORT layer as optimization metric.
Skips structurally-incompatible variants (target-module variants, rank variants)."""
import sys, os, glob, json, math, random, copy
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '/root/corl_work/scripts')
import torch
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

def fresh_molora():
    m = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k,v in bsd.items()}
    m.load_state_dict(bsd, strict=False)
    apply_molora(m.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0); m.to(DEVICE)
    set_active_expert(m, 1); m.eval(); return m

def load_lora(pth):
    sd = torch.load(pth, map_location=DEVICE, weights_only=False)['model']
    return {k: v.clone() for k, v in sd.items() if any(s in k for s in ['shared_A','shared_B','experts_A','experts_B'])}

def avg(lds):
    keys = set(lds[0])
    for d in lds[1:]:
        keys = keys.intersection(set(d))
    return {k: sum(d[k] for d in lds) / len(lds) for k in keys}

def apply_lora(model, ld):
    sd = model.state_dict()
    for k, v in ld.items():
        if k in sd: sd[k] = v
    model.load_state_dict(sd)

def score(model, npz_list):
    cfg_v3 = CCProxyConfig()
    per = []
    for p in npz_list:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        per.append(score_trajectory_v3(ego_xy=xy, ego_v0=v0, dt=0.1, cfg=cfg_v3))
    agg = aggregate_bucket_results(per)
    return agg['_overall']['comfort_violation_rate'], agg

# Eval pool
with open('/root/corl_work/outputs/npz_val.txt') as f:
    val = [l.strip() for l in f if l.strip()]
random.seed(2026); random.shuffle(val)
eval_npz = val[:100]
print(f'soup eval n={len(eval_npz)}')

# Eligible variants — homogeneous default rank 4/8 alpha 32
EXCLUDE = {'rank_2_4','rank_8_16','rank_16_32','alpha_16','alpha_64','tgt_preproj','tgt_blocks','w0.0_pure_dpo'}
results=[]
for h in sorted(glob.glob('/root/corl_work/outputs/sweep/*/history.json')):
    n = os.path.basename(os.path.dirname(h))
    if n in EXCLUDE: continue
    d = json.load(open(h)); hist = d.get('history', [])
    bv = min((x.get('vio_rate', 1.0) for x in hist if 'vio_rate' in x), default=1.0)
    pth = os.path.join(os.path.dirname(h), 'best.pth')
    if not os.path.exists(pth): pth = os.path.join(os.path.dirname(h), 'final.pth')
    results.append((bv, n, pth))
results.sort()
TOP_K = 12
cand = results[:TOP_K]
print(f'\nsoup candidates (top {TOP_K}, comfort-optimized):')
for bv, n, _ in cand: print(f'  {n}: v2_vio={bv*100:.1f}%')

# Build base
base = fresh_molora()
loras = [(n, load_lora(p)) for _, n, p in cand]

# Individual comfort
print('\n=== individual comfort eval:')
indiv = []
for n, ld in loras:
    apply_lora(base, ld)
    c, agg = score(base, eval_npz)
    indiv.append((c, n, ld, agg))
    print(f'  {n:<28s} comfort={c*100:5.1f}%')
indiv.sort()

# Greedy soup (try every accept that lowers comfort)
soup = [indiv[0][2]]
apply_lora(base, soup[0])
soup_c, soup_agg = score(base, eval_npz)
print(f'\n=== greedy soup (seed: {indiv[0][1]} comfort={soup_c*100:.1f}%):')
curve = [{'step':0, 'added':indiv[0][1], 'soup_comfort':soup_c, 'n_in_soup':1}]
for i, (cc, name, ld, _) in enumerate(indiv[1:], 1):
    trial = avg(soup + [ld])
    apply_lora(base, trial)
    tc, _ = score(base, eval_npz)
    accept = tc < soup_c - 1e-4
    if accept:
        soup.append(ld); soup_c = tc
        print(f'  +{name:<28s} trial={tc*100:5.2f}%  ACCEPT (soup n={len(soup)})')
    else:
        print(f'  +{name:<28s} trial={tc*100:5.2f}%  reject (soup {soup_c*100:.2f}%)')
    curve.append({'step':i,'cand':name,'trial':tc,'accept':accept,'soup_n':len(soup)})

# Final soup
final = avg(soup)
apply_lora(base, final)
final_c, final_agg = score(base, eval_npz)
out_dir = '/root/corl_work/outputs/greedy_soup_v3_comfort'
os.makedirs(out_dir, exist_ok=True)
torch.save({'model': base.state_dict(), 'expert_idx':1, 'n_in_soup':len(soup), 'curve':curve},
           os.path.join(out_dir, 'soup_final.pth'))
json.dump({'n_in_soup': len(soup), 'final_comfort': final_c, 'final_agg': final_agg,
           'individual_comfort': [(n, c, agg['_overall']) for c,n,_,agg in indiv],
           'curve': curve}, open(os.path.join(out_dir, 'soup_summary.json'),'w'), indent=2, default=float)
print(f'\n=== final soup: n={len(soup)} comfort={final_c*100:.2f}%')
print(f'    strict={final_agg["_overall"]["strict_safety_violation_rate"]*100:.2f}%')
print(f'    saved {out_dir}/')
