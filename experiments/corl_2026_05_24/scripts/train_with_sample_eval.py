"""Hybrid DPO+L2 training WITH sample-based mid-training evaluation.
Every N epochs, sample 50 trajectories from current LoRA, compute C&C violation +
jerk_p50 + mean_speed. Use this as early-stopping signal (not DPO ranking acc)."""
import sys, os, time, json, math, argparse, glob, random
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization')
sys.path.insert(0, '/root/corl_work/scripts')
import torch, numpy as np, torch.nn.functional as F
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_utils.dpm_solver_pytorch import NoiseScheduleVP
from train_molora import apply_molora, set_active_expert, MoLoRALinear
from train_reward_backprop import differentiable_dpm_solver_sample
from utils import load_npz_data
from cc_violation_score_v2 import score_trajectory_v2 as score_trajectory, CCConfigV2 as CCConfig

DEVICE = torch.device('cuda')
DEFAULT_BETA = 0.1

def compute_trajectory_loss(model, data, traj_array, cfg, noise, t):
    B = data['ego_current_state'].shape[0]; P = 1 + cfg.predicted_neighbor_num; Tf = cfg.future_len
    gt = torch.tensor(traj_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    em = cfg.state_normalizer.mean[0].to(DEVICE); es = cfg.state_normalizer.std[0].to(DEVICE)
    gt_n = (gt - em) / es
    gt_future = torch.zeros(B, P, Tf, 4, device=DEVICE); gt_future[:, 0] = gt_n
    ec = data['ego_current_state'][:, :4]
    nc = data['neighbor_agents_past'][:, :P-1, -1, :4] if P > 1 else torch.zeros(B, 0, 4, device=DEVICE)
    cs = torch.cat([ec[:, None], nc], dim=1)
    ag = torch.cat([cs[:, :, None, :], gt_future], dim=2)
    mean, std = VPSDE_linear().marginal_prob(ag[..., 1:, :], t)
    std = std.view(-1, *([1]*(len(ag[..., 1:, :].shape)-1)))
    xT = mean + std * noise
    xT_full = torch.cat([ag[:, :, :1, :], xT], dim=2)
    dc = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    dn = cfg.observation_normalizer(dc)
    inp = {**dn, 'gt_trajectories': ag, 'sampled_trajectories': xT_full, 'diffusion_time': t}
    _, out = model(inp)
    pred = out.get('model_output', out.get('prediction'))[:, 0, 1:, :] if 'model_output' in out else out['prediction'][:, 0]
    return F.mse_loss(pred, gt_n, reduction='mean')

def sample_eval(model, cfg, ns_eval, eval_npz):
    """Sample N trajectories from model, return C&C metrics."""
    cc_cfg = CCConfig.for_mode('jama_inspired')
    n_vio = 0; jerks = []; speeds = []
    for p in eval_npz:
        data = load_npz_data(p, DEVICE)
        v0 = float(math.hypot(data['ego_current_state'][0,4].cpu(), data['ego_current_state'][0,5].cpu()))
        data_n = cfg.observation_normalizer(data)
        with torch.no_grad():
            traj = differentiable_dpm_solver_sample(model, cfg, data_n, ns_eval, device=DEVICE).cpu().numpy()
        xy = [(float(traj[j,0]), float(traj[j,1]), float(traj[j,2]), j*0.1) for j in range(len(traj))]
        v = [v0]
        for j in range(1, len(traj)):
            v.append(float(math.hypot(traj[j,0]-traj[j-1,0], traj[j,1]-traj[j-1,1]))*10)
        speeds.append(float(np.mean(v)))
        sc = score_trajectory(ego_xy=xy, ego_v_provided=v, ego_v0=v0, dt=0.1, cfg=cc_cfg)
        jerks.append(sc['details']['max_jerk_mps3'])
        if sc['violation']: n_vio += 1
    return {
        'vio_rate': n_vio / len(eval_npz),
        'jerk_p50': float(np.median(jerks)),  # jerks list now uses v2 'max_jerk_mps3'
        'jerk_p95': float(np.percentile(jerks, 95)),
        'mean_speed': float(np.mean(speeds)),
    }

def train(pairs_jsonl, out_dir, lr=3e-5, w_imit=0.5, n_epochs=20, eval_every=5,
          n_eval=30, init_lora=None, seed=42, max_pairs=None, wd=1e-4, schedule='constant', sam=False,
          shared_rank=4, expert_rank=8, lora_alpha=32.0, dpo_beta=DEFAULT_BETA, target_modules='all'):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
    ns_eval = NoiseScheduleVP()

    policy = Diffusion_Planner(cfg).to(DEVICE)
    bsd = torch.load('/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth', map_location=DEVICE, weights_only=False)
    bsd = bsd.get('model', bsd.get('ema_state_dict', bsd))
    bsd = {k.replace('module.','').replace('_orig_mod.',''): v for k,v in bsd.items()}
    policy.load_state_dict(bsd, strict=False)
    # target_modules: 'all' | 'mlp' | 'preproj' — controls which DiT layers get MoLoRA
    if target_modules == 'all':
        apply_molora(policy.decoder.dit, n_experts=4, shared_rank=shared_rank, expert_rank=expert_rank, alpha=lora_alpha)
    elif target_modules == 'preproj':
        # only preproj.fc1 + fc2; skip blocks + final_layer
        policy.decoder.dit.preproj.fc1 = MoLoRALinear(policy.decoder.dit.preproj.fc1, 4, shared_rank, expert_rank, lora_alpha)
        policy.decoder.dit.preproj.fc2 = MoLoRALinear(policy.decoder.dit.preproj.fc2, 4, shared_rank, expert_rank, lora_alpha)
    elif target_modules == 'blocks':
        for blk in policy.decoder.dit.blocks:
            blk.mlp1.fc1 = MoLoRALinear(blk.mlp1.fc1, 4, shared_rank, expert_rank, lora_alpha)
            blk.mlp1.fc2 = MoLoRALinear(blk.mlp1.fc2, 4, shared_rank, expert_rank, lora_alpha)
            blk.mlp2.fc1 = MoLoRALinear(blk.mlp2.fc1, 4, shared_rank, expert_rank, lora_alpha)
            blk.mlp2.fc2 = MoLoRALinear(blk.mlp2.fc2, 4, shared_rank, expert_rank, lora_alpha)
    else:
        apply_molora(policy.decoder.dit, n_experts=4, shared_rank=shared_rank, expert_rank=expert_rank, alpha=lora_alpha)
    policy.to(DEVICE); set_active_expert(policy, 1)
    if init_lora:
        sd = torch.load(init_lora, map_location=DEVICE, weights_only=False)['model']
        policy.load_state_dict(sd, strict=False)
    reference = Diffusion_Planner(cfg).to(DEVICE)
    reference.load_state_dict(bsd, strict=False); reference.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    lora_params = [p for n, p in policy.named_parameters() if any(s in n for s in ['shared_A','shared_B','experts_A','experts_B'])]
    for n, p in policy.named_parameters():
        p.requires_grad_(any(s in n for s in ['shared_A','shared_B','experts_A','experts_B']))
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=wd)
    sched = None
    if schedule == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs * (max_pairs or 1500))
    sam_outer_opt = None
    if sam:
        try:
            from torch.optim import SGD as _SGD
        except Exception: pass
        # Simple SAM: perturb weights with epsilon * grad, recompute loss, step on the perturbed grad
        # Implemented inline below.

    pairs = [json.loads(l) for l in open(pairs_jsonl)]
    if max_pairs: pairs = pairs[:max_pairs]
    print(f'  init_lora={init_lora or "base"}  pairs={len(pairs)}  lr={lr} w_imit={w_imit} epochs={n_epochs} seed={seed}')

    # Held-out eval set: STRICTLY disjoint from training pairs (npz-identity based)
    val_npz_list_path = '/root/corl_work/outputs/npz_val.txt'
    if os.path.exists(val_npz_list_path):
        with open(val_npz_list_path) as f:
            val_npz = [l.strip() for l in f if l.strip()]
        random.seed(9999); random.shuffle(val_npz)
        eval_npz = val_npz[:n_eval]
    else:
        # legacy fallback (NOT disjoint — only for old runs)
        all_kashiwa = []
        for r in ['kashiwa_route1','kashiwa_route2','kashiwa_route3','kashiwa_route4']:
            all_kashiwa += sorted(glob.glob(f'/root/autoware_ws/grpo_data/npz_multimap/{r}/*.npz'))
        random.seed(9999); random.shuffle(all_kashiwa)
        eval_npz = all_kashiwa[:n_eval]

    history = []; best_vio = 1.0; best_epoch = -1
    t0 = time.time()
    for epoch in range(n_epochs):
        policy.train()
        np.random.seed(epoch + seed*1000)
        idx_perm = np.random.permutation(len(pairs))
        ep_l, ep_acc, ep_im = 0., 0., 0.
        for pidx in idx_perm:
            pair = pairs[pidx]
            data = load_npz_data(pair['npz'], DEVICE)
            chosen = np.array([(p[0], p[1], math.cos(p[2]), math.sin(p[2])) for p in pair['chosen']], dtype=np.float32)
            rej = np.array([(p[0], p[1], math.cos(p[2]), math.sin(p[2])) for p in pair['rejected']], dtype=np.float32)
            B = data['ego_current_state'].shape[0]; P = 1 + cfg.predicted_neighbor_num; Tf = cfg.future_len
            nw = torch.randn(B, P, Tf, 4, device=DEVICE); nl = torch.randn(B, P, Tf, 4, device=DEVICE)
            t = torch.rand(B, device=DEVICE) * 0.999 + 1e-3
            lw = compute_trajectory_loss(policy, data, chosen, cfg, nw, t)
            ll = compute_trajectory_loss(policy, data, rej, cfg, nl, t)
            with torch.no_grad():
                lrw = compute_trajectory_loss(reference, data, chosen, cfg, nw.clone(), t)
                lrl = compute_trajectory_loss(reference, data, rej, cfg, nl.clone(), t)
            logits = -dpo_beta * ((lw - lrw) - (ll - lrl))
            dpo_loss = -F.logsigmoid(logits)
            loss = (1 - w_imit) * dpo_loss + w_imit * lw
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            if sam:
                # SAM: w_t' = w_t + eps * g / ||g||; second forward at w_t'; step using g'
                with torch.no_grad():
                    eps_rho = 0.05
                    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in lora_params if p.grad is not None]) + 1e-12)
                    for p in lora_params:
                        if p.grad is None: continue
                        e = eps_rho * p.grad / (grad_norm + 1e-12)
                        p.add_(e); p._sam_perturb = e
                # Recompute loss at perturbed weights
                lw2 = compute_trajectory_loss(policy, data, chosen, cfg, nw, t)
                ll2 = compute_trajectory_loss(policy, data, rej, cfg, nl, t)
                logits2 = -dpo_beta * ((lw2 - lrw) - (ll2 - lrl))
                loss2 = (1 - w_imit) * (-F.logsigmoid(logits2)) + w_imit * lw2
                opt.zero_grad(); loss2.backward()
                # Undo perturbation
                with torch.no_grad():
                    for p in lora_params:
                        if hasattr(p, '_sam_perturb'):
                            p.sub_(p._sam_perturb); del p._sam_perturb
                torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            if sched is not None: sched.step()
            ep_l += loss.item(); ep_acc += float(logits.item() > 0); ep_im += lw.item()
        ep_l /= len(pairs); ep_acc /= len(pairs); ep_im /= len(pairs)

        elapsed = time.time() - t0
        if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
            policy.eval()
            eval_m = sample_eval(policy, cfg, ns_eval, eval_npz)
            line = f'  ep {epoch+1:3d}: train_loss={ep_l:.3f} dpo_acc={ep_acc:.3f} imit={ep_im:.4f} | sample vio={eval_m["vio_rate"]*100:5.1f}% jerk_p50={eval_m["jerk_p50"]:.1f} v_mean={eval_m["mean_speed"]:.2f} | {elapsed:.0f}s'
            history.append({'epoch':epoch+1, 'train_loss':ep_l, 'dpo_acc':ep_acc, 'imit_loss':ep_im,
                            **eval_m, 'elapsed':elapsed})
            if eval_m['vio_rate'] < best_vio:
                best_vio = eval_m['vio_rate']; best_epoch = epoch+1
                torch.save({'model': policy.state_dict(), 'expert_idx':1, 'epoch':epoch+1,
                            'eval': eval_m, 'history': history,
                            'note': f'best-vio checkpoint @ ep{epoch+1}'},
                           os.path.join(out_dir, 'best.pth'))
            print(line)
        else:
            print(f'  ep {epoch+1:3d}: train_loss={ep_l:.3f} dpo_acc={ep_acc:.3f} | {elapsed:.0f}s')
    # save final + history
    torch.save({'model': policy.state_dict(), 'expert_idx':1, 'history':history,
                'config':{'lr':lr, 'w_imit':w_imit, 'epochs':n_epochs, 'seed':seed, 'pairs':len(pairs)}},
               os.path.join(out_dir, 'final.pth'))
    json.dump({'config':{'lr':lr, 'w_imit':w_imit, 'epochs':n_epochs, 'seed':seed, 'pairs':len(pairs),
                          'init':init_lora}, 'history':history,
               'best_vio':best_vio, 'best_epoch':best_epoch}, open(os.path.join(out_dir, 'history.json'),'w'), indent=2)
    return best_vio, best_epoch

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='/root/corl_work/outputs/large_pref_pairs.jsonl')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--w-imit', type=float, default=0.5)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--n-eval', type=int, default=30)
    ap.add_argument('--init-lora', default=None)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-pairs', type=int, default=None)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--schedule', choices=['constant','cosine'], default='constant')
    ap.add_argument('--sam', action='store_true', help='use sharpness-aware minimization')
    ap.add_argument('--shared-rank', type=int, default=4)
    ap.add_argument('--expert-rank', type=int, default=8)
    ap.add_argument('--lora-alpha', type=float, default=32.0)
    ap.add_argument('--dpo-beta', type=float, default=DEFAULT_BETA)
    ap.add_argument('--target-modules', choices=['all','preproj','blocks'], default='all')
    a = ap.parse_args()
    train(a.pairs, a.out_dir, a.lr, a.w_imit, a.epochs, a.eval_every, a.n_eval, a.init_lora, a.seed, a.max_pairs,
          wd=a.wd, schedule=a.schedule, sam=a.sam,
          shared_rank=a.shared_rank, expert_rank=a.expert_rank, lora_alpha=a.lora_alpha,
          dpo_beta=a.dpo_beta, target_modules=a.target_modules)
