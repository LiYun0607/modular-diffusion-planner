"""Real-DPO LoRA training: chosen=real driver, rejected=reward-LoRA sample,
LoRA layers trainable, base frozen."""
import sys, os, time, json, math
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner/preference_optimization')
import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.utils.config import Config
from train_molora import apply_molora, set_active_expert, MoLoRALinear
from utils import load_npz_data

DEVICE = torch.device('cuda')
BETA = 0.1

def compute_trajectory_loss(model, data, traj_array, cfg, noise, t):
    """MSE loss between model's denoising of (current+future) noised by (noise, t) and traj_array.
    Mirrors compute_trajectory_loss from train_dpo.py."""
    B = data['ego_current_state'].shape[0]
    P = 1 + cfg.predicted_neighbor_num
    Tf = cfg.future_len
    gt_traj = torch.tensor(traj_array, dtype=torch.float32, device=DEVICE).unsqueeze(0)  # [1, T, 4]
    ego_mean = cfg.state_normalizer.mean[0].to(DEVICE)
    ego_std = cfg.state_normalizer.std[0].to(DEVICE)
    gt_norm = (gt_traj - ego_mean) / ego_std  # [1, T, 4]
    gt_future = torch.zeros(B, P, Tf, 4, device=DEVICE)
    gt_future[:, 0, :, :] = gt_norm
    ego_cur = data['ego_current_state'][:, :4]
    nbr_cur = data['neighbor_agents_past'][:, :P-1, -1, :4] if P > 1 else torch.zeros(B, 0, 4, device=DEVICE)
    cur_states = torch.cat([ego_cur[:, None], nbr_cur], dim=1)
    all_gt = torch.cat([cur_states[:, :, None, :], gt_future], dim=2)
    mean, std = VPSDE_linear().marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1]*(len(all_gt[..., 1:, :].shape)-1)))
    xT = mean + std * noise
    xT_full = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    data_clone = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    data_n = cfg.observation_normalizer(data_clone)
    inp = {**data_n, 'gt_trajectories': all_gt, 'sampled_trajectories': xT_full, 'diffusion_time': t}
    _, out = model(inp)
    if 'model_output' in out:
        pred = out['model_output'][:, 0, 1:, :]
    else:
        pred = out['prediction'][:, 0, :, :]
    return F.mse_loss(pred, gt_norm, reduction='mean')


def main(pairs_jsonl, base_pth, init_lora_pth, out_dir, n_epochs=2, lr=1e-4, max_pairs=200):
    os.makedirs(out_dir, exist_ok=True)
    cfg = Config('/root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json', guidance_fn=None)
    # Policy model with LoRA
    policy = Diffusion_Planner(cfg).to(DEVICE)
    base_sd = torch.load(base_pth, map_location=DEVICE, weights_only=False)
    base_sd = base_sd.get('model', base_sd.get('ema_state_dict', base_sd))
    base_sd = {k.replace('module.','').replace('_orig_mod.',''): v for k, v in base_sd.items()}
    policy.load_state_dict(base_sd, strict=False)
    apply_molora(policy.decoder.dit, n_experts=4, shared_rank=4, expert_rank=8, alpha=32.0)
    policy.to(DEVICE)
    if init_lora_pth and os.path.exists(init_lora_pth):
        sd = torch.load(init_lora_pth, map_location=DEVICE, weights_only=False)['model']
        policy.load_state_dict(sd, strict=False)
        print(f'  loaded init LoRA from {init_lora_pth}')
    set_active_expert(policy, 1)
    # Reference (frozen base, no LoRA modifications)
    reference = Diffusion_Planner(cfg).to(DEVICE)
    reference.load_state_dict(base_sd, strict=False)
    reference.eval()
    for p in reference.parameters(): p.requires_grad_(False)
    # Freeze base params in policy, only LoRA trainable
    lora_params = []
    for name, p in policy.named_parameters():
        if any(s in name for s in ['shared_A', 'shared_B', 'experts_A', 'experts_B']):
            p.requires_grad_(True); lora_params.append(p)
        else:
            p.requires_grad_(False)
    print(f'  trainable LoRA params: {sum(p.numel() for p in lora_params)}')
    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)

    pairs = [json.loads(l) for l in open(pairs_jsonl)]
    pairs = pairs[:max_pairs]
    print(f'  pairs: {len(pairs)}, epochs: {n_epochs}, lr: {lr}, beta: {BETA}')

    losses = []
    accs = []
    log_lines = []
    t0 = time.time()
    step = 0
    for epoch in range(n_epochs):
        np.random.seed(epoch); idx_perm = np.random.permutation(len(pairs))
        ep_loss, ep_acc = 0.0, 0.0
        for pidx in idx_perm:
            pair = pairs[pidx]
            data = load_npz_data(pair['npz'], DEVICE)
            # Trajectory format: [80, 4] expected by compute_trajectory_loss
            chosen_arr = np.array([(p[0], p[1], math.cos(p[2]), math.sin(p[2])) for p in pair['chosen']], dtype=np.float32)
            rej_arr = np.array([(p[0], p[1], math.cos(p[2]), math.sin(p[2])) for p in pair['rejected']], dtype=np.float32)
            # Sample shared noise + t
            B = data['ego_current_state'].shape[0]
            P = 1 + cfg.predicted_neighbor_num
            Tf = cfg.future_len
            noise_w = torch.randn(B, P, Tf, 4, device=DEVICE)
            noise_l = torch.randn(B, P, Tf, 4, device=DEVICE)
            t = torch.rand(B, device=DEVICE) * 0.999 + 1e-3
            # policy losses
            lw = compute_trajectory_loss(policy, data, chosen_arr, cfg, noise_w, t)
            ll = compute_trajectory_loss(policy, data, rej_arr, cfg, noise_l, t)
            with torch.no_grad():
                lrw = compute_trajectory_loss(reference, data, chosen_arr, cfg, noise_w.clone(), t)
                lrl = compute_trajectory_loss(reference, data, rej_arr, cfg, noise_l.clone(), t)
            # DPO loss: -log sigmoid(-beta * ((lw - lrw) - (ll - lrl)))
            logits = -BETA * ((lw - lrw) - (ll - lrl))
            loss = -F.logsigmoid(logits)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            opt.step()
            ep_loss += loss.item()
            acc = float(logits.item() > 0)  # 1 if winner has lower loss
            ep_acc += acc
            step += 1
        ep_loss /= len(pairs); ep_acc /= len(pairs)
        losses.append(ep_loss); accs.append(ep_acc)
        elapsed = time.time() - t0
        log_lines.append(f'epoch {epoch+1}/{n_epochs}: loss={ep_loss:.4f} acc={ep_acc:.3f} elapsed={elapsed:.0f}s')
        print(log_lines[-1])

    # Save trained LoRA
    out_pth = os.path.join(out_dir, 'real_dpo_lora_v1.pth')
    torch.save({
        'model': policy.state_dict(),
        'expert_idx': 1, 'n_steps': step, 'note': f'Real-DPO LoRA, beta={BETA}, lr={lr}, epochs={n_epochs}',
        'losses': losses, 'accs': accs,
    }, out_pth)
    print(f'\nSaved LoRA: {out_pth}')
    with open(os.path.join(out_dir, 'train.log'), 'w') as f:
        f.write('\n'.join(log_lines))
    return out_pth


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--pairs', default='/root/corl_work/outputs/real_pref_pairs_v5_pure.jsonl')
    p.add_argument('--base-pth', default='/root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth')
    p.add_argument('--init-lora-pth', default='/tmp/lora_kashiwa_v5_pure.pth', help='Warm-start from this LoRA')
    p.add_argument('--out-dir', default='/root/corl_work/outputs/real_dpo_lora_v1')
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--max-pairs', type=int, default=200)
    a = p.parse_args()
    main(a.pairs, a.base_pth, a.init_lora_pth, a.out_dir, a.epochs, a.lr, a.max_pairs)
