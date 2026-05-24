"""Build the encoder-latent training-distribution bank for OOD scaling (§7.2).

For each of N (default 200) randomly-sampled training-set npz scenes, runs
the FROZEN base Diffusion-Planner context encoder and saves the latent
vector. Auto-calibrates (tau, sigma) for the OODScaler from the bank's
internal NN-distance distribution.

Usage:
  python build_latent_bank.py \\
    --base-ckpt /root/autoware_ws/scripts/train/Diffusion-Planner/best_model.pth \\
    --args-json /root/autoware_data/diffusion_planner/v3.0/diffusion_planner.param.json \\
    --npz-dir   /root/autoware_ws/grpo_data/npz/kashiwa \\
    --n-samples 200 \\
    --out-npz   /root/corl_work/outputs/latent_bank_kashiwa.npz \\
    --out-json  /root/corl_work/outputs/ood_scaler_calib.json

Output:
  latent_bank_kashiwa.npz: {'Z': [N, D], 'ids': [N]}
  ood_scaler_calib.json: {'tau': float, 'sigma': float, 'metric': 'l2', 'floor_alpha': 0.0}
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
from pathlib import Path
from glob import glob

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base-ckpt', required=True)
    p.add_argument('--args-json', required=True)
    p.add_argument('--npz-dir', required=True)
    p.add_argument('--n-samples', type=int, default=200)
    p.add_argument('--out-npz', required=True)
    p.add_argument('--out-json', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--encoder-output-key', default='encoder_outputs_pooled',
                   help='Which model output to use as latent. Pooled or pre-pool ok.')
    p.add_argument('--pool-method', choices=['mean', 'first'], default='mean',
                   help='If output is per-token, how to reduce to a single vector')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # Lazy imports
    try:
        import torch
        sys.path.insert(0, '/root/autoware_ws/scripts/train/Diffusion-Planner')
        from diffusion_planner.model.diffusion_planner import Diffusion_Planner
        from diffusion_planner.utils.config import Config
        from preference_optimization.utils import load_npz_data
    except Exception as e:
        print(f"deps missing: {e}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # Load model
    cfg = Config(args.args_json, guidance_fn=None)
    model = Diffusion_Planner(cfg).to(device)
    ckpt = torch.load(args.base_ckpt, map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt.get('ema_state_dict', ckpt))
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    # Find encoder submodule
    enc = None
    for name, mod in model.named_modules():
        if name.endswith('encoder') or name == 'context_encoder':
            enc = mod; break
    if enc is None:
        # fall back to running full model and extracting encoder output from hooks
        print("WARN: couldn't find a dedicated 'encoder' attribute; full forward")

    # Sample npz files
    npz_files = sorted(glob(os.path.join(args.npz_dir, '*.npz')))
    if len(npz_files) == 0:
        print(f"no npz in {args.npz_dir}"); sys.exit(1)
    random.shuffle(npz_files)
    npz_files = npz_files[:args.n_samples]
    print(f"will process {len(npz_files)} npz files")

    Z = []
    ids = []
    with torch.no_grad():
        for npz_path in npz_files:
            try:
                data = load_npz_data(npz_path, device)
                # We need a batch dim
                for k, v in data.items():
                    if isinstance(v, torch.Tensor) and v.dim() == data.get('ego_current_state').dim() - 1:
                        data[k] = v.unsqueeze(0)
                # Normalize
                data_n = cfg.observation_normalizer(data)
                # Run encoder (model.encoder if present)
                if enc is not None:
                    out = enc(data_n)
                    # out: typically a dict with 'context_embedding' or similar
                    if isinstance(out, dict):
                        # find tensor with shape [1, T, D]
                        for k, v in out.items():
                            if isinstance(v, torch.Tensor) and v.dim() == 3:
                                z = v[0]; break
                        else:
                            z = list(out.values())[0][0]
                    else:
                        z = out[0]
                    if z.dim() > 1:
                        if args.pool_method == 'mean':
                            z = z.mean(dim=0)
                        else:
                            z = z[0]
                    Z.append(z.cpu().numpy())
                    ids.append(os.path.basename(npz_path))
                else:
                    print(f"WARN: skipping {npz_path}, no encoder found")
            except Exception as e:
                print(f"  ! fail on {npz_path}: {e}")

    if len(Z) == 0:
        print("no latents extracted"); sys.exit(1)
    Z = np.stack(Z).astype(np.float32)
    print(f"\nlatent bank shape: {Z.shape}")

    # Save bank
    np.savez(args.out_npz, Z=Z, ids=np.array(ids))
    print(f"saved bank: {args.out_npz}")

    # Calibrate tau, sigma using LatentBank
    sys.path.insert(0, os.path.dirname(__file__))
    from ood_aware_lora_scaling import LatentBank
    bank = LatentBank(Z, ids=ids)
    tau, sigma = bank.calibrate_tau_sigma(k=5, sigma_factor=0.3)
    print(f"calibrated tau={tau:.4f}, sigma={sigma:.4f}")
    with open(args.out_json, 'w') as f:
        json.dump({'tau': tau, 'sigma': sigma, 'metric': 'l2', 'floor_alpha': 0.0,
                   'n_bank': len(Z), 'D': int(Z.shape[1])}, f, indent=2)
    print(f"saved calib: {args.out_json}")


if __name__ == '__main__':
    main()
