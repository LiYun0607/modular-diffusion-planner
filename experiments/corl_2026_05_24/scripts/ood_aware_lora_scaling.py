"""Inference-time OOD-aware LoRA scaling for Diffusion-Planner.

CORE IDEA (CoRL §7 method):
A fine-tuned LoRA delta gives gains on inputs similar to its training distribution
but corrupts predictions on OOD inputs (cold-start v=0, novel ODD, unseen route).
Instead of retraining, we attenuate the LoRA contribution at inference time
based on a single scalar — the encoder-latent distance from a training-distribution
latent bank.

PIPELINE:
  1. (offline, once) Build a latent bank Z_train ∈ R^[N, D] by running ~200 training
     npz scenes through the FROZEN base encoder.
  2. (offline, once) Choose τ (distance threshold) and σ (transition width).
     We use the latent bank's own internal distance distribution to set
     τ = mean(min-distance to k=5 NN over the bank).
  3. (per-frame, inference) For each input scene:
        z = base_encoder(scene)
        d = min L2 distance from z to Z_train
        α(d) = sigmoid(-(d - τ) / σ)   # α≈1 when in-dist, α→0 when very OOD
        effective_LoRA_delta = α * LoRA_delta
     This is a SCALAR multiplier on the LoRA delta; no retraining needed.

WHY IT WORKS:
  - Cold-start (v=0) inputs land in low-density region of training latent space
    → d is large → α small → fall back to base, which we showed is sim-clean.
  - Kashiwa input under a kashiwa-trained LoRA → d small → α≈1 → full LoRA gain.
  - Provides smooth interpolation (vs. hard switch) so trajectories don't jump.

THIS MODULE:
  - LatentBank: collect / save / load / nearest-distance queries
  - OODScaler: takes (latent_bank, tau, sigma) and gives α for any new latent
  - Wrapper: ScaledLoRADelta that wraps a callable returning LoRA params and
    multiplies them by α(z) at inference time

The actual integration with Diffusion_Planner inference loop is shown in
__example__ at bottom. Real plumbing needs to hook into apply_molora to scale
each LoRA matrix's contribution to forward pass.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np


class LatentBank:
    """Holds [N, D] latent matrix from base encoder over training scenes."""

    def __init__(self, latents: np.ndarray, ids: list[str] | None = None):
        assert latents.ndim == 2, f"expected [N,D], got {latents.shape}"
        self.Z = latents.astype(np.float32)
        self.N, self.D = self.Z.shape
        self.ids = list(ids) if ids is not None else [str(i) for i in range(self.N)]

    @classmethod
    def from_npy(cls, path: str | Path) -> 'LatentBank':
        data = np.load(path, allow_pickle=True)
        if isinstance(data, np.ndarray):
            return cls(data)
        return cls(data['Z'], list(data.get('ids', [])))

    def save_npy(self, path: str | Path) -> None:
        np.savez(path, Z=self.Z, ids=np.array(self.ids))

    def min_distance(self, z: np.ndarray, metric: str = 'l2') -> float:
        """Distance from z (shape [D]) to its nearest neighbor in the bank."""
        assert z.shape == (self.D,), f"got {z.shape}, expected ({self.D},)"
        if metric == 'l2':
            d = np.linalg.norm(self.Z - z[None], axis=1)
        elif metric == 'cosine':
            zn = z / (np.linalg.norm(z) + 1e-9)
            Zn = self.Z / (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)
            d = 1.0 - Zn @ zn
        else:
            raise ValueError(f"unknown metric: {metric}")
        return float(d.min())

    def knn_distances(self, z: np.ndarray, k: int = 5, metric: str = 'l2') -> np.ndarray:
        """Sorted distances to top-k nearest neighbors."""
        if metric == 'l2':
            d = np.linalg.norm(self.Z - z[None], axis=1)
        elif metric == 'cosine':
            zn = z / (np.linalg.norm(z) + 1e-9)
            Zn = self.Z / (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)
            d = 1.0 - Zn @ zn
        else:
            raise ValueError(metric)
        return np.sort(d)[:k]

    def calibrate_tau_sigma(self, k: int = 5, sigma_factor: float = 0.3,
                            metric: str = 'l2') -> tuple[float, float]:
        """Auto-calibrate (tau, sigma) from internal distance distribution.

        For each bank entry, take mean distance to its k-NN within the bank.
        Use the 90th percentile of those as τ (so points further than the typical
        in-bank density are deemed OOD). Sigma = sigma_factor * τ.
        """
        means_to_knn = []
        for i in range(self.N):
            z = self.Z[i]
            if metric == 'l2':
                d = np.linalg.norm(self.Z - z[None], axis=1)
            elif metric == 'cosine':
                zn = z / (np.linalg.norm(z) + 1e-9)
                Zn = self.Z / (np.linalg.norm(self.Z, axis=1, keepdims=True) + 1e-9)
                d = 1.0 - Zn @ zn
            d_sorted = np.sort(d)
            mean_knn = float(d_sorted[1:k+1].mean())  # exclude self at 0
            means_to_knn.append(mean_knn)
        tau = float(np.percentile(means_to_knn, 90))
        sigma = max(sigma_factor * tau, 1e-3)
        return tau, sigma


class OODScaler:
    """Smooth in-distribution → OOD transition via sigmoid gating."""

    def __init__(self, bank: LatentBank, tau: float, sigma: float,
                 metric: str = 'l2', floor_alpha: float = 0.0):
        self.bank = bank
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.metric = metric
        self.floor_alpha = float(floor_alpha)

    def alpha(self, z: np.ndarray) -> float:
        """Return α ∈ [floor_alpha, 1]: 1 when in-dist, → floor_alpha when very OOD."""
        d = self.bank.min_distance(z, metric=self.metric)
        x = -(d - self.tau) / self.sigma
        s = 1.0 / (1.0 + np.exp(-x))
        return float(self.floor_alpha + (1 - self.floor_alpha) * s)

    def alpha_with_distance(self, z: np.ndarray) -> tuple[float, float]:
        d = self.bank.min_distance(z, metric=self.metric)
        x = -(d - self.tau) / self.sigma
        s = 1.0 / (1.0 + np.exp(-x))
        a = self.floor_alpha + (1 - self.floor_alpha) * s
        return float(a), float(d)

    def to_json(self, path: str | Path) -> None:
        with open(path, 'w') as f:
            json.dump({'tau': self.tau, 'sigma': self.sigma,
                       'metric': self.metric, 'floor_alpha': self.floor_alpha}, f, indent=2)


# ----- The torch integration shim (only imports torch lazily) -----

def make_scaled_apply_molora(apply_molora_fn, get_latent_fn, ood_scaler: OODScaler):
    """Wrap an existing apply_molora_fn so each forward pass scales the LoRA delta
    by α(z) computed from the current input scene's latent.

    apply_molora_fn: existing callable (model, scale=1.0) -> None that registers
                    LoRA hooks. We assume it accepts a `scale` kwarg; if not,
                    the wrapper instead scales the LoRA A.weight or B.weight by α.
    get_latent_fn:  callable (scene_inputs) -> np.ndarray of shape [D]; runs base
                    encoder forward and returns the latent.
    ood_scaler:     pre-built OODScaler.

    Returns: a function (model, scene_inputs) -> α that you call once before
            each forward inference; it applies the scaled LoRA in-place.
    """
    def apply_with_ood(model, scene_inputs):
        z = get_latent_fn(scene_inputs)
        a = ood_scaler.alpha(z)
        # NB: real implementation depends on apply_molora signature; here we
        # assume it can take a `scale` arg, OR you manually multiply the deltas.
        apply_molora_fn(model, scale=a)
        return a
    return apply_with_ood


def _self_test():
    """Smoke test: build a synthetic bank, query in-dist + OOD latents."""
    np.random.seed(0)
    # Synthetic bank: 200 latents drawn from N(0, I_64)
    bank_data = np.random.randn(200, 64).astype(np.float32)
    bank = LatentBank(bank_data, ids=[f'train_{i}' for i in range(200)])
    tau, sigma = bank.calibrate_tau_sigma(k=5, sigma_factor=0.3)
    print(f"calibrated tau={tau:.3f}, sigma={sigma:.3f}")

    scaler = OODScaler(bank, tau, sigma)

    # query 1: in-distribution latent (sample from same N(0, I))
    z_id = np.random.randn(64).astype(np.float32)
    a_id, d_id = scaler.alpha_with_distance(z_id)
    # query 2: OOD latent (shifted far away)
    z_ood = np.random.randn(64).astype(np.float32) + 10.0
    a_ood, d_ood = scaler.alpha_with_distance(z_ood)
    # query 3: extreme OOD (shifted very far)
    z_xood = np.random.randn(64).astype(np.float32) * 5 + 50.0
    a_xood, d_xood = scaler.alpha_with_distance(z_xood)

    print(f"in-dist:     d={d_id:.2f}  α={a_id:.3f}  (expect α≈1)")
    print(f"OOD shift 10: d={d_ood:.2f}  α={a_ood:.3f}  (expect α≈0.5 or so)")
    print(f"extreme OOD:  d={d_xood:.2f}  α={a_xood:.3f}  (expect α→0)")

    # sweep: distance vs alpha curve
    ds = np.linspace(0, 100, 50)
    alphas = []
    for d in ds:
        # synthesize z at distance d from bank center
        z = np.zeros(64); z[0] = d
        z = z + bank.Z.mean(axis=0)
        a, dd = scaler.alpha_with_distance(z)
        alphas.append((dd, a))
    return alphas, scaler.tau, scaler.sigma


if __name__ == '__main__':
    _self_test()
