from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class WaypointMDNConfig:
    z_dim: int = 6
    n_components: int = 8
    min_log_sigma: float = -7.0
    max_log_sigma: float = 1.0
    clamp_sample: bool = True


class WaypointMDN(nn.Module):
    """
    MDN for macro latent z = [wp1, wp2, end_anchor] in normalized position space.

    Inputs:
      - obs: (B, H, 4)  [pos_norm, vel_norm]
      - cond: (B, 6)    [hour, day, trip_o(y,x), trip_d(y,x)]

    Outputs (params):
      - logits: (B, M)
      - mu: (B, M, D)
      - log_sigma: (B, M, D) diagonal covariance in log-space
    """

    def __init__(
        self,
        *,
        obs_dim: int = 4,
        cond_dim: int = 6,
        hidden_dim: int = 128,
        cfg: WaypointMDNConfig = WaypointMDNConfig(),
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.cfg = cfg

        self.encoder = nn.LSTM(
            input_size=int(self.obs_dim + self.cond_dim),
            hidden_size=int(self.hidden_dim),
            num_layers=1,
            batch_first=True,
        )
        self.head_logits = nn.Linear(int(self.hidden_dim), int(self.cfg.n_components))
        self.head_mu = nn.Linear(int(self.hidden_dim), int(self.cfg.n_components * self.cfg.z_dim))
        self.head_log_sigma = nn.Linear(int(self.hidden_dim), int(self.cfg.n_components * self.cfg.z_dim))

    def mdn_params(self, obs: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.ndim != 3 or int(obs.shape[-1]) != int(self.obs_dim):
            raise ValueError(f"Expected obs (B,H,{self.obs_dim}), got {tuple(obs.shape)}")
        if cond.ndim != 2 or int(cond.shape[-1]) != int(self.cond_dim):
            raise ValueError(f"Expected cond (B,{self.cond_dim}), got {tuple(cond.shape)}")

        B, H, _ = obs.shape
        cond_expanded = cond[:, None, :].expand(int(B), int(H), int(self.cond_dim))
        x = torch.cat([obs, cond_expanded], dim=-1)
        _, (h_n, _) = self.encoder(x)
        h = h_n[-1]

        logits = self.head_logits(h)
        mu = self.head_mu(h).view(int(B), int(self.cfg.n_components), int(self.cfg.z_dim))
        log_sigma = self.head_log_sigma(h).view(int(B), int(self.cfg.n_components), int(self.cfg.z_dim))
        log_sigma = torch.clamp(log_sigma, min=float(self.cfg.min_log_sigma), max=float(self.cfg.max_log_sigma))
        return logits, mu, log_sigma

    def nll(self, obs: torch.Tensor, cond: torch.Tensor, target_z: torch.Tensor, *, sample_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if target_z.ndim != 2 or int(target_z.shape[-1]) != int(self.cfg.z_dim):
            raise ValueError(f"Expected target_z (B,{self.cfg.z_dim}), got {tuple(target_z.shape)}")
        logits, mu, log_sigma = self.mdn_params(obs, cond)

        x = target_z[:, None, :]
        inv_sigma = torch.exp(-log_sigma)
        z = (x - mu) * inv_sigma
        log_2pi = float(torch.log(torch.tensor(2.0 * torch.pi, device=obs.device, dtype=obs.dtype)))
        log_prob = -0.5 * (z * z + 2.0 * log_sigma + log_2pi)
        log_prob = log_prob.sum(dim=-1)

        log_mix = F.log_softmax(logits, dim=-1) + log_prob
        log_p = torch.logsumexp(log_mix, dim=-1)
        loss = -log_p

        if sample_weight is not None:
            if sample_weight.ndim != 1 or int(sample_weight.shape[0]) != int(loss.shape[0]):
                raise ValueError(f"sample_weight must be (B,), got {tuple(sample_weight.shape)}")
            w = sample_weight.to(device=loss.device, dtype=loss.dtype)
            w = w / torch.clamp_min(w.mean(), 1e-6)
            return (loss * w).mean()
        return loss.mean()

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, cond: torch.Tensor, *, k: int, clamp: Optional[bool] = None) -> torch.Tensor:
        logits, mu, log_sigma = self.mdn_params(obs, cond)
        weights = torch.softmax(logits, dim=-1)
        B, M = int(weights.shape[0]), int(weights.shape[1])
        k = int(k)
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")

        comp = torch.multinomial(weights, num_samples=int(k), replacement=True)
        mu_sel = mu.gather(1, comp[:, :, None].expand(B, k, int(self.cfg.z_dim)))
        log_sigma_sel = log_sigma.gather(1, comp[:, :, None].expand(B, k, int(self.cfg.z_dim)))
        sigma_sel = torch.exp(log_sigma_sel)
        eps = torch.randn_like(mu_sel)
        z = mu_sel + sigma_sel * eps

        do_clamp = bool(self.cfg.clamp_sample) if clamp is None else bool(clamp)
        if do_clamp:
            z = torch.clamp(z, -1.0, 1.0)
        return z

    @torch.no_grad()
    def mode(self, obs: torch.Tensor, cond: torch.Tensor, *, clamp: Optional[bool] = None) -> torch.Tensor:
        logits, mu, _ = self.mdn_params(obs, cond)
        idx = torch.argmax(logits, dim=-1)
        z = mu[torch.arange(mu.shape[0], device=mu.device), idx]
        do_clamp = bool(self.cfg.clamp_sample) if clamp is None else bool(clamp)
        if do_clamp:
            z = torch.clamp(z, -1.0, 1.0)
        return z
