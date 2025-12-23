import torch
import torch.nn as nn
from typing import Optional

from src.models.base_model import BaseTrajectoryModel
from src.models.diffusion.unet1d import UNet1D


class RectifiedFlowTrajectoryModel(BaseTrajectoryModel):
    """
    Rectified Flow / Flow Matching trajectory model (KISS pilot).

    We learn a continuous-time vector field v_theta(x_t, t | cond) that transports
    a Gaussian prior x_0 ~ N(0, sigma^2 I) to the data x_1 (here: future velocity sequence).
    """

    def __init__(
        self,
        *,
        obs_dim: int = 4,
        act_dim: int = 2,
        cond_dim: int = 6,
        obs_len: int = 8,
        pred_len: int = 12,
        hidden_dim: int = 128,
        time_scale: float = 1000.0,
        noise_sigma: float = 1.0,
        solver_steps: int = 20,
    ):
        super().__init__()

        self.pred_len = int(pred_len)
        self.act_dim = int(act_dim)
        self.solver_steps = int(solver_steps)

        hist_flat_dim = int(obs_len) * int(obs_dim)
        input_cond_dim = hist_flat_dim + int(cond_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(input_cond_dim, int(hidden_dim) * 2),
            nn.SiLU(),
            nn.Linear(int(hidden_dim) * 2, int(hidden_dim) * 4),
        )

        self.unet = UNet1D(
            in_dim=int(act_dim),
            model_dim=int(hidden_dim),
            emb_dim=int(hidden_dim) * 4,
            dim_mults=(1, 2, 4),
            cond_dim=int(hidden_dim) * 4,
        )

        self.register_buffer("rf_time_scale", torch.tensor(float(time_scale), dtype=torch.float32))
        self.register_buffer("rf_noise_sigma", torch.tensor(float(noise_sigma), dtype=torch.float32))

    def set_noise_sigma(self, sigma: float) -> None:
        self.rf_noise_sigma.data = torch.tensor(float(sigma), dtype=self.rf_noise_sigma.dtype, device=self.rf_noise_sigma.device)

    def get_global_cond(self, obs: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        obs_flat = obs.reshape(B, -1)
        x = torch.cat([obs_flat, cond], dim=-1)
        return self.cond_encoder(x)

    def compute_loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        *,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = obs.shape[0]
        device = obs.device

        x1 = target.permute(0, 2, 1)  # (B, act_dim, F)

        sigma = self.rf_noise_sigma.to(device=device, dtype=x1.dtype)
        z0 = torch.randn_like(x1) * sigma

        t = torch.rand((B,), device=device, dtype=x1.dtype)
        t_ = t[:, None, None]
        x_t = (1.0 - t_) * z0 + t_ * x1
        v_target = x1 - z0

        global_cond = self.get_global_cond(obs, cond)
        steps = t * self.rf_time_scale.to(device=device, dtype=x1.dtype)
        v_pred = self.unet(x_t, steps, cond=global_cond)

        per = (v_pred - v_target) ** 2
        per = per.mean(dim=(1, 2))
        if sample_weight is not None:
            w = sample_weight.to(dtype=per.dtype, device=per.device).flatten()
            w = torch.clamp_min(w, 1e-6)
            return (per * w).sum() / w.sum()
        return per.mean()

    def forward(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target is None:
            return torch.tensor(0.0, device=obs.device)
        return self.compute_loss(obs, cond, target, sample_weight=sample_weight)

    def sample_trajectory(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        horizon: int,
        *,
        cond_uncond: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        B = obs.shape[0]
        device = obs.device

        global_cond = self.get_global_cond(obs, cond)
        use_cfg = (cond_uncond is not None) and (float(cfg_scale) != 0.0)
        global_cond_uncond = None
        if use_cfg:
            global_cond_uncond = self.get_global_cond(obs, cond_uncond.to(device=device, dtype=cond.dtype))

        sigma = self.rf_noise_sigma.to(device=device, dtype=obs.dtype)
        x = torch.randn((B, self.act_dim, int(horizon)), device=device, dtype=obs.dtype) * sigma

        n_steps = max(1, int(self.solver_steps))
        dt = 1.0 / float(n_steps)

        for i in range(n_steps):
            t = (float(i) + 0.5) * dt
            steps = torch.full((B,), t, device=device, dtype=obs.dtype) * self.rf_time_scale.to(device=device, dtype=obs.dtype)
            if use_cfg:
                v_u = self.unet(x, steps, cond=global_cond_uncond)
                v_c = self.unet(x, steps, cond=global_cond)
                v = v_u + float(cfg_scale) * (v_c - v_u)
            else:
                v = self.unet(x, steps, cond=global_cond)
            x = x + dt * v

        return x.permute(0, 2, 1)

