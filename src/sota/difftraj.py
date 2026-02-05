from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.diffusion.scheduler import DDPMScheduler
from src.models.diffusion.unet1d import UNet1D
from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class DiffTrajCfg:
    traj_len: int = 256
    act_dim: int = 2
    hidden_dim: int = 128
    emb_dim: int = 512
    diffusion_steps: int = 100
    prediction_type: str = "eps"  # eps only (KISS)

    d_model: int = 256
    n_route_cities: int = 4
    coord_scale: float = 1024.0


class DiffTrajModel(nn.Module):
    """
    GPS-space diffusion baseline (simplified):
      - generate fixed-length (y,x) sequence in grid coordinates, relative to start_pos
      - conditioned on OD/time/city via ConditionEncoder
    """

    def __init__(self, *, cfg: DiffTrajCfg) -> None:
        super().__init__()
        self.cfg = cfg
        if str(cfg.prediction_type) != "eps":
            raise ValueError("DiffTrajModel only supports prediction_type='eps' (KISS)")

        self.cond_enc = ConditionEncoder(
            ConditionEncoderCfg(d_model=int(cfg.d_model), n_route_cities=int(cfg.n_route_cities), coord_scale=float(cfg.coord_scale))
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(int(cfg.d_model), int(cfg.emb_dim)),
            nn.SiLU(),
            nn.Linear(int(cfg.emb_dim), int(cfg.emb_dim)),
        )
        self.unet = UNet1D(
            in_dim=int(cfg.act_dim),
            model_dim=int(cfg.hidden_dim),
            emb_dim=int(cfg.emb_dim),
            dim_mults=(1, 2, 4),
            cond_dim=int(cfg.emb_dim),
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=int(cfg.diffusion_steps))

    def to(self, device: torch.device) -> "DiffTrajModel":
        super().to(device)
        self.scheduler.to(device)
        return self

    def _global_cond(self, route_cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.cond_enc(
            start_pos=route_cond["start_pos"],
            dest_pos=route_cond["dest_pos"],
            hour=route_cond["hour"],
            dow=route_cond["dow"],
            route_city=route_cond["route_city"],
        )
        return self.cond_mlp(cond)

    def compute_loss(
        self,
        *,
        traj_yx_rel: torch.Tensor,  # (B,T,2)
        route_cond: Dict[str, torch.Tensor],
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if traj_yx_rel.ndim != 3 or int(traj_yx_rel.shape[-1]) != 2:
            raise ValueError(f"traj_yx_rel must be (B,T,2), got {tuple(traj_yx_rel.shape)}")
        B = int(traj_yx_rel.shape[0])
        device = traj_yx_rel.device
        self.scheduler.to(device)

        x0 = traj_yx_rel.to(dtype=torch.float32).permute(0, 2, 1).contiguous()  # (B,2,T)
        noise = torch.randn_like(x0)
        t = torch.randint(0, int(self.scheduler.num_train_timesteps), (B,), device=device).long()
        x_t = self.scheduler.add_noise(x0, noise, t)

        cond = self._global_cond(route_cond).to(device=device, dtype=torch.float32)
        eps = self.unet(x_t, t, cond=cond)

        per = (eps - noise) ** 2  # (B,2,T)
        per = per.mean(dim=(1, 2))  # (B,)
        if sample_weight is not None:
            w = sample_weight.to(dtype=per.dtype, device=per.device).flatten()
            w = torch.clamp_min(w, 1e-6)
            return (per * w).sum() / w.sum()
        return per.mean()

    @torch.no_grad()
    def sample(
        self,
        *,
        route_cond: Dict[str, torch.Tensor],
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Returns:
            traj_yx_rel: (B,T,2) float32
        """
        device = next(self.parameters()).device
        B = int(route_cond["route_city"].shape[0])
        T = int(self.cfg.traj_len)
        self.scheduler.to(device)

        n_steps = int(steps) if steps is not None else int(self.scheduler.num_train_timesteps)
        n_steps = max(1, min(int(n_steps), int(self.scheduler.num_train_timesteps)))

        x = torch.randn((B, int(self.cfg.act_dim), T), device=device, dtype=torch.float32)
        cond = self._global_cond(route_cond).to(device=device, dtype=torch.float32)

        # Use DDPM ancestral sampling; if steps < diffusion_steps, we subsample timesteps uniformly.
        if int(n_steps) == int(self.scheduler.num_train_timesteps):
            ts = list(range(int(self.scheduler.num_train_timesteps) - 1, -1, -1))
        else:
            idx = torch.linspace(0, int(self.scheduler.num_train_timesteps) - 1, steps=int(n_steps), device="cpu")
            ts = [int(round(float(t))) for t in idx.tolist()][::-1]
            ts = sorted(set(ts), reverse=True)

        for t in ts:
            tt = torch.full((B,), int(t), device=device, dtype=torch.long)
            eps = self.unet(x, tt, cond=cond)
            x = self.scheduler.step(eps, int(t), x)

        return x.permute(0, 2, 1).contiguous()  # (B,T,2)

    def state_dict_cpu(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.state_dict().items()}

    def ckpt_payload(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "model_state_dict": self.state_dict_cpu()}

