from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.semantic.semantic_patch_sampler import sample_patch_at_rel_waypoints


@dataclass(frozen=True)
class PosEncConfig:
    in_channels: int
    num_waypoints: int
    extent: float
    emb_dim: int
    diff_steps: int
    mlp_hidden_dim: int
    weight: float


class WaypointSemanticPosEnc(nn.Module):
    """
    Scheme-B: position-aligned semantic conditioning for waypoint diffusion.

    For each diffusion step, we:
      1) unnormalize current noisy waypoints x_t -> rel (s,o)
      2) sample OD-aligned semantic patch at those waypoint locations
      3) map [rel, sampled_sem] -> extra global embedding (added to global_cond)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        num_waypoints: int,
        extent: float,
        rel_mean: torch.Tensor,  # (2,)
        rel_std: torch.Tensor,  # (2,)
        emb_dim: int,
        diff_steps: int,
        mlp_hidden_dim: int = 256,
        weight: float = 1.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        k = int(num_waypoints)
        d = int(emb_dim)
        tmax = int(diff_steps)
        h = int(mlp_hidden_dim)
        w = float(weight)
        ext = float(extent)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if k <= 0:
            raise ValueError("num_waypoints must be > 0")
        if d <= 0:
            raise ValueError("emb_dim must be > 0")
        if tmax <= 0:
            raise ValueError("diff_steps must be > 0")
        if h <= 0:
            raise ValueError("mlp_hidden_dim must be > 0")
        if not (ext > 0.0):
            raise ValueError("extent must be > 0")
        if not torch.isfinite(torch.tensor(w)) or w < 0.0:
            raise ValueError("weight must be finite and >= 0")

        self.cfg = PosEncConfig(
            in_channels=c_in,
            num_waypoints=k,
            extent=ext,
            emb_dim=d,
            diff_steps=tmax,
            mlp_hidden_dim=h,
            weight=w,
        )

        rel_mean = rel_mean.detach().to(dtype=torch.float32).reshape(2)
        rel_std = rel_std.detach().to(dtype=torch.float32).reshape(2)
        rel_std = torch.clamp(rel_std, min=1e-3)
        self.register_buffer("rel_mean", rel_mean)
        self.register_buffer("rel_std", rel_std)

        feat_dim = int(k) * int(2 + c_in)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, h),
            nn.SiLU(),
            nn.Linear(h, d),
        )
        self.t_embed = nn.Embedding(tmax, d)

    def forward(
        self,
        x_t: torch.Tensor,  # (B,2,K) normalized rel
        timesteps: torch.Tensor,  # (B,)
        *,
        grid_patch: torch.Tensor,  # (B,C,S,S)
        start_pos: torch.Tensor,  # (B,2) grid coords
        dest_pos: torch.Tensor,  # (B,2)
    ) -> torch.Tensor:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be (B,2,K), got {tuple(x_t.shape)}")
        b = int(x_t.shape[0])
        if int(x_t.shape[1]) != 2 or int(x_t.shape[2]) != int(self.cfg.num_waypoints):
            raise ValueError(f"x_t must be (B,2,{int(self.cfg.num_waypoints)}), got {tuple(x_t.shape)}")
        if grid_patch.ndim != 4 or int(grid_patch.shape[0]) != b or int(grid_patch.shape[1]) != int(self.cfg.in_channels):
            raise ValueError(f"grid_patch must be (B,{int(self.cfg.in_channels)},S,S), got {tuple(grid_patch.shape)}")
        if start_pos.shape != (b, 2) or dest_pos.shape != (b, 2):
            raise ValueError(f"start_pos/dest_pos must be (B,2), got {tuple(start_pos.shape)} and {tuple(dest_pos.shape)}")

        # x_t: (B,2,K) -> rel_norm (B,K,2)
        rel_norm = x_t.permute(0, 2, 1).to(dtype=torch.float32)
        rel = rel_norm * self.rel_std[None, None, :] + self.rel_mean[None, None, :]  # (B,K,2)
        sem = sample_patch_at_rel_waypoints(
            patch=grid_patch,
            start_pos=start_pos,
            dest_pos=dest_pos,
            rel=rel,
            extent=float(self.cfg.extent),
        )  # (B,K,C)
        feat = torch.cat([rel, sem], dim=-1).reshape(b, -1)  # (B, K*(2+C))
        out = self.mlp(feat) + self.t_embed(timesteps.to(device=feat.device, dtype=torch.long).clamp(min=0, max=int(self.cfg.diff_steps) - 1))
        return out * float(self.cfg.weight)

