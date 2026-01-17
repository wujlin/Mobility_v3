from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.diffusion.scheduler import DDPMScheduler
from src.models.diffusion.unet1d import UNet1D
from src.models.way_casd.conditions import ConditionEncoder, ConditionEncoderCfg


@dataclass(frozen=True)
class SkeletonCrossAttnCfg:
    d_skel: int = 256
    act_dim: int = 2
    # Attention internal dim (compute cost ~ O(T*L*model_dim)).
    model_dim: int = 128
    # Output channels for UNet control_mid. If None, defaults to model_dim.
    out_dim: Optional[int] = None
    num_heads: int = 4
    diff_steps: int = 100
    weight: float = 1.0


class SkeletonCrossAttentionControlMid(nn.Module):
    """
    Cross-attention ControlNet-like mid control for UNet1D.

    Query:  x_t (B,act_dim,T) projected to (B,T,model_dim) + timestep embedding
    Key/Val: skeleton_latent (B,L,d_skel) projected to (B,L,model_dim)

    Output: control_mid (B,model_dim,T) to be fed into UNet1D(control_mid=...).
    """

    def __init__(self, *, cfg: SkeletonCrossAttnCfg) -> None:
        super().__init__()
        self.cfg = cfg
        d = int(cfg.model_dim)
        out_d = int(cfg.out_dim) if cfg.out_dim is not None else d
        h = int(cfg.num_heads)
        if d <= 0:
            raise ValueError("model_dim must be > 0")
        if h <= 0 or (d % h) != 0:
            raise ValueError("num_heads must be > 0 and divide model_dim")
        tmax = int(cfg.diff_steps)
        if tmax <= 0:
            raise ValueError("diff_steps must be > 0")
        w = float(cfg.weight)
        if not torch.isfinite(torch.tensor(w)) or w < 0.0:
            raise ValueError("weight must be finite and >= 0")

        self.q_proj = nn.Conv1d(int(cfg.act_dim), d, kernel_size=1)
        self.kv_proj = nn.Linear(int(cfg.d_skel), d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.t_embed = nn.Embedding(tmax, d)
        self.out_proj = nn.Identity() if int(out_d) == int(d) else nn.Conv1d(d, int(out_d), kernel_size=1)

        self.record_attn: bool = False
        self.last_attn: Optional[torch.Tensor] = None  # (B,H,T,L) if record_attn

    def forward(
        self,
        x_t: torch.Tensor,  # (B,act_dim,T)
        timesteps: torch.Tensor,  # (B,) int64
        *,
        skeleton_latent: torch.Tensor,  # (B,L,d_skel)
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if x_t.ndim != 3:
            raise ValueError(f"x_t must be (B,act_dim,T), got {tuple(x_t.shape)}")
        if skeleton_latent.ndim != 3:
            raise ValueError(f"skeleton_latent must be (B,L,d_skel), got {tuple(skeleton_latent.shape)}")
        B = int(x_t.shape[0])
        if int(skeleton_latent.shape[0]) != B:
            raise ValueError(f"batch mismatch: x_t B={B} vs skeleton_latent B={int(skeleton_latent.shape[0])}")

        # Query: (B,act_dim,T) -> (B,T,model_dim)
        q = self.q_proj(x_t.to(dtype=torch.float32)).transpose(1, 2).contiguous()
        t = timesteps.to(device=q.device, dtype=torch.long).clamp(min=0, max=int(self.cfg.diff_steps) - 1)
        q = q + self.t_embed(t)[:, None, :]

        # KV: (B,L,d_skel) -> (B,L,model_dim)
        kv = self.kv_proj(skeleton_latent.to(dtype=torch.float32))

        attn_out, attn_w = self.attn(
            q,
            kv,
            kv,
            need_weights=(bool(need_weights) or bool(self.record_attn)),
            average_attn_weights=False,
        )
        control_mid = attn_out.transpose(1, 2).contiguous()  # (B,model_dim,T)
        control_mid = self.out_proj(control_mid) * float(self.cfg.weight)  # (B,out_dim,T)

        if bool(self.record_attn):
            self.last_attn = attn_w.detach()
        return control_mid, (attn_w.detach() if bool(need_weights) else None)


@dataclass(frozen=True)
class GPSDiffusionCfg:
    traj_len: int = 256
    act_dim: int = 2
    hidden_dim: int = 128
    emb_dim: int = 512
    diffusion_steps: int = 100
    prediction_type: str = "eps"  # eps|v

    # Condition encoder
    d_model: int = 256
    n_route_cities: int = 4
    coord_scale: float = 1024.0

    # Skeleton cross-attention control (mid)
    skel_attn: SkeletonCrossAttnCfg = SkeletonCrossAttnCfg()
    skel_noise_sigma: float = 0.1


class GPSDiffusionExecutionModel(nn.Module):
    """
    Execution stage: generate a fixed-length (y,x) trajectory conditioned on:
      - route_cond (OD/time/city)
      - skeleton_latent tokens from decision stage (Way-CASD latent)

    Output representation:
      - traj_yx_rel: (B,T,2) in grid coordinates, relative to start (first point == 0).
    """

    def __init__(self, *, cfg: GPSDiffusionCfg) -> None:
        super().__init__()
        self.cfg = cfg

        pred_type = str(cfg.prediction_type)
        if pred_type not in ("eps", "v"):
            raise ValueError(f"Unknown prediction_type: {pred_type} (expected: eps|v)")
        self.prediction_type = pred_type

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

        sk = cfg.skel_attn
        # UNet mid control requires channel match. UNet mid channels = hidden_dim * dim_mults[-1].
        mid_ch = int(self.unet.mid_block1.block1[0].out_channels)
        self.skel_ctrl = SkeletonCrossAttentionControlMid(
            cfg=SkeletonCrossAttnCfg(
                d_skel=int(sk.d_skel),
                act_dim=int(cfg.act_dim),
                model_dim=int(cfg.hidden_dim),
                out_dim=int(mid_ch),
                num_heads=int(sk.num_heads),
                diff_steps=int(cfg.diffusion_steps),
                weight=float(sk.weight),
            )
        )

    def to(self, device: torch.device) -> "GPSDiffusionExecutionModel":
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

    def _maybe_noise_skeleton(self, skel: torch.Tensor) -> torch.Tensor:
        sig = float(getattr(self.cfg, "skel_noise_sigma", 0.0))
        if (not self.training) or sig <= 0.0:
            return skel
        return skel + torch.randn_like(skel) * sig

    def _control_mid(self, x_t: torch.Tensor, timesteps: torch.Tensor, *, skeleton_latent: torch.Tensor) -> torch.Tensor:
        sk = self._maybe_noise_skeleton(skeleton_latent)
        ctrl, _ = self.skel_ctrl(x_t, timesteps, skeleton_latent=sk, need_weights=False)
        return ctrl

    def compute_loss(
        self,
        *,
        traj_yx_rel: torch.Tensor,  # (B,T,2)
        route_cond: Dict[str, torch.Tensor],
        skeleton_latent: torch.Tensor,  # (B,L,d)
        sample_weight: Optional[torch.Tensor] = None,
        return_x0_pred: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        DDPM training loss. Returns:
          - loss
          - (loss, x0_pred) if return_x0_pred=True, where x0_pred is (B,2,T)
        """
        if traj_yx_rel.ndim != 3 or int(traj_yx_rel.shape[-1]) != 2:
            raise ValueError(f"traj_yx_rel must be (B,T,2), got {tuple(traj_yx_rel.shape)}")
        B = int(traj_yx_rel.shape[0])
        device = traj_yx_rel.device
        self.scheduler.to(device)

        x_0 = traj_yx_rel.to(dtype=torch.float32).permute(0, 2, 1).contiguous()  # (B,2,T)
        noise = torch.randn_like(x_0)
        timesteps = torch.randint(0, int(self.scheduler.num_train_timesteps), (B,), device=device).long()
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)

        global_cond = self._global_cond(route_cond).to(device=device, dtype=torch.float32)
        control_mid = self._control_mid(x_t, timesteps, skeleton_latent=skeleton_latent)

        model_out = self.unet(x_t, timesteps, cond=global_cond, control_mid=control_mid)

        sqrt_alpha_prod = self.scheduler.sqrt_alphas_cumprod[timesteps].flatten()[:, None, None]
        sqrt_one_minus_alpha_prod = self.scheduler.sqrt_one_minus_alphas_cumprod[timesteps].flatten()[:, None, None]

        if self.prediction_type == "v":
            target_out = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * x_0
        else:
            target_out = noise

        per = (model_out - target_out) ** 2  # (B,2,T)
        per = per.mean(dim=(1, 2))  # (B,)
        if sample_weight is not None:
            w = sample_weight.to(dtype=per.dtype, device=per.device).flatten()
            w = torch.clamp_min(w, 1e-6)
            loss = (per * w).sum() / w.sum()
        else:
            loss = per.mean()

        if not bool(return_x0_pred):
            return loss

        if self.prediction_type == "v":
            x0_pred = sqrt_alpha_prod * x_t - sqrt_one_minus_alpha_prod * model_out
        else:
            x0_pred = (x_t - sqrt_one_minus_alpha_prod * model_out) / (sqrt_alpha_prod + 1e-8)
        return loss, x0_pred

    @torch.no_grad()
    def sample(
        self,
        *,
        route_cond: Dict[str, torch.Tensor],
        skeleton_latent: torch.Tensor,
        traj_len: Optional[int] = None,
        fix_ends: bool = True,
    ) -> torch.Tensor:
        """
        Reverse diffusion sampling.

        Returns:
          traj_yx_rel: (B,T,2) float32
        """
        device = next(self.parameters()).device
        self.scheduler.to(device)
        B = int(route_cond["start_pos"].shape[0])
        T = int(traj_len) if traj_len is not None else int(self.cfg.traj_len)
        C = int(self.cfg.act_dim)

        global_cond = self._global_cond(route_cond).to(device=device, dtype=torch.float32)
        x_t = torch.randn((B, C, T), device=device, dtype=torch.float32)

        # Hard constraint in relative coords: first point is always 0.
        # If fix_ends: also fix last point to (dest-start) in relative coords.
        start = route_cond["start_pos"].to(device=device, dtype=torch.float32)
        dest = route_cond["dest_pos"].to(device=device, dtype=torch.float32)
        rel_end = dest - start
        coord_scale = float(getattr(self.cfg, "coord_scale", 0.0))
        if coord_scale > 0:
            rel_end = rel_end / coord_scale

        for t in reversed(range(int(self.scheduler.num_train_timesteps))):
            ts = torch.full((B,), t, device=device, dtype=torch.long)
            control_mid = self._control_mid(x_t, ts, skeleton_latent=skeleton_latent)
            model_out = self.unet(x_t, ts, cond=global_cond, control_mid=control_mid)

            if self.prediction_type == "v":
                alpha = self.scheduler.sqrt_alphas_cumprod[t].to(device=device, dtype=x_t.dtype)
                sigma = self.scheduler.sqrt_one_minus_alphas_cumprod[t].to(device=device, dtype=x_t.dtype)
                eps_pred = sigma * x_t + alpha * model_out
            else:
                eps_pred = model_out

            x_t = self.scheduler.step(eps_pred, t, x_t)
            if bool(fix_ends):
                x_t[:, :, 0] = 0.0
                x_t[:, :, -1] = rel_end

        return x_t.permute(0, 2, 1).contiguous()
