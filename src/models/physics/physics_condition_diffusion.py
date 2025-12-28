import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from src.models.base_model import BaseTrajectoryModel
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.cnn_encoder import CNNEncoder

class PhysicsConditionDiffusion(BaseTrajectoryModel):
    """
    Physics-Informed Diffusion Model using Condition Learning.
    Input: History + Global Cond + Nav Patch (encoded).
    """
    def __init__(self, 
                 obs_dim: int = 4, 
                 act_dim: int = 2, 
                 cond_dim: int = 6,
                 nav_patch_size: int = 32,
                 nav_emb_dim: int = 32,
                 nav_emb_scale: float = 1.0,
                 nav_emb_dropout: float = 0.0,
                 nav_gate: str = "none",
                 nav_gate_hidden: int = 32,
                 nav_gate_dropout: float = 0.0,
                 nav_query: str = "none",
                 nav_query_field: str = "dist",
                 nav_query_dist_sigma: float = 3.0,
                 pos_min: Optional[Tuple[float, float]] = None,
                 pos_range: Optional[Tuple[float, float]] = None,
                 obs_len: int = 8,
                 pred_len: int = 12,
                 hidden_dim: int = 64,
                 diffusion_steps: int = 100,
                 prediction_type: str = "eps"):
        super().__init__()

        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.obs_len = int(obs_len)

        self.nav_encoder = CNNEncoder(output_dim=nav_emb_dim, patch_size=nav_patch_size)
        self.nav_emb_scale = float(nav_emb_scale)
        self.nav_emb_dropout = float(nav_emb_dropout)

        nav_query = str(nav_query)
        if nav_query not in ("none", "global"):
            raise ValueError(f"Unknown nav_query: {nav_query} (expected: none|global)")
        self.nav_query_mode = nav_query
        nav_query_field = str(nav_query_field)
        if nav_query_field not in ("dist", "count"):
            raise ValueError(f"Unknown nav_query_field: {nav_query_field} (expected: dist|count)")
        self.nav_query_field = nav_query_field
        self.nav_query_dist_sigma = float(nav_query_dist_sigma)
        self.nav_query_mlp: Optional[nn.Module] = None
        if self.nav_query_mode != "none":
            self.nav_query_mlp = nn.Sequential(
                nn.Linear(1, hidden_dim * 2),
                nn.SiLU(),
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
            )
        self.pos_min_buf = None
        self.pos_range_buf = None
        if pos_min is not None and pos_range is not None:
            self.register_buffer("pos_min_buf", torch.tensor(pos_min, dtype=torch.float32))
            self.register_buffer("pos_range_buf", torch.tensor(pos_range, dtype=torch.float32))

        nav_gate = str(nav_gate)
        if nav_gate not in ("none", "obscond"):
            raise ValueError(f"Unknown nav_gate: {nav_gate} (expected: none|obscond)")
        self.nav_gate_mode = nav_gate
        self.nav_gate_dropout = float(nav_gate_dropout)
        self.nav_gate: Optional[nn.Module] = None
        if self.nav_gate_mode != "none":
            gate_in_dim = self.obs_len * self.obs_dim + self.cond_dim
            h = int(nav_gate_hidden)
            self.nav_gate = nn.Sequential(
                nn.Linear(gate_in_dim, h),
                nn.SiLU(),
                nn.Linear(h, 1),
            )
        
        # Instantiate wrapped diffusion model
        # Condition dim increases by nav_emb_dim
        self.diffusion = DiffusionTrajectoryModel(
            obs_dim=obs_dim,
            act_dim=act_dim,
            cond_dim=cond_dim + nav_emb_dim,
            obs_len=obs_len,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            diffusion_steps=diffusion_steps,
            prediction_type=str(prediction_type),
        )

    def _nav_query_emb(self, x_t: torch.Tensor, *, nav_global: torch.Tensor) -> torch.Tensor:
        if self.nav_query_mode == "none":
            raise RuntimeError("_nav_query_emb called but nav_query_mode is none")
        if self.nav_query_mlp is None:
            raise RuntimeError("_nav_query_emb called but nav_query_mlp is None")
        if self.pos_min_buf is None or self.pos_range_buf is None:
            raise RuntimeError("nav_query requires pos_min/pos_range buffers (pass pos_min,pos_range when constructing the model).")

        B = int(x_t.shape[0])
        z = x_t.permute(0, 2, 1)  # (B,L,2) in normalized pos
        z = torch.clamp(z, -1.0, 1.0)

        pos_min = self.pos_min_buf.to(device=x_t.device, dtype=z.dtype)
        pos_range = self.pos_range_buf.to(device=x_t.device, dtype=z.dtype)
        pos_grid = (z + 1.0) * 0.5 * pos_range[None, None, :] + pos_min[None, None, :]  # (B,L,2) [y,x]

        if nav_global.ndim == 2:
            nav_global = nav_global[None, None, :, :]
        elif nav_global.ndim == 3:
            nav_global = nav_global[None, :, :, :]
        if nav_global.ndim != 4:
            raise ValueError(f"nav_global must be (C,H,W) or (1,C,H,W), got {tuple(nav_global.shape)}")

        nav_global = nav_global.to(device=x_t.device, dtype=z.dtype)
        if int(nav_global.shape[0]) == 1 and B > 1:
            nav_global = nav_global.expand(B, -1, -1, -1)
        if int(nav_global.shape[0]) != B:
            raise ValueError(f"nav_global batch mismatch: got {int(nav_global.shape[0])}, expected {B}")

        H = int(nav_global.shape[2])
        W = int(nav_global.shape[3])
        x = pos_grid[:, :, 1]
        y = pos_grid[:, :, 0]
        x_n = (x / max(float(W - 1), 1.0)) * 2.0 - 1.0
        y_n = (y / max(float(H - 1), 1.0)) * 2.0 - 1.0
        grid = torch.stack([x_n, y_n], dim=-1).unsqueeze(2)  # (B,L,1,2)

        sampled = F.grid_sample(nav_global, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B,C,L,1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B,L,C)

        if self.nav_query_field == "dist":
            sigma = float(self.nav_query_dist_sigma)
            sigma = 1.0 if sigma <= 0.0 else sigma
            sampled = torch.tanh(sampled / sigma)
        elif self.nav_query_field == "count":
            sampled = torch.log1p(torch.clamp_min(sampled, 0.0))

        if sampled.shape[-1] != 1:
            sampled = sampled[..., :1]
        BL = int(sampled.shape[0] * sampled.shape[1])
        c = self.nav_query_mlp(sampled.reshape(BL, 1)).reshape(B, int(sampled.shape[1]), -1)  # (B,L,emb)
        return c.mean(dim=1)  # (B,emb)

    def _apply_nav_emb(self, obs: torch.Tensor, cond: torch.Tensor, nav_emb: torch.Tensor) -> torch.Tensor:
        if self.nav_emb_scale != 1.0:
            nav_emb = nav_emb * self.nav_emb_scale
        if self.nav_gate is not None:
            B = obs.shape[0]
            gate_in = torch.cat([obs.reshape(B, -1), cond], dim=-1)
            gate = torch.sigmoid(self.nav_gate(gate_in))  # (B, 1) in (0, 1)
            if self.nav_gate_dropout > 0.0:
                gate = F.dropout(gate, p=float(self.nav_gate_dropout), training=self.training)
            nav_emb = nav_emb * gate
        if self.nav_emb_dropout > 0.0:
            nav_emb = F.dropout(nav_emb, p=float(self.nav_emb_dropout), training=self.training)
        return nav_emb

    def compute_loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        target: torch.Tensor,
        *,
        nav_patch: torch.Tensor,
        nav_global: Optional[torch.Tensor] = None,
        sample_weight: Optional[torch.Tensor] = None,
        return_x0_pred: bool = False,
        return_timesteps: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Compute diffusion loss for physics-conditioned model.

        Args:
            obs: (B, H, 4)
            cond: (B, 6)
            target: (B, F, 2)
            nav_patch: (B, 3, K, K)
            sample_weight: optional per-sample weight for diffusion loss (shape: B,)
            return_x0_pred: if True, also return x0_pred (B, act_dim, F)
            return_timesteps: if True, also return timesteps (B,)
        """
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model")

        nav_emb = self.nav_encoder(nav_patch)  # (B, nav_emb_dim)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        full_cond = torch.cat([cond, nav_emb], dim=-1)

        cond_emb_extra_fn = None
        if self.nav_query_mode != "none":
            if nav_global is None:
                raise ValueError("nav_query is enabled but nav_global is None")
            def cond_emb_extra_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
                return self._nav_query_emb(x_t, nav_global=nav_global)

        return self.diffusion.compute_loss(
            obs,
            full_cond,
            target,
            sample_weight=sample_weight,
            cond_emb_extra_fn=cond_emb_extra_fn,
            return_x0_pred=return_x0_pred,
            return_timesteps=return_timesteps,
        )

    def forward(self, obs, cond, target=None, nav_patch=None, sample_weight: Optional[torch.Tensor] = None):
        """
        obs: (B, H, 4)
        cond: (B, 6)
        target: (B, F, 2)
        nav_patch: (B, 3, K, K)
        """
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model")
            
        # Encode Nav
        nav_emb = self.nav_encoder(nav_patch) # (B, nav_emb_dim)
        nav_emb = self._apply_nav_emb(obs, cond, nav_emb)
        
        # Concat Cond
        full_cond = torch.cat([cond, nav_emb], dim=-1)
        
        return self.diffusion.forward(obs, full_cond, target, sample_weight=sample_weight)
        
    def sample_trajectory(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor,
        horizon: int,
        nav_patch: Optional[torch.Tensor] = None,
        nav_global: Optional[torch.Tensor] = None,
        *,
        cond_uncond: Optional[torch.Tensor] = None,
        cfg_scale: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        if nav_patch is None:
            raise ValueError("Nav Patch is required for Physics Model inference")
            
        # Encode Nav (shared base)
        nav_emb_base = self.nav_encoder(nav_patch)

        nav_emb = self._apply_nav_emb(obs, cond, nav_emb_base)
        full_cond = torch.cat([cond, nav_emb], dim=-1)

        full_cond_uncond = None
        if cond_uncond is not None:
            nav_emb_u = self._apply_nav_emb(obs, cond_uncond, nav_emb_base)
            full_cond_uncond = torch.cat([cond_uncond, nav_emb_u], dim=-1)

        cond_emb_extra_fn = None
        if self.nav_query_mode != "none":
            if nav_global is None:
                raise ValueError("nav_query is enabled but nav_global is None")
            def cond_emb_extra_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
                return self._nav_query_emb(x_t, nav_global=nav_global)

        return self.diffusion.sample_trajectory(
            obs,
            full_cond,
            horizon,
            cond_uncond=full_cond_uncond,
            cfg_scale=float(cfg_scale),
            cond_emb_extra_fn=cond_emb_extra_fn,
            **kwargs,
        )

    def to(self, device):
        super().to(device)
        self.nav_encoder.to(device)
        self.diffusion.to(device)
        return self
