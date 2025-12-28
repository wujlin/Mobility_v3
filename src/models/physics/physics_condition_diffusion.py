import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from src.models.base_model import BaseTrajectoryModel
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.cnn_encoder import CNNEncoder
from src.models.physics.nav_controlnet import NavControlNet2D

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
                 nav_control: str = "none",
                 nav_control_scale: float = 1.0,
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

        nav_control = str(nav_control)
        if nav_control not in ("none", "controlnet"):
            raise ValueError(f"Unknown nav_control: {nav_control} (expected: none|controlnet)")
        self.nav_control_mode = nav_control
        self.nav_control_scale = float(nav_control_scale)
        self.nav_controlnet: Optional[NavControlNet2D] = None
        if self.nav_control_mode != "none":
            # Match UNet1D down block channels: model_dim * (1,2,4)
            base = int(hidden_dim)
            self.nav_controlnet = NavControlNet2D(in_channels=3, channels=[base, base * 2, base * 4])
        # Buffers used by nav_query to map normalized coords -> grid coords for global field sampling.
        # Register first to avoid "attribute already exists" errors when later assigning.
        self.register_buffer("pos_min_buf", None, persistent=True)
        self.register_buffer("pos_range_buf", None, persistent=True)
        if pos_min is not None and pos_range is not None:
            self.pos_min_buf = torch.tensor(pos_min, dtype=torch.float32)
            self.pos_range_buf = torch.tensor(pos_range, dtype=torch.float32)

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

    def _control_down_mid(
        self,
        x_t: torch.Tensor,  # (B,2,L) normalized pos (noisy)
        *,
        start_grid: torch.Tensor,  # (B,2) [y,x] in grid coords
        control_maps: Tuple[torch.Tensor, ...],  # list of (B,C,H,W) from nav_controlnet
        patch_size: int,
    ) -> Tuple[list[torch.Tensor], torch.Tensor]:
        if self.nav_controlnet is None:
            raise RuntimeError("_control_down_mid called but nav_controlnet is None")

        B = int(x_t.shape[0])
        L0 = int(x_t.shape[2])
        z = x_t.permute(0, 2, 1)  # (B,L,2)
        z = torch.clamp(z, -1.0, 1.0)

        if self.pos_min_buf is None or self.pos_range_buf is None:
            raise RuntimeError("nav_control requires pos_min/pos_range buffers (pass pos_min,pos_range when constructing the model).")
        pos_min = self.pos_min_buf.to(device=x_t.device, dtype=z.dtype)
        pos_range = self.pos_range_buf.to(device=x_t.device, dtype=z.dtype)
        pos_grid = (z + 1.0) * 0.5 * pos_range[None, None, :] + pos_min[None, None, :]  # (B,L,2) [y,x]

        # Global -> patch coordinates (centered at start)
        r = float(int(patch_size) // 2)
        rel = pos_grid - start_grid[:, None, :]  # (B,L,2)
        patch_xy = rel + r  # (B,L,2) in patch pixel coords

        # Compute expected UNet lengths at each down stage (matches Conv1d stride=2,k=3,p=1).
        def _down_len(n: int) -> int:
            return int((int(n) + 1) // 2)

        stage_lens = [int(L0)]
        for _ in range(len(control_maps)):
            stage_lens.append(_down_len(stage_lens[-1]))

        control_down: list[torch.Tensor] = []
        for i, fmap in enumerate(control_maps):
            # fmap: (B,C,H,W) where H,W ~= patch_size / 2^i
            H = int(fmap.shape[2])
            W = int(fmap.shape[3])
            scale = float(2 ** i)
            coords = patch_xy / scale  # (B,L,2) in fmap coords
            x = coords[:, :, 1]
            y = coords[:, :, 0]
            x_n = (x / max(float(W - 1), 1.0)) * 2.0 - 1.0
            y_n = (y / max(float(H - 1), 1.0)) * 2.0 - 1.0
            grid = torch.stack([x_n, y_n], dim=-1).unsqueeze(2)  # (B,L,1,2)
            sampled = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B,C,L,1)
            ctrl = sampled.squeeze(-1)  # (B,C,L)
            if int(ctrl.shape[2]) != int(stage_lens[i]):
                ctrl = F.interpolate(ctrl, size=int(stage_lens[i]), mode="linear", align_corners=False)
            control_down.append(ctrl)

        # Mid control uses deepest fmap
        mid_fmap = control_maps[-1]
        Hm = int(mid_fmap.shape[2])
        Wm = int(mid_fmap.shape[3])
        scale_m = float(2 ** (len(control_maps) - 1))
        coords_m = patch_xy / scale_m
        xm = coords_m[:, :, 1]
        ym = coords_m[:, :, 0]
        xm_n = (xm / max(float(Wm - 1), 1.0)) * 2.0 - 1.0
        ym_n = (ym / max(float(Hm - 1), 1.0)) * 2.0 - 1.0
        grid_m = torch.stack([xm_n, ym_n], dim=-1).unsqueeze(2)
        sampled_m = F.grid_sample(mid_fmap, grid_m, mode="bilinear", padding_mode="zeros", align_corners=True)  # (B,C,L,1)
        ctrl_mid = sampled_m.squeeze(-1)  # (B,C,L)
        mid_len = int(stage_lens[-1])
        if int(ctrl_mid.shape[2]) != mid_len:
            ctrl_mid = F.interpolate(ctrl_mid, size=mid_len, mode="linear", align_corners=False)

        if self.nav_control_scale != 1.0:
            s = float(self.nav_control_scale)
            control_down = [c * s for c in control_down]
            ctrl_mid = ctrl_mid * s

        return control_down, ctrl_mid

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
        unet_kwargs_fn = None
        if self.nav_query_mode != "none":
            if nav_global is None:
                raise ValueError("nav_query is enabled but nav_global is None")
            def cond_emb_extra_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
                return self._nav_query_emb(x_t, nav_global=nav_global)
        if self.nav_control_mode != "none":
            if self.nav_controlnet is None:
                raise RuntimeError("nav_control is enabled but nav_controlnet is None")
            if self.pos_min_buf is None or self.pos_range_buf is None:
                raise RuntimeError("nav_control requires pos_min/pos_range buffers.")
            # Precompute control maps once (nav_patch is static per call); sample by x_t inside the closure.
            control_maps = tuple(self.nav_controlnet(nav_patch))
            start_norm = obs[:, -1, :2]  # (B,2) normalized pos
            pos_min = self.pos_min_buf.to(device=obs.device, dtype=start_norm.dtype)
            pos_range = self.pos_range_buf.to(device=obs.device, dtype=start_norm.dtype)
            start_grid = (start_norm + 1.0) * 0.5 * pos_range[None, :] + pos_min[None, :]  # (B,2)

            def unet_kwargs_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, Any]:
                cd, cm = self._control_down_mid(
                    x_t,
                    start_grid=start_grid,
                    control_maps=control_maps,
                    patch_size=int(nav_patch.shape[-1]),
                )
                return {"control_down": cd, "control_mid": cm}

        return self.diffusion.compute_loss(
            obs,
            full_cond,
            target,
            sample_weight=sample_weight,
            cond_emb_extra_fn=cond_emb_extra_fn,
            unet_kwargs_fn=unet_kwargs_fn,
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
        unet_kwargs_fn = None
        if self.nav_query_mode != "none":
            if nav_global is None:
                raise ValueError("nav_query is enabled but nav_global is None")
            def cond_emb_extra_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
                return self._nav_query_emb(x_t, nav_global=nav_global)
        if self.nav_control_mode != "none":
            if self.nav_controlnet is None:
                raise RuntimeError("nav_control is enabled but nav_controlnet is None")
            if self.pos_min_buf is None or self.pos_range_buf is None:
                raise RuntimeError("nav_control requires pos_min/pos_range buffers.")
            control_maps = tuple(self.nav_controlnet(nav_patch))
            start_norm = obs[:, -1, :2]
            pos_min = self.pos_min_buf.to(device=obs.device, dtype=start_norm.dtype)
            pos_range = self.pos_range_buf.to(device=obs.device, dtype=start_norm.dtype)
            start_grid = (start_norm + 1.0) * 0.5 * pos_range[None, :] + pos_min[None, :]

            def unet_kwargs_fn(x_t: torch.Tensor, timesteps: torch.Tensor) -> Dict[str, Any]:
                cd, cm = self._control_down_mid(
                    x_t,
                    start_grid=start_grid,
                    control_maps=control_maps,
                    patch_size=int(nav_patch.shape[-1]),
                )
                return {"control_down": cd, "control_mid": cm}

        return self.diffusion.sample_trajectory(
            obs,
            full_cond,
            horizon,
            cond_uncond=full_cond_uncond,
            cfg_scale=float(cfg_scale),
            cond_emb_extra_fn=cond_emb_extra_fn,
            unet_kwargs_fn=unet_kwargs_fn,
            **kwargs,
        )

    def to(self, device):
        super().to(device)
        self.nav_encoder.to(device)
        self.diffusion.to(device)
        return self
