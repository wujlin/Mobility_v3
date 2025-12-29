from __future__ import annotations

import torch
import torch.nn as nn


class MacroHardSupportNet(nn.Module):
    """
    Macro hard-support model: predict per-pixel heatmaps for z=[wp1, wp2, end_anchor]
    on a nav_patch (KxK), conditioned on obs + trip_od.

    Output:
      logits: (B, 3, K, K) unnormalized logits (masking handled outside).
    """

    def __init__(
        self,
        *,
        obs_len: int = 8,
        obs_dim: int = 4,
        cond_dim: int = 6,
        patch_size: int = 64,
        in_channels: int = 3,
        hidden_dim: int = 64,
        use_coord: bool = True,
        cond_mode: str = "film",
    ) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.use_coord = bool(use_coord)
        self.cond_mode = str(cond_mode)
        if self.cond_mode not in {"film", "add"}:
            raise ValueError(f"cond_mode must be one of {{'film','add'}}, got: {self.cond_mode}")

        cin = int(in_channels) + (2 if self.use_coord else 0)
        h = int(hidden_dim)

        g = 8
        g = 1 if h % g != 0 else g

        self.nav_backbone = nn.Sequential(
            nn.Conv2d(cin, h, 3, padding=1),
            nn.GroupNorm(g, h),
            nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1),
            nn.GroupNorm(g, h),
            nn.SiLU(),
            nn.Conv2d(h, h, 3, padding=1),
            nn.GroupNorm(g, h),
            nn.SiLU(),
        )

        in_cond = self.obs_len * self.obs_dim + self.cond_dim
        if self.cond_mode == "film":
            # Feature-wise Linear Modulation (FiLM): conditioning affects per-pixel ordering via channel-wise scale+shift.
            self.cond_mlp = nn.Sequential(
                nn.Linear(in_cond, h),
                nn.SiLU(),
                nn.Linear(h, 2 * h),
            )
            self.post_act = nn.SiLU()
        else:
            # Legacy additive conditioning (kept for backward-compatible checkpoint loading).
            self.cond_mlp = nn.Sequential(
                nn.Linear(in_cond, h),
                nn.SiLU(),
                nn.Linear(h, h),
            )
            self.post_act = None

        self.head = nn.Conv2d(h, 3, 1)

        self.register_buffer("_coord", None, persistent=False)

    def _coord_grid(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k = int(self.patch_size)
        if self._coord is not None and tuple(self._coord.shape) == (2, k, k) and self._coord.device == device and self._coord.dtype == dtype:
            return self._coord
        yy = torch.linspace(-1.0, 1.0, steps=k, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, steps=k, device=device, dtype=dtype)
        yv, xv = torch.meshgrid(yy, xx, indexing="ij")
        coord = torch.stack([yv, xv], dim=0)  # (2,K,K)
        self._coord = coord
        return coord

    def forward(self, *, obs: torch.Tensor, cond: torch.Tensor, nav_patch: torch.Tensor) -> torch.Tensor:
        """
        Args:
          obs: (B,H,4)
          cond: (B,6) trip_od
          nav_patch: (B,3,K,K)
        Returns:
          logits: (B,3,K,K)
        """
        if obs.ndim != 3 or int(obs.shape[1]) != int(self.obs_len) or int(obs.shape[2]) != int(self.obs_dim):
            raise ValueError(f"Expected obs (B,{self.obs_len},{self.obs_dim}), got {tuple(obs.shape)}")
        if cond.ndim != 2 or int(cond.shape[1]) != int(self.cond_dim):
            raise ValueError(f"Expected cond (B,{self.cond_dim}), got {tuple(cond.shape)}")
        if nav_patch.ndim != 4 or int(nav_patch.shape[2]) != int(self.patch_size) or int(nav_patch.shape[3]) != int(self.patch_size):
            raise ValueError(f"Expected nav_patch (B,C,{self.patch_size},{self.patch_size}), got {tuple(nav_patch.shape)}")

        B = int(obs.shape[0])
        flat = torch.cat([obs.reshape(B, -1), cond], dim=-1)
        if self.cond_mode == "film":
            film = self.cond_mlp(flat).view(B, 2 * int(self.hidden_dim), 1, 1)  # (B,2h,1,1)
            scale, shift = film.chunk(2, dim=1)
            scale = torch.tanh(scale)  # (-1,1) => (1+scale) in (0,2)
        else:
            g = self.cond_mlp(flat).view(B, int(self.hidden_dim), 1, 1)  # (B,h,1,1)

        x = nav_patch
        if self.use_coord:
            coord = self._coord_grid(device=nav_patch.device, dtype=nav_patch.dtype).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coord], dim=1)

        feat = self.nav_backbone(x)
        if self.cond_mode == "film":
            feat = feat * (1.0 + scale) + shift
            if self.post_act is not None:
                feat = self.post_act(feat)
        else:
            feat = feat + g
        return self.head(feat)
