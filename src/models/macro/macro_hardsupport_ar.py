from __future__ import annotations

import torch
import torch.nn as nn


class MacroHardSupportARNet(nn.Module):
    """
    Autoregressive Macro hard-support model (pixel heatmap).

    Predicts ONE waypoint heatmap at a time, conditioned on:
      - obs (history)
      - trip_od (hour/day + trip_o + trip_d)
      - nav_patch (dir_y, dir_x, count_norm)
      - prev_maps: (wp1_map, wp2_map) in patch pixels (one-hot), enabling AR dependency

    Output:
      logits: (B, K, K)
    """

    def __init__(
        self,
        *,
        obs_len: int = 8,
        obs_dim: int = 4,
        cond_dim: int = 6,
        patch_size: int = 64,
        nav_channels: int = 3,
        prev_channels: int = 2,
        hidden_dim: int = 64,
        use_coord: bool = True,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
    ) -> None:
        super().__init__()
        self.obs_len = int(obs_len)
        self.obs_dim = int(obs_dim)
        self.cond_dim = int(cond_dim)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.use_coord = bool(use_coord)

        cin = int(nav_channels) + int(prev_channels) + (2 if self.use_coord else 0)
        h = int(hidden_dim)

        g = 8
        g = 1 if h % g != 0 else g

        layers: list[nn.Module] = []
        in_c = cin
        for d in tuple(int(x) for x in dilations):
            layers.append(nn.Conv2d(in_c, h, 3, padding=d, dilation=d))
            layers.append(nn.GroupNorm(g, h))
            layers.append(nn.SiLU())
            in_c = h
        self.nav_backbone = nn.Sequential(*layers)

        in_cond = self.obs_len * self.obs_dim + self.cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(in_cond, h),
            nn.SiLU(),
            nn.Linear(h, 2 * h),
        )
        self.post_act = nn.SiLU()

        self.head = nn.Conv2d(h, 1, 1)
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

    def forward(
        self,
        *,
        obs: torch.Tensor,  # (B,H,4)
        cond: torch.Tensor,  # (B,6)
        nav_patch: torch.Tensor,  # (B,3,K,K)
        prev_maps: torch.Tensor,  # (B,2,K,K)
    ) -> torch.Tensor:
        if obs.ndim != 3 or int(obs.shape[1]) != int(self.obs_len) or int(obs.shape[2]) != int(self.obs_dim):
            raise ValueError(f"Expected obs (B,{self.obs_len},{self.obs_dim}), got {tuple(obs.shape)}")
        if cond.ndim != 2 or int(cond.shape[1]) != int(self.cond_dim):
            raise ValueError(f"Expected cond (B,{self.cond_dim}), got {tuple(cond.shape)}")
        if nav_patch.ndim != 4 or int(nav_patch.shape[2]) != int(self.patch_size) or int(nav_patch.shape[3]) != int(self.patch_size):
            raise ValueError(f"Expected nav_patch (B,3,{self.patch_size},{self.patch_size}), got {tuple(nav_patch.shape)}")
        if prev_maps.ndim != 4 or int(prev_maps.shape[1]) != 2 or int(prev_maps.shape[2]) != int(self.patch_size) or int(prev_maps.shape[3]) != int(self.patch_size):
            raise ValueError(f"Expected prev_maps (B,2,{self.patch_size},{self.patch_size}), got {tuple(prev_maps.shape)}")

        B = int(obs.shape[0])
        flat = torch.cat([obs.reshape(B, -1), cond], dim=-1)
        film = self.cond_mlp(flat).view(B, 2 * int(self.hidden_dim), 1, 1)
        scale, shift = film.chunk(2, dim=1)
        scale = torch.tanh(scale)

        x = torch.cat([nav_patch, prev_maps], dim=1)
        if self.use_coord:
            coord = self._coord_grid(device=nav_patch.device, dtype=nav_patch.dtype).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coord], dim=1)

        feat = self.nav_backbone(x)
        feat = feat * (1.0 + scale) + shift
        feat = self.post_act(feat)
        logits = self.head(feat).squeeze(1)  # (B,K,K)
        return logits

