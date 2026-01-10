from __future__ import annotations

import torch
import torch.nn as nn


class GridCNNEncoder(nn.Module):
    """
    A minimal spatial encoder for OD-aligned raster patches.

    Input:  (B, C, S, S)
    Output: (B, out_dim)
    """

    def __init__(self, *, in_channels: int, out_dim: int = 64) -> None:
        super().__init__()
        c_in = int(in_channels)
        d_out = int(out_dim)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if d_out <= 0:
            raise ValueError("out_dim must be > 0")

        self.in_channels = c_in
        self.out_dim = d_out

        self.net = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)

