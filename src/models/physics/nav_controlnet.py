import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def _zero_init_conv(conv: nn.Conv2d) -> nn.Conv2d:
    nn.init.zeros_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)
    return conv


class _ConvBlock2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, groups: int = 8):
        super().__init__()
        g = int(groups)
        g = 1 if out_ch % g != 0 else g
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), 3, padding=1),
            nn.GroupNorm(g, int(out_ch)),
            nn.SiLU(),
            nn.Conv2d(int(out_ch), int(out_ch), 3, padding=1),
            nn.GroupNorm(g, int(out_ch)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NavControlNet2D(nn.Module):
    """
    ControlNet-style 2D branch for nav_patch (B,C,32,32).

    Produces a multi-scale list of control feature maps aligned with UNet1D down blocks.
    Each stage output is passed through a zero-initialized 1x1 conv (ControlNet "zero conv"),
    so the injected control starts at 0 and is learned during training.
    """

    def __init__(self, *, in_channels: int, channels: List[int]):
        super().__init__()
        if not channels:
            raise ValueError("channels must be non-empty")
        self.channels = [int(c) for c in channels]

        blocks = []
        zero_convs = []
        c_in = int(in_channels)
        for c_out in self.channels:
            blocks.append(_ConvBlock2D(c_in, c_out))
            zero_convs.append(_zero_init_conv(nn.Conv2d(c_out, c_out, 1)))
            c_in = c_out

        self.blocks = nn.ModuleList(blocks)
        self.zero_convs = nn.ModuleList(zero_convs)

    def forward(self, nav_patch: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            nav_patch: (B,C,H,W)
        Returns:
            controls: list of (B, C_i, H_i, W_i), len = len(channels)
        """
        x = nav_patch
        controls: List[torch.Tensor] = []
        for i, (blk, zc) in enumerate(zip(self.blocks, self.zero_convs)):
            x = blk(x)
            controls.append(zc(x))
            if i != len(self.blocks) - 1:
                x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return controls

