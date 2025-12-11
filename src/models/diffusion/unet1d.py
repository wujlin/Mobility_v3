import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock1D(nn.Module):
    def __init__(self, inp, out, emb_dim=None, dropout=0.0):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(inp, out, 3, padding=1),
            nn.GroupNorm(8, out),
            nn.SiLU(),
        )
        self.emb_proj = nn.Linear(emb_dim, out) if emb_dim else None
        
        self.block2 = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(out, out, 3, padding=1),
            nn.GroupNorm(8, out),
            nn.SiLU(),
        )
        
        self.residual = nn.Identity() if inp == out else nn.Conv1d(inp, out, 1)

    def forward(self, x, emb=None):
        h = self.block1(x)
        
        if self.emb_proj is not None and emb is not None:
             # Add conditioning (B, dim) -> (B, dim, 1) broadcast
             h = h + self.emb_proj(emb)[:, :, None]
        
        h = self.block2(h)
        return h + self.residual(x)

class Downsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
        
    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, 
                 in_dim: int, 
                 model_dim: int = 64, 
                 emb_dim: int = 256, 
                 dim_mults: tuple = (1, 2, 4),
                 cond_dim: int = 0):
        super().__init__()
        
        self.init_conv = nn.Conv1d(in_dim, model_dim, 3, padding=1)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(model_dim),
            nn.Linear(model_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        
        # Condition Projection
        self.cond_proj = nn.Linear(cond_dim, emb_dim) if cond_dim > 0 else None
        
        # Down Path
        self.downs = nn.ModuleList()
        curr_dim = model_dim
        dims = [model_dim]
        
        for mult in dim_mults:
            dim_out = model_dim * mult
            self.downs.append(nn.ModuleList([
                ResBlock1D(curr_dim, dim_out, emb_dim),
                ResBlock1D(dim_out, dim_out, emb_dim),
                Downsample1D(dim_out)
            ]))
            curr_dim = dim_out
            dims.append(curr_dim)
            
        # Mid Path
        mid_dim = dims[-1]
        self.mid_block1 = ResBlock1D(mid_dim, mid_dim, emb_dim)
        self.mid_block2 = ResBlock1D(mid_dim, mid_dim, emb_dim)
        
        # Up Path
        self.ups = nn.ModuleList()
        for mult in reversed(dim_mults):
            dim_out = model_dim * mult
            dim_in = mid_dim + dim_out # Skip connection concatenation logic?
            # Wait, standard unet: 
            # down: x -> d1 -> down -> d2
            # up: u2 + d2 -> ...
            # Actually, `dims` captured structure. 
            # In UNet, we pop from stack.
            
            # Reconstruct correct dims logic
            # input to up is current `mid_dim`
            # we concat with corresponding skip `dim_out` (which was output of ResBlock in Down)
            
            self.ups.append(nn.ModuleList([
                Upsample1D(mid_dim),
                ResBlock1D(mid_dim + dim_out, dim_out, emb_dim),
                ResBlock1D(dim_out, dim_out, emb_dim)
            ]))
            mid_dim = dim_out
            
        self.final_conv = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 3, padding=1),
            nn.GroupNorm(8, model_dim),
            nn.SiLU(),
            nn.Conv1d(model_dim, in_dim, 1)
        )

    def forward(self, x, steps, cond=None):
        """
        x: (B, C, L)
        steps: (B,) time steps
        cond: (B, D) global condition
        """
        # Embed Time
        t = self.time_mlp(steps) # (B, emb_dim)
        
        # Embed Condition
        if self.cond_proj is not None and cond is not None:
            c = self.cond_proj(cond) # (B, emb_dim)
            t = t + c # Additive conditioning
            
        h = self.init_conv(x)
        skips = [h]
        
        # Down
        for res1, res2, down in self.downs:
            h = res1(h, t)
            h = res2(h, t)
            skips.append(h) # Save for skip
            h = down(h)
            
        # Mid
        h = self.mid_block1(h, t)
        h = self.mid_block2(h, t)
        
        # Up
        for up, res1, res2 in self.ups:
            h = up(h)
            skip = skips.pop()
            # Crop/Pad if dimensions differ (e.g. odd length)?
            # For 1D, if L is odd, downsample floor. upsample can match.
            # Assuming L is power of 2 for simplicity.
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='linear', align_corners=False)
                
            h = torch.cat((h, skip), dim=1)
            h = res1(h, t)
            h = res2(h, t)
            
        return self.final_conv(h)
