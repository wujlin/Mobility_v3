from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PatchFrame:
    """
    OD-aligned patch frame:
      - patch is centered at mid = (start + dest)/2
      - u axis is along OD direction (e_par)
      - v axis is perpendicular (e_perp)
    """

    start_pos: torch.Tensor  # (B,2) in grid coords [y,x]
    dest_pos: torch.Tensor  # (B,2)


def rel_so_to_abs_waypoints(
    *,
    start_pos: torch.Tensor,  # (B,2)
    dest_pos: torch.Tensor,  # (B,2)
    rel: torch.Tensor,  # (B,K,2) (s,o) in chord-normalized units
    eps: float = 1e-6,
) -> torch.Tensor:
    start_pos = start_pos.to(dtype=torch.float32)
    dest_pos = dest_pos.to(dtype=torch.float32)
    rel = rel.to(dtype=torch.float32)
    v = dest_pos - start_pos  # (B,2)
    L = torch.linalg.norm(v, dim=-1, keepdim=True).clamp_min(float(eps))  # (B,1)
    e_par = v / L
    e_perp = torch.stack([-e_par[:, 1], e_par[:, 0]], dim=-1)
    s = rel[..., 0:1]
    o = rel[..., 1:2]
    return start_pos[:, None, :] + (s * L[:, None, :]) * e_par[:, None, :] + (o * L[:, None, :]) * e_perp[:, None, :]


def abs_points_to_patch_grid(
    *,
    start_pos: torch.Tensor,  # (B,2)
    dest_pos: torch.Tensor,  # (B,2)
    points: torch.Tensor,  # (B,T,2) absolute grid coords [y,x]
    extent: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Convert absolute points into grid_sample coordinates for an OD-aligned patch.

    Returns:
      grid: (B, 1, T, 2) with last dim (x_norm, y_norm) in [-1,1]
    """
    points = points.to(dtype=torch.float32)
    start_pos = start_pos.to(dtype=torch.float32)
    dest_pos = dest_pos.to(dtype=torch.float32)

    ext = float(extent)
    if not (ext > 0.0):
        raise ValueError(f"extent must be > 0, got {extent}")

    mid = 0.5 * (start_pos + dest_pos)  # (B,2)
    d = dest_pos - start_pos
    L = torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(float(eps))  # (B,1)
    e_par = d / L
    e_perp = torch.stack([-e_par[:, 1], e_par[:, 0]], dim=-1)

    delta = points - mid[:, None, :]  # (B,T,2)
    u = torch.sum(delta * e_par[:, None, :], dim=-1)  # (B,T)
    v = torch.sum(delta * e_perp[:, None, :], dim=-1)  # (B,T)
    y_norm = u / float(ext)
    x_norm = v / float(ext)
    grid = torch.stack([x_norm, y_norm], dim=-1)  # (B,T,2)
    return grid[:, None, :, :]  # (B,1,T,2)


def sample_patch_at_abs_points(
    *,
    patch: torch.Tensor,  # (B,C,S,S)
    start_pos: torch.Tensor,  # (B,2)
    dest_pos: torch.Tensor,  # (B,2)
    points: torch.Tensor,  # (B,T,2)
    extent: float,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Sample an OD-aligned patch at absolute grid points.

    Returns:
      sem: (B,T,C)
    """
    grid = abs_points_to_patch_grid(start_pos=start_pos, dest_pos=dest_pos, points=points, extent=float(extent))
    out = F.grid_sample(patch, grid, mode=str(mode), padding_mode=str(padding_mode), align_corners=False)  # (B,C,1,T)
    out = out.squeeze(2).transpose(1, 2).contiguous()  # (B,T,C)
    return out


def sample_patch_at_rel_waypoints(
    *,
    patch: torch.Tensor,  # (B,C,S,S)
    start_pos: torch.Tensor,  # (B,2)
    dest_pos: torch.Tensor,  # (B,2)
    rel: torch.Tensor,  # (B,K,2) (s,o)
    extent: float,
) -> torch.Tensor:
    wp = rel_so_to_abs_waypoints(start_pos=start_pos, dest_pos=dest_pos, rel=rel)  # (B,K,2)
    return sample_patch_at_abs_points(patch=patch, start_pos=start_pos, dest_pos=dest_pos, points=wp, extent=float(extent))


def sample_patch_mean_along_skeleton(
    *,
    patch: torch.Tensor,  # (B,C,S,S)
    start_pos: torch.Tensor,  # (B,2)
    dest_pos: torch.Tensor,  # (B,2)
    rel: torch.Tensor,  # (B,K,2)
    extent: float,
    samples_per_segment: int = 8,
) -> torch.Tensor:
    """
    Sample patch values along the polyline: start -> waypoints -> dest.

    Returns:
      mean_sem: (B,C)
    """
    k = int(rel.shape[1])
    wp = rel_so_to_abs_waypoints(start_pos=start_pos, dest_pos=dest_pos, rel=rel)  # (B,K,2)
    verts = torch.cat([start_pos[:, None, :], wp, dest_pos[:, None, :]], dim=1)  # (B,K+2,2)
    segs = int(k + 1)
    m = int(samples_per_segment)
    if m <= 1:
        pts = verts
    else:
        t = torch.linspace(0.0, 1.0, steps=m, device=patch.device, dtype=torch.float32)  # (m,)
        pts_list = []
        for si in range(segs):
            p0 = verts[:, si : si + 1, :]  # (B,1,2)
            p1 = verts[:, si + 1 : si + 2, :]  # (B,1,2)
            # (B,m,2)
            seg_pts = (1.0 - t[None, :, None]) * p0 + t[None, :, None] * p1
            if si > 0:
                seg_pts = seg_pts[:, 1:, :]  # drop duplicate boundary
            pts_list.append(seg_pts)
        pts = torch.cat(pts_list, dim=1)  # (B,T,2)

    sem = sample_patch_at_abs_points(patch=patch, start_pos=start_pos, dest_pos=dest_pos, points=pts, extent=float(extent))
    return sem.mean(dim=1)

