from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch


PriorMode = Literal["none", "checkpoint", "skeleton_wp"]
CondMode = Literal["trip_od", "oracle_wp_end"]


@dataclass(frozen=True)
class CondSpec:
    cond_mode: CondMode = "trip_od"
    num_waypoints: int = 2

    @property
    def cond_dim(self) -> int:
        if self.cond_mode == "trip_od":
            return 6
        if self.cond_mode == "oracle_wp_end":
            # [hour, day] + K*2 waypoint pos + [end_y, end_x]
            return 2 + 2 * int(self.num_waypoints) + 2
        raise ValueError(f"Unknown cond_mode: {self.cond_mode}")


def _denormalize_pos(pos_norm: torch.Tensor, *, pos_min: torch.Tensor, pos_range: torch.Tensor) -> torch.Tensor:
    # [-1,1] -> [pos_min, pos_max]
    return (pos_norm + 1.0) * 0.5 * pos_range + pos_min


def _normalize_vel(vel: torch.Tensor, *, vel_mean: torch.Tensor, vel_std: torch.Tensor) -> torch.Tensor:
    return (vel - vel_mean) / vel_std


def split_wp_end_from_cond(cond: torch.Tensor, *, num_waypoints: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    cond layout (oracle_wp_end):
      [hour, day, wp1_y, wp1_x, ..., wpK_y, wpK_x, end_y, end_x]

    Returns:
      waypoints_norm: (B,K,2)
      end_norm: (B,2)
    """
    if cond.ndim != 2:
        raise ValueError(f"Expected cond (B,C), got {tuple(cond.shape)}")
    k = int(num_waypoints)
    need = 2 + 2 * k + 2
    if int(cond.shape[1]) != int(need):
        raise ValueError(f"oracle_wp_end cond_dim mismatch: expected {need}, got {int(cond.shape[1])}")
    wp = cond[:, 2 : 2 + 2 * k].reshape(cond.shape[0], k, 2)
    end = cond[:, 2 + 2 * k : 2 + 2 * k + 2]
    return wp, end


def build_skeleton_prior_vel_norm_k2(
    *,
    obs: torch.Tensor,  # (B,H,4) normalized
    cond: torch.Tensor,  # (B, 2+2*K+2) normalized (oracle_wp_end)
    pred_len: int,
    num_waypoints: int,
    pos_min: torch.Tensor,  # (2,)
    pos_range: torch.Tensor,  # (2,)
    vel_mean: torch.Tensor,  # (2,)
    vel_std: torch.Tensor,  # (2,)
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Deterministic skeleton prior in normalized velocity space.

    - Uses only (start_pos from obs) + (oracle waypoints + end from cond).
    - Skeleton geometry: start -> wp1 -> wp2 -> end.
    - Time parameterization: equal arc-length increments along the polyline.

    Returns:
      prior_vel_norm: (B, F, 2)
    """
    k = int(num_waypoints)
    if k != 2:
        raise NotImplementedError("Currently only supports num_waypoints=2 (KISS).")
    if obs.ndim != 3 or obs.shape[-1] < 2:
        raise ValueError(f"Expected obs (B,H,>=2), got {tuple(obs.shape)}")
    B = int(obs.shape[0])
    F = int(pred_len)
    if F <= 0:
        raise ValueError(f"pred_len must be > 0, got {F}")

    start_norm = obs[:, -1, :2]  # (B,2)
    wp_norm, end_norm = split_wp_end_from_cond(cond, num_waypoints=k)

    start = _denormalize_pos(start_norm, pos_min=pos_min, pos_range=pos_range)  # (B,2)
    wp = _denormalize_pos(wp_norm, pos_min=pos_min, pos_range=pos_range)  # (B,2,2)
    end = _denormalize_pos(end_norm, pos_min=pos_min, pos_range=pos_range)  # (B,2)

    # vertices: (B,4,2)
    v0 = start[:, None, :]
    v3 = end[:, None, :]
    vertices = torch.cat([v0, wp, v3], dim=1)
    seg = vertices[:, 1:, :] - vertices[:, :-1, :]  # (B,3,2)
    seg_len = torch.linalg.norm(seg, dim=-1)  # (B,3)

    s1 = seg_len[:, 0:1]  # (B,1)
    s2 = (seg_len[:, 0] + seg_len[:, 1])[:, None]  # (B,1)
    total = (seg_len[:, 0] + seg_len[:, 1] + seg_len[:, 2])[:, None]  # (B,1)

    t = (torch.arange(1, F + 1, device=obs.device, dtype=obs.dtype) / float(F))[None, :]  # (1,F)
    s = total * t  # (B,F)

    # segment masks
    m0 = s <= s1
    m1 = (s > s1) & (s <= s2)
    # m2 = else

    # avoid /0
    d0 = torch.clamp_min(seg_len[:, 0:1], float(eps))
    d1 = torch.clamp_min(seg_len[:, 1:2], float(eps))
    d2 = torch.clamp_min(seg_len[:, 2:3], float(eps))

    a0 = s / d0  # (B,F)
    a1 = (s - s1) / d1
    a2 = (s - s2) / d2

    p0 = vertices[:, 0:1, :] + a0[:, :, None] * seg[:, 0:1, :]  # (B,F,2)
    p1 = vertices[:, 1:2, :] + a1[:, :, None] * seg[:, 1:2, :]
    p2 = vertices[:, 2:3, :] + a2[:, :, None] * seg[:, 2:3, :]

    pos = torch.where(m0[:, :, None], p0, torch.where(m1[:, :, None], p1, p2))  # (B,F,2)

    vel = torch.zeros((B, F, 2), device=obs.device, dtype=obs.dtype)
    vel[:, 0, :] = pos[:, 0, :] - start
    if F > 1:
        vel[:, 1:, :] = pos[:, 1:, :] - pos[:, :-1, :]

    # ---- Corner smoothing (KISS) ----
    # Linear skeleton has sharp turns at waypoint connections, which can create
    # impulsive acceleration profiles. We smooth velocities only in a 1-step
    # neighborhood around segment boundaries, then apply a tiny global correction
    # to keep the final endpoint fixed.
    if F >= 3:
        seg_id = m1.to(torch.int64) + (~(m0 | m1)).to(torch.int64) * 2  # 0/1/2 for seg0/seg1/seg2
        boundary = seg_id[:, 1:] != seg_id[:, :-1]  # (B,F-1), boundary at pos index i in [1..F-1]
        idx_pos = torch.arange(1, F, device=obs.device, dtype=torch.int64)[None, :]  # (1,F-1)
        big = torch.full((B, F - 1), F + 1, device=obs.device, dtype=torch.int64)
        idx1 = torch.where(boundary, idx_pos, big).min(dim=1).values  # (B,)
        idx2 = torch.where(boundary & (idx_pos > idx1[:, None]), idx_pos, big).min(dim=1).values  # (B,)

        vel_s = vel.clone()
        b = torch.arange(B, device=obs.device, dtype=torch.int64)
        valid_mask = total[:, 0] > float(eps)

        def _smooth_at(step_idx: torch.Tensor) -> None:
            # Smooth vel at step_idx and step_idx+1 using 3-tap averaging.
            i = step_idx.to(torch.int64)

            m = valid_mask & (i >= 1) & (i <= F - 2)
            if bool(torch.any(m)):
                bb = b[m]
                ii = i[m]
                vel_s[bb, ii] = (vel_s[bb, ii - 1] + vel_s[bb, ii] + vel_s[bb, ii + 1]) / 3.0

            j = i + 1
            m2 = valid_mask & (j >= 1) & (j <= F - 2)
            if bool(torch.any(m2)):
                bb2 = b[m2]
                jj = j[m2]
                vel_s[bb2, jj] = (vel_s[bb2, jj - 1] + vel_s[bb2, jj] + vel_s[bb2, jj + 1]) / 3.0

        _smooth_at(idx1)
        _smooth_at(idx2)

        # Endpoint correction: distribute the residual displacement evenly.
        desired_disp = end - start  # (B,2)
        disp = vel_s.sum(dim=1)  # (B,2)
        delta = (desired_disp - disp) / float(F)
        delta = delta * valid_mask.to(dtype=vel_s.dtype)[:, None]
        vel = vel_s + delta[:, None, :]

    # For degenerate cases (total length ~0), return zeros.
    valid = (total[:, 0] > float(eps)).to(dtype=vel.dtype)[:, None, None]
    vel = vel * valid
    return _normalize_vel(vel, vel_mean=vel_mean, vel_std=vel_std)
