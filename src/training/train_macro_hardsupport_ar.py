from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

try:  # Optional; used for fast global nearest-drivable projection of GT labels.
    from scipy import ndimage  # type: ignore
except Exception:  # pragma: no cover
    ndimage = None

from src.data.datasets_diffusion import DiffusionDataset
from src.models.macro.macro_hardsupport_ar import MacroHardSupportARNet


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train Macro Hard Support (AR): wp1->wp2->end heatmaps with teacher forcing.")
    p.add_argument("--exp_name", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--nav_file", type=str, required=True)
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    p.add_argument("--splits_dir", type=str, default=None)

    p.add_argument("--obs_len", type=int, default=8)
    p.add_argument("--pred_len", type=int, default=12, help="micro horizon used to extract oracle wp/end in dataset")

    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--nav_patch_channel2", type=str, choices=["count"], default="count")
    p.add_argument("--count_thr", type=float, default=1.0, help="Strict drivable definition (must match G1 gate).")

    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--use_coord", action="store_true")

    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_batches", type=int, default=None)
    p.add_argument("--log_every", type=int, default=100)

    p.add_argument("--label_project", action="store_true", help="Project GT z to nearest global drivable cell (requires scipy).")
    return p


def _make_global_projector(nav_count: np.ndarray, *, count_thr: float) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ndimage is None:
        return None
    drivable = np.asarray(nav_count >= float(count_thr), dtype=bool)
    offroad = ~drivable
    _, (iy, ix) = ndimage.distance_transform_edt(offroad, return_indices=True)
    return iy.astype(np.int64, copy=False), ix.astype(np.int64, copy=False)


def _denorm_pos(pos_norm: torch.Tensor, *, pos_min: torch.Tensor, pos_range: torch.Tensor) -> torch.Tensor:
    return (pos_norm + 1.0) * 0.5 * pos_range + pos_min


def _labels_from_z_grid(
    *,
    z_grid: np.ndarray,  # (B,3,2) float grid [y,x]
    start_pos_grid: np.ndarray,  # (B,2) float grid [y,x]
    patch_size: int,
    nav_count: np.ndarray,  # (H,W)
    count_thr: float,
    projector: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert GT z (global grid) -> patch pixel labels (y,x) with optional global nearest-drivable projection.
    Returns:
      labels_yx: (B,3,2) int64, -1 for invalid/out-of-patch
      valid: (B,3) bool
    """
    B = int(z_grid.shape[0])
    K = int(patch_size)
    r = int(K // 2)
    H, W = int(nav_count.shape[0]), int(nav_count.shape[1])

    z_ij = np.rint(z_grid).astype(np.int64)  # (B,3,2)
    yy = np.clip(z_ij[..., 0], 0, H - 1)
    xx = np.clip(z_ij[..., 1], 0, W - 1)
    drv = (nav_count[yy, xx] >= float(count_thr))
    if projector is not None:
        iy, ix = projector
        bad = ~drv
        if np.any(bad):
            py = iy[yy, xx]
            px = ix[yy, xx]
            yy = np.where(bad, py, yy)
            xx = np.where(bad, px, xx)
            drv = np.ones_like(drv, dtype=bool)

    center = np.floor(start_pos_grid).astype(np.int64)  # (B,2)
    cy = center[:, 0:1]
    cx = center[:, 1:2]
    py = yy - cy + r
    px = xx - cx + r
    inb = (py >= 0) & (py < K) & (px >= 0) & (px < K)
    valid = inb & drv
    labels = np.stack([py, px], axis=-1).astype(np.int64)
    labels[~valid] = -1
    return labels, valid


def _one_hot_map(y: torch.Tensor, x: torch.Tensor, valid: torch.Tensor, *, K: int) -> torch.Tensor:
    """
    Build one-hot map in patch pixels.
    y,x: (B,) int64
    valid: (B,) bool
    Returns: (B,K,K) float32
    """
    B = int(y.shape[0])
    m = torch.zeros((B, int(K), int(K)), device=y.device, dtype=torch.float32)
    if bool(torch.any(valid)):
        idx = torch.arange(B, device=y.device, dtype=torch.int64)[valid]
        yy = y[valid].clamp(0, int(K) - 1)
        xx = x[valid].clamp(0, int(K) - 1)
        m[idx, yy, xx] = 1.0
    return m


def main() -> None:
    args = build_argparser().parse_args()
    _set_seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using seed: {int(args.seed)}")

    traj_ids = None
    if str(args.split) != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    dataset = DiffusionDataset(
        args.data_path,
        obs_len=int(args.obs_len),
        pred_len=int(args.pred_len),
        nav_field_file=str(args.nav_file),
        nav_patch_size=int(args.patch_size),
        nav_patch_channel2=str(args.nav_patch_channel2),
        traj_ids=traj_ids,
        cond_mode="oracle_wp_end",
        waypoint_mode="rdp_dev",
        num_waypoints=2,
    )

    nav_count = dataset.nav_field.count if dataset.nav_field is not None else None
    if nav_count is None:
        raise RuntimeError("nav_field.npz must contain count for hard-support masking/label projection.")

    if bool(args.label_project) and ndimage is None:
        raise ImportError("--label_project requires scipy (missing scipy.ndimage).")
    projector = _make_global_projector(np.asarray(nav_count, dtype=np.float32), count_thr=float(args.count_thr)) if bool(args.label_project) else None

    pos_min = torch.tensor(dataset.normalizer.pos_min, dtype=torch.float32, device=device)
    pos_range = torch.tensor(dataset.normalizer.pos_range, dtype=torch.float32, device=device)

    thr_norm = float(np.log1p(float(args.count_thr)) / float(getattr(dataset, "_nav_count_log1p_max", 1.0)))

    g = torch.Generator()
    g.manual_seed(int(args.seed))
    pin_memory = bool(torch.cuda.is_available())
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
        pin_memory=pin_memory,
        persistent_workers=(int(args.num_workers) > 0),
    )

    model = MacroHardSupportARNet(
        obs_len=int(args.obs_len),
        obs_dim=4,
        cond_dim=6,
        patch_size=int(args.patch_size),
        hidden_dim=int(args.hidden_dim),
        use_coord=bool(args.use_coord),
    ).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path("data/experiments") / str(args.exp_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_type": "macro_hardsupport_ar_pixel",
        "data_path": str(args.data_path),
        "nav_file": str(args.nav_file),
        "split": str(args.split),
        "splits_dir": (str(args.splits_dir) if args.splits_dir else None),
        "obs_len": int(args.obs_len),
        "pred_len_micro": int(args.pred_len),
        "patch_size": int(args.patch_size),
        "nav_patch_channel2": str(args.nav_patch_channel2),
        "count_thr": float(args.count_thr),
        "hidden_dim": int(args.hidden_dim),
        "use_coord": bool(args.use_coord),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
        "max_batches": (int(args.max_batches) if args.max_batches is not None else None),
        "label_project": bool(args.label_project),
        "thr_norm": float(thr_norm),
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    K = int(args.patch_size)
    for epoch in range(int(args.epochs)):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        nb = 0

        for bidx, batch in enumerate(loader):
            obs = batch["obs"].to(device)
            cond_oracle = batch["cond"].to(device)  # (B,8) [hour,day,wp1,wp2,end] normalized pos
            nav_patch = batch["nav_patch"].to(device)  # (B,3,K,K)
            trip_o = batch["trip_o"].to(device)
            trip_d = batch["trip_d"].to(device)

            B = int(obs.shape[0])
            if B == 0:
                continue

            cond_trip_od = torch.cat([cond_oracle[:, :2], trip_o, trip_d], dim=-1)  # (B,6)
            z_norm = cond_oracle[:, 2:].view(B, 3, 2)  # (B,3,2)

            start_pos = _denorm_pos(obs[:, -1, :2], pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)
            z_grid = _denorm_pos(z_norm, pos_min=pos_min, pos_range=pos_range).detach().cpu().numpy().astype(np.float32, copy=False)

            labels_yx, valid = _labels_from_z_grid(
                z_grid=z_grid,
                start_pos_grid=start_pos,
                patch_size=K,
                nav_count=np.asarray(nav_count, dtype=np.float32),
                count_thr=float(args.count_thr),
                projector=projector,
            )  # (B,3,2), (B,3)

            labels_y = torch.from_numpy(labels_yx[..., 0]).to(device=device, dtype=torch.long)
            labels_x = torch.from_numpy(labels_yx[..., 1]).to(device=device, dtype=torch.long)
            valid_global = torch.from_numpy(valid).to(device=device)

            strict = (nav_patch[:, 2] >= float(thr_norm))  # (B,K,K)
            empty = (strict.view(B, -1).sum(dim=1) == 0)
            if bool(torch.any(empty)):
                strict[empty] = True
                valid_global[empty] = False

            by = torch.arange(B, device=device).view(B, 1).expand(B, 3)
            ly = labels_y.clamp(0, K - 1)
            lx = labels_x.clamp(0, K - 1)
            label_in_strict = strict[by, ly, lx]
            mismatch = valid_global & (~label_in_strict)
            valid_t = valid_global & label_in_strict
            if not bool(torch.any(valid_t)):
                continue

            prev0 = torch.zeros((B, 2, K, K), device=device, dtype=torch.float32)

            wp1_map = _one_hot_map(labels_y[:, 0], labels_x[:, 0], valid_t[:, 0], K=K)
            prev1 = torch.stack([wp1_map, torch.zeros_like(wp1_map)], dim=1)

            wp2_map = _one_hot_map(labels_y[:, 1], labels_x[:, 1], valid_t[:, 1], K=K)
            prev2 = torch.stack([wp1_map, wp2_map], dim=1)

            logits0 = model(obs=obs, cond=cond_trip_od, nav_patch=nav_patch, prev_maps=prev0).masked_fill(~strict, -1e9)
            logits1 = model(obs=obs, cond=cond_trip_od, nav_patch=nav_patch, prev_maps=prev1).masked_fill(~strict, -1e9)
            logits2 = model(obs=obs, cond=cond_trip_od, nav_patch=nav_patch, prev_maps=prev2).masked_fill(~strict, -1e9)

            losses = []
            for si, logits in enumerate([logits0, logits1, logits2]):
                li = labels_y[:, si] * K + labels_x[:, si]
                li = torch.where(valid_t[:, si], li, torch.full_like(li, -1))
                if bool(torch.any(li != -1)):
                    losses.append(F.cross_entropy(logits.view(B, K * K), li, ignore_index=-1))
            if not losses:
                continue
            loss = sum(losses) / float(len(losses))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            nb += 1

            if int(args.log_every) > 0 and (bidx % int(args.log_every) == 0):
                with torch.no_grad():
                    vr = float(valid_t.to(torch.float32).mean().item())
                    mr = float(mismatch.to(torch.float32).mean().item())
                    v0 = float(valid_t[:, 0].to(torch.float32).mean().item())
                    v1 = float(valid_t[:, 1].to(torch.float32).mean().item())
                    v2 = float(valid_t[:, 2].to(torch.float32).mean().item())
                    print(
                        f"Epoch {epoch} | Batch {bidx} | Loss {loss.item():.4f} | "
                        f"thr_norm={thr_norm:.4f} valid={vr:.4f} (wp1={v0:.4f},wp2={v1:.4f},end={v2:.4f}) mismatch={mr:.4f}"
                    )

            if args.max_batches is not None and int(bidx) + 1 >= int(args.max_batches):
                break

        avg = total_loss / max(nb, 1)
        dt = time.time() - t0
        print(f"Epoch {epoch} Done. Avg Loss: {avg:.4f}. Time: {dt:.1f}s")

        ckpt = {"model_state_dict": model.state_dict(), "config": run_config, "epoch": int(epoch)}
        torch.save(ckpt, save_dir / "last.pt")


if __name__ == "__main__":
    main()

