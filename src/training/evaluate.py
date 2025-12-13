import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.models.seq.seq_baseline import SeqBaseline
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.data.datasets_seq import SeqDataset
from src.data.datasets_diffusion import DiffusionDataset
from src.evaluation.micro_metrics import compute_dtw_per_sample, compute_frechet_per_sample


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        # allow direct state_dict saved via torch.save(model.state_dict())
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _infer_ckpt_model_type(state_dict: dict) -> str | None:
    keys = state_dict.keys()
    if any(str(k).startswith("nav_encoder.") for k in keys):
        return "physics"
    if any(str(k).startswith("unet.") for k in keys):
        return "diffusion"
    if any(str(k).startswith("encoder.") for k in keys) and any(str(k).startswith("decoder_cell.") for k in keys):
        return "baseline"
    return None


def _infer_hidden_dim(model_type: str, state_dict: dict) -> int | None:
    if model_type == "baseline":
        w = state_dict.get("head.weight")
        return int(w.shape[1]) if hasattr(w, "shape") and len(w.shape) == 2 else None
    if model_type == "diffusion":
        w = state_dict.get("unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    if model_type == "physics":
        w = state_dict.get("diffusion.unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    return None


def _integrate_positions(start_pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
    """
    Integrate step displacement into positions.

    Args:
        start_pos: (B, 2)
        vel: (B, F, 2)
    Returns:
        pos: (B, F, 2)
    """
    return start_pos[:, None, :] + np.cumsum(vel, axis=1)


def _accumulate_msd(pred_pos: np.ndarray, msd_sum: np.ndarray, msd_count: np.ndarray) -> None:
    """Accumulate MSD numerator/denominator for streaming average."""
    B, T, _ = pred_pos.shape
    for lag in range(1, T):
        diff = pred_pos[:, lag:] - pred_pos[:, :-lag]  # (B, T-lag, 2)
        sq = np.sum(diff * diff, axis=-1)  # (B, T-lag)
        msd_sum[lag - 1] += float(np.sum(sq))
        msd_count[lag - 1] += sq.size


def _accumulate_rog(pred_pos: np.ndarray) -> np.ndarray:
    """Return per-trajectory Rog: (B,)"""
    mean_pos = pred_pos.mean(axis=1, keepdims=True)
    diff = pred_pos - mean_pos
    sq = np.sum(diff * diff, axis=-1).mean(axis=1)
    return np.sqrt(sq)

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    # Evaluation usually on 'test' split via mode='r'? 
    # Or separate file. Assuming args.data_path points to test file.

    traj_ids = None
    if args.split != 'all':
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    if args.model_type == 'baseline':
        dataset = SeqDataset(args.data_path, obs_len=args.obs_len, pred_len=args.pred_len, traj_ids=traj_ids)
    else:
        # Physics or Diffusion
        nav_file = args.nav_file if args.model_type == 'physics' else None
        dataset = DiffusionDataset(
            args.data_path, 
            obs_len=args.obs_len, 
            pred_len=args.pred_len,
            nav_field_file=nav_file,
            nav_patch_size=args.patch_size,
            traj_ids=traj_ids,
        )
        
    # IMPORTANT: denormalization must use the same stats as the dataset.
    norm = dataset.normalizer
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 2. Load Model (auto-align hyperparams to checkpoint to avoid size mismatch)
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    state_dict = _load_state_dict(args.checkpoint, device=device)

    ckpt_type = _infer_ckpt_model_type(state_dict)
    if ckpt_type is not None and ckpt_type != args.model_type:
        print(f"[WARN] checkpoint 看起来是 {ckpt_type}，但你指定了 --model_type {args.model_type}，可能会加载失败。")

    ckpt_hidden_dim = _infer_hidden_dim(args.model_type, state_dict)
    if ckpt_hidden_dim is not None and int(args.hidden_dim) != int(ckpt_hidden_dim):
        print(f"[WARN] hidden_dim 不匹配：checkpoint={ckpt_hidden_dim}, args={args.hidden_dim}；已自动改为 checkpoint 值以匹配权重。")
        args.hidden_dim = int(ckpt_hidden_dim)
    
    if args.model_type == 'baseline':
        model = SeqBaseline(
            obs_dim=4, act_dim=2, cond_dim=6,
            hidden_dim=args.hidden_dim
        )
    elif args.model_type == 'diffusion':
        model = DiffusionTrajectoryModel(
            obs_dim=4, act_dim=2, cond_dim=6,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
    elif args.model_type == 'physics':
        model = PhysicsConditionDiffusion(
            obs_dim=4, act_dim=2, cond_dim=6,
            nav_patch_size=args.patch_size,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps
        )
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # 3. Inference Loop (streaming aggregation to avoid OOM)
    K = 1 if args.model_type == "baseline" else int(args.num_samples_per_condition)

    total_n = 0
    ade_mean_sum = 0.0
    ade_std_sum = 0.0
    ade_best_sum = 0.0
    fde_mean_sum = 0.0
    fde_std_sum = 0.0
    fde_best_sum = 0.0
    frechet_mean_sum = 0.0
    frechet_std_sum = 0.0
    frechet_best_sum = 0.0
    dtw_mean_sum = 0.0
    dtw_std_sum = 0.0
    dtw_best_sum = 0.0

    msd_sum = np.zeros((args.pred_len - 1,), dtype=np.float64)
    msd_count = np.zeros((args.pred_len - 1,), dtype=np.int64)
    rog_sum = 0.0
    rog_count = 0

    save_preds = []
    save_targets = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break

            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)

            nav_patch = batch['nav_patch'].to(device) if args.model_type == 'physics' else None

            start_pos_norm = obs[:, -1, :2]
            start_pos = norm.denormalize_pos(start_pos_norm.cpu().numpy())

            # GT future velocities
            if args.model_type == 'baseline':
                gt_vel_norm = batch['target_vel'].cpu().numpy()
            else:
                gt_vel_norm = batch['action'].cpu().numpy()
            gt_vel = norm.denormalize_vel(gt_vel_norm)

            gt_pos = _integrate_positions(start_pos, gt_vel)  # (B, F, 2)

            ade_list = []
            fde_list = []
            frechet_list = []
            dtw_list = []

            for k in range(K):
                if args.model_type == 'physics':
                    pred_vel_norm = model.sample_trajectory(obs, cond, args.pred_len, nav_patch=nav_patch)
                else:
                    pred_vel_norm = model.sample_trajectory(obs, cond, args.pred_len)

                pred_vel = norm.denormalize_vel(pred_vel_norm.cpu().numpy())
                pred_pos = _integrate_positions(start_pos, pred_vel)

                # micro errors per condition
                dist = np.linalg.norm(pred_pos - gt_pos, axis=-1)  # (B, F)
                ade = dist.mean(axis=1)  # (B,)
                fde = dist[:, -1]  # (B,)
                ade_list.append(ade.astype(np.float32))
                fde_list.append(fde.astype(np.float32))

                frechet = compute_frechet_per_sample(pred_pos, gt_pos)  # (B,)
                dtw = compute_dtw_per_sample(pred_pos, gt_pos)  # (B,)
                frechet_list.append(frechet.astype(np.float32))
                dtw_list.append(dtw.astype(np.float32))

                # macro accumulation over generated samples
                _accumulate_msd(pred_pos, msd_sum, msd_count)
                rog = _accumulate_rog(pred_pos)
                rog_sum += float(np.sum(rog))
                rog_count += int(rog.shape[0])

                # save a few examples (only k=0)
                if k == 0 and len(save_preds) < int(args.save_samples):
                    remaining = int(args.save_samples) - len(save_preds)
                    take = min(remaining, pred_pos.shape[0])
                    save_preds.extend(pred_pos[:take])
                    save_targets.extend(gt_pos[:take])

            ade_k = np.stack(ade_list, axis=0)  # (K, B)
            fde_k = np.stack(fde_list, axis=0)  # (K, B)
            frechet_k = np.stack(frechet_list, axis=0)  # (K, B)
            dtw_k = np.stack(dtw_list, axis=0)  # (K, B)

            ade_mean = ade_k.mean(axis=0)
            ade_std = ade_k.std(axis=0)
            ade_best = ade_k.min(axis=0)
            fde_mean = fde_k.mean(axis=0)
            fde_std = fde_k.std(axis=0)
            fde_best = fde_k.min(axis=0)
            frechet_mean = frechet_k.mean(axis=0)
            frechet_std = frechet_k.std(axis=0)
            frechet_best = frechet_k.min(axis=0)
            dtw_mean = dtw_k.mean(axis=0)
            dtw_std = dtw_k.std(axis=0)
            dtw_best = dtw_k.min(axis=0)

            B = int(ade_mean.shape[0])
            total_n += B
            ade_mean_sum += float(np.sum(ade_mean))
            ade_std_sum += float(np.sum(ade_std))
            ade_best_sum += float(np.sum(ade_best))
            fde_mean_sum += float(np.sum(fde_mean))
            fde_std_sum += float(np.sum(fde_std))
            fde_best_sum += float(np.sum(fde_best))
            frechet_mean_sum += float(np.sum(frechet_mean))
            frechet_std_sum += float(np.sum(frechet_std))
            frechet_best_sum += float(np.sum(frechet_best))
            dtw_mean_sum += float(np.sum(dtw_mean))
            dtw_std_sum += float(np.sum(dtw_std))
            dtw_best_sum += float(np.sum(dtw_best))

    if total_n == 0:
        raise RuntimeError("No samples were evaluated (empty dataset or too strict filtering).")

    msd_curve = (msd_sum / np.maximum(msd_count, 1)).astype(np.float64)

    results = {
        "split": args.split,
        "num_conditions": int(total_n),
        "K": int(K),
        "ADE_mean": ade_mean_sum / total_n,
        "ADE_std": ade_std_sum / total_n,
        "ADE_best": ade_best_sum / total_n,
        "FDE_mean": fde_mean_sum / total_n,
        "FDE_std": fde_std_sum / total_n,
        "FDE_best": fde_best_sum / total_n,
        "Frechet_mean": frechet_mean_sum / total_n,
        "Frechet_std": frechet_std_sum / total_n,
        "Frechet_best": frechet_best_sum / total_n,
        "DTW_mean": dtw_mean_sum / total_n,
        "DTW_std": dtw_std_sum / total_n,
        "DTW_best": dtw_best_sum / total_n,
        "MSD_1": float(msd_curve[0]) if msd_curve.size > 0 else 0.0,
        "MSD_5": float(msd_curve[4]) if msd_curve.size > 4 else 0.0,
        "MSD_10": float(msd_curve[9]) if msd_curve.size > 9 else 0.0,
        "msd_curve": msd_curve.tolist(),
        "Rog": (rog_sum / rog_count) if rog_count > 0 else 0.0,
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # 6. Save
    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    if save_preds:
        np.savez(out_dir / "samples.npz", preds=np.stack(save_preds, axis=0), targets=np.stack(save_targets, axis=0))
             
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['baseline', 'diffusion', 'physics'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--splits_dir', type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")
    
    # Physics args
    parser.add_argument('--nav_file', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    
    # Model args
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128) # check defaults
    parser.add_argument('--diff_steps', type=int, default=100)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples_per_condition', type=int, default=20, help="K for diffusion/physics (baseline uses 1)")
    parser.add_argument('--save_samples', type=int, default=100, help="number of (pred,target) pairs to save")
    parser.add_argument('--max_batches', type=int, default=None, help="limit evaluation batches for quick runs")
    
    args = parser.parse_args()
    evaluate(args)
