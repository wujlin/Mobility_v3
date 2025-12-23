import torch
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import json
import numpy as np
import random
from typing import Optional, Tuple
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, *args, **kwargs):  # type: ignore[no-redef]
        return x

from src.models.seq.seq_baseline import SeqBaseline
from src.models.seq.seq_cvae import SeqCVAE
from src.models.diffusion.diffusion_model import DiffusionTrajectoryModel
from src.models.physics.physics_condition_diffusion import PhysicsConditionDiffusion
from src.models.flow.rectified_flow_model import RectifiedFlowTrajectoryModel
from src.models.physics.physics_condition_flow import PhysicsConditionFlow
from src.data.datasets_seq import SeqDataset
from src.data.datasets_diffusion import DiffusionDataset
from src.evaluation.micro_metrics import compute_dtw_per_sample, compute_frechet_per_sample


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[dict, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        return ckpt["model_state_dict"], (cfg if isinstance(cfg, dict) else {})
    if isinstance(ckpt, dict):
        # allow direct state_dict saved via torch.save(model.state_dict())
        return ckpt, {}
    raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")


def _load_state_dict(checkpoint_path: str, device: torch.device) -> dict:
    state_dict, _ = _load_checkpoint(checkpoint_path, device=device)
    return state_dict

def _load_baseline_prior(prior_checkpoint: str, device: torch.device) -> SeqBaseline:
    """
    Load a frozen SeqBaseline as a deterministic prior for residual diffusion evaluation.
    """
    ckpt = torch.load(prior_checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        cfg = ckpt.get("config", {})
        hidden_dim = cfg.get("hidden_dim") if isinstance(cfg, dict) else None
    elif isinstance(ckpt, dict):
        state_dict = ckpt
        hidden_dim = None
    else:
        raise TypeError(f"Unsupported prior checkpoint format: {type(ckpt)}")

    if hidden_dim is None:
        w = state_dict.get("head.weight")
        if hasattr(w, "shape") and len(w.shape) == 2:
            hidden_dim = int(w.shape[1])

    if hidden_dim is None:
        raise ValueError(f"Cannot infer prior hidden_dim from checkpoint: {prior_checkpoint}")

    model = SeqBaseline(obs_dim=4, act_dim=2, cond_dim=6, hidden_dim=int(hidden_dim))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _infer_ckpt_model_type(state_dict: dict) -> Optional[str]:
    keys = state_dict.keys()
    if any(str(k).startswith("obs_encoder.") for k in keys) and any(str(k).startswith("prior_mu.") for k in keys):
        return "cvae"
    has_flow = any(str(k).endswith("rf_time_scale") for k in keys)
    has_nav = any(str(k).startswith("nav_encoder.") for k in keys)
    if has_flow:
        return "physics_flow" if has_nav else "flow"
    if any(str(k).startswith("nav_encoder.") for k in keys):
        return "physics"
    if any(str(k).startswith("unet.") for k in keys):
        return "diffusion"
    if any(str(k).startswith("encoder.") for k in keys) and any(str(k).startswith("decoder_cell.") for k in keys):
        return "baseline"
    return None


def _infer_hidden_dim(model_type: str, state_dict: dict) -> Optional[int]:
    if model_type == "baseline":
        w = state_dict.get("head.weight")
        return int(w.shape[1]) if hasattr(w, "shape") and len(w.shape) == 2 else None
    if model_type == "cvae":
        w = state_dict.get("decoder_cell.weight_hh")
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[1])
        w = state_dict.get("obs_encoder.weight_ih_l0")
        if hasattr(w, "shape") and len(w.shape) == 2:
            return int(w.shape[0]) // 4
        return None
    if model_type == "diffusion":
        w = state_dict.get("unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    if model_type == "physics":
        w = state_dict.get("diffusion.unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    if model_type == "flow":
        w = state_dict.get("unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    if model_type == "physics_flow":
        w = state_dict.get("flow.unet.init_conv.weight")
        return int(w.shape[0]) if hasattr(w, "shape") and len(w.shape) == 3 else None
    return None


def _infer_latent_dim(model_type: str, state_dict: dict) -> Optional[int]:
    if model_type != "cvae":
        return None
    w = state_dict.get("prior_mu.weight")
    if hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    b = state_dict.get("prior_mu.bias")
    if hasattr(b, "shape") and len(b.shape) == 1:
        return int(b.shape[0])
    return None


def _infer_nav_gate_hidden(state_dict: dict) -> Optional[int]:
    """
    Detect optional Physics nav_gate (learnable gating) from checkpoint keys.

    Returns:
        hidden_dim if nav_gate exists, else None.
    """
    w = state_dict.get("nav_gate.0.weight")
    if hasattr(w, "shape") and len(w.shape) == 2:
        # Linear(in_dim, hidden) -> weight shape (hidden, in_dim)
        return int(w.shape[0])
    return None


def _infer_nav_emb_dim(model_type: str, state_dict: dict, obs_len: int, obs_dim: int = 4, base_cond_dim: int = 6) -> Optional[int]:
    """
    Infer nav_emb_dim from checkpoint by inspecting cond_encoder input dim.

    For PhysicsConditionDiffusion:
        key = diffusion.cond_encoder.0.weight  # shape (hidden*2, obs_len*obs_dim + cond_dim+nav_emb_dim)
    For PhysicsConditionFlow:
        key = flow.cond_encoder.0.weight
    """
    key = None
    if model_type == "physics":
        key = "diffusion.cond_encoder.0.weight"
    if model_type == "physics_flow":
        key = "flow.cond_encoder.0.weight"
    if key is None:
        return None
    w = state_dict.get(key)
    if not (hasattr(w, "shape") and len(w.shape) == 2):
        return None
    in_features = int(w.shape[1])
    hist = int(obs_len) * int(obs_dim)
    cond_total = in_features - hist
    nav_dim = int(cond_total) - int(base_cond_dim)
    return nav_dim if nav_dim > 0 else None


def _resolve_pred_type(arg_pred_type: str, ckpt_cfg: dict) -> str:
    """
    Resolve diffusion parameterization type for evaluation.

    Args:
        arg_pred_type: one of {"auto", "eps", "v"}
        ckpt_cfg: checkpoint config dict if exists
    Returns:
        resolved: one of {"eps", "v"}
    """
    arg_pred_type = str(arg_pred_type)
    if arg_pred_type not in ("auto", "eps", "v"):
        raise ValueError(f"Invalid --pred_type: {arg_pred_type} (expected: auto|eps|v)")

    ckpt_pred: Optional[str] = None
    if isinstance(ckpt_cfg, dict):
        for k in ("pred_type", "prediction_type"):
            v = ckpt_cfg.get(k)
            if str(v) in ("eps", "v"):
                ckpt_pred = str(v)
                break

    if arg_pred_type == "auto":
        return ckpt_pred or "eps"

    if ckpt_pred is not None and ckpt_pred != arg_pred_type:
        print(f"[WARN] pred_type 不匹配：ckpt={ckpt_pred}, args={arg_pred_type}；已自动改为 ckpt 值以避免错误评估。")
        return ckpt_pred

    return arg_pred_type


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


def _speed_sum_and_count_from_vel(vel: np.ndarray) -> Tuple[float, int]:
    """
    Compute summed step speed and count.

    Args:
        vel: (B, F, 2)
    Returns:
        (sum_speed, count)
    """
    speed = np.linalg.norm(vel, axis=-1)  # (B, F)
    return float(np.sum(speed)), int(speed.size)


def _path_len_sum_and_count_from_vel(vel: np.ndarray) -> Tuple[float, int]:
    """
    Compute summed path length (sum of step speeds) and trajectory count.

    Args:
        vel: (B, F, 2)
    Returns:
        (sum_path_len, count_traj)
    """
    speed = np.linalg.norm(vel, axis=-1)  # (B, F)
    path_len = np.sum(speed, axis=1)  # (B,)
    return float(np.sum(path_len)), int(path_len.shape[0])


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
    _set_seed(int(args.seed))
    print(f"Using seed: {int(args.seed)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.prior_checkpoint and args.model_type in ("baseline", "cvae"):
        raise ValueError("--prior_checkpoint 仅用于 diffusion/physics/flow 的 residual 评估（baseline/cvae 不需要 prior）")
    
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
    elif args.model_type == 'cvae':
        dataset = SeqDataset(args.data_path, obs_len=args.obs_len, pred_len=args.pred_len, traj_ids=traj_ids)
    else:
        # Diffusion-like dataset (diffusion / physics / flow / physics_flow)
        if args.model_type in ("physics", "physics_flow") and not args.nav_file:
            raise ValueError("--nav_file is required for --model_type physics/physics_flow")
        nav_file = args.nav_file if args.model_type in ('physics', 'physics_flow') else None
        dataset = DiffusionDataset(
            args.data_path, 
            obs_len=args.obs_len, 
            pred_len=args.pred_len,
            nav_field_file=nav_file,
            nav_patch_size=args.patch_size,
            nav_patch_channel2=args.nav_patch_channel2,
            traj_ids=traj_ids,
        )
        
    # IMPORTANT: denormalization must use the same stats as the dataset.
    norm = dataset.normalizer
    try:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=int(args.num_workers))
    except PermissionError:
        print("[WARN] DataLoader 多进程初始化失败，已自动降级为 num_workers=0（单进程加载）。")
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 2. Load Model (auto-align hyperparams to checkpoint to avoid size mismatch)
    print(f"Loading {args.model_type} model from {args.checkpoint}...")
    state_dict, ckpt_cfg = _load_checkpoint(args.checkpoint, device=device)

    # Align diffusion parameterization (eps vs v) to checkpoint to avoid semantic mismatch.
    if args.model_type in ("diffusion", "physics"):
        args.pred_type = _resolve_pred_type(getattr(args, "pred_type", "auto"), ckpt_cfg)
        print(f"[OK] pred_type={args.pred_type}")
        if float(getattr(args, "cfg_scale", 0.0)) != 0.0:
            cfg_drop = ckpt_cfg.get("cfg_drop_dest_prob") if isinstance(ckpt_cfg, dict) else None
            try:
                cfg_drop_f = float(cfg_drop) if cfg_drop is not None else 0.0
            except Exception:
                cfg_drop_f = 0.0
            if cfg_drop_f <= 0.0:
                print("[WARN] 你设置了 --cfg_scale，但 checkpoint 未启用 cfg_drop_dest_prob（CFG 训练 dropout）；CFG 可能无效或不稳定。")

    ckpt_type = _infer_ckpt_model_type(state_dict)
    if ckpt_type is not None and ckpt_type != args.model_type:
        print(f"[WARN] checkpoint 看起来是 {ckpt_type}，但你指定了 --model_type {args.model_type}，可能会加载失败。")

    ckpt_hidden_dim = _infer_hidden_dim(args.model_type, state_dict)
    if ckpt_hidden_dim is not None and int(args.hidden_dim) != int(ckpt_hidden_dim):
        print(f"[WARN] hidden_dim 不匹配：checkpoint={ckpt_hidden_dim}, args={args.hidden_dim}；已自动改为 checkpoint 值以匹配权重。")
        args.hidden_dim = int(ckpt_hidden_dim)

    ckpt_latent_dim = _infer_latent_dim(args.model_type, state_dict)
    if ckpt_latent_dim is not None and int(args.latent_dim) != int(ckpt_latent_dim):
        print(f"[WARN] latent_dim 不匹配：checkpoint={ckpt_latent_dim}, args={args.latent_dim}；已自动改为 checkpoint 值以匹配权重。")
        args.latent_dim = int(ckpt_latent_dim)

    # Physics-only: auto-align optional nav_gate to checkpoint to avoid load mismatch.
    nav_emb_dim = 32
    if args.model_type in ("physics", "physics_flow"):
        ckpt_nav_gate_hidden = _infer_nav_gate_hidden(state_dict)
        ckpt_has_gate = ckpt_nav_gate_hidden is not None
        user_nav_gate = str(getattr(args, "nav_gate", "auto"))
        if user_nav_gate == "auto":
            args.nav_gate = "obscond" if ckpt_has_gate else "none"
        else:
            if ckpt_has_gate and user_nav_gate == "none":
                print("[WARN] checkpoint 含 nav_gate，但你指定了 --nav_gate none；已自动改为 obscond 以匹配权重。")
                args.nav_gate = "obscond"
            elif (not ckpt_has_gate) and user_nav_gate != "none":
                print("[WARN] checkpoint 不含 nav_gate，但你指定了 --nav_gate!=none；已自动改为 none 以匹配权重。")
                args.nav_gate = "none"
            else:
                args.nav_gate = user_nav_gate
        if ckpt_has_gate:
            args.nav_gate_hidden = int(ckpt_nav_gate_hidden)
        ckpt_nav_emb_dim = _infer_nav_emb_dim(str(args.model_type), state_dict, int(args.obs_len))
        if ckpt_nav_emb_dim is not None:
            nav_emb_dim = int(ckpt_nav_emb_dim)
    
    if args.model_type == 'baseline':
        model = SeqBaseline(
            obs_dim=4, act_dim=2, cond_dim=6,
            hidden_dim=args.hidden_dim
        )
    elif args.model_type == 'cvae':
        model = SeqCVAE(
            obs_dim=4, act_dim=2, cond_dim=6,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
        )
    elif args.model_type == 'diffusion':
        model = DiffusionTrajectoryModel(
            obs_dim=4, act_dim=2, cond_dim=6,
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps,
            prediction_type=str(getattr(args, "pred_type", "eps")),
        )
    elif args.model_type == 'physics':
        model = PhysicsConditionDiffusion(
            obs_dim=4, act_dim=2, cond_dim=6,
            nav_patch_size=args.patch_size,
            nav_emb_dim=int(nav_emb_dim),
            obs_len=args.obs_len, pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            diffusion_steps=args.diff_steps,
            prediction_type=str(getattr(args, "pred_type", "eps")),
            nav_emb_scale=float(args.nav_emb_scale),
            nav_gate=str(args.nav_gate),
            nav_gate_hidden=int(getattr(args, "nav_gate_hidden", 32)),
            nav_gate_dropout=float(getattr(args, "nav_gate_dropout", 0.0)),
        )
    elif args.model_type == "flow":
        flow_steps = getattr(args, "flow_steps", None)
        solver_steps = int(flow_steps) if flow_steps is not None else int(args.diff_steps)
        model = RectifiedFlowTrajectoryModel(
            obs_dim=4,
            act_dim=2,
            cond_dim=6,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            solver_steps=int(solver_steps),
        )
    elif args.model_type == "physics_flow":
        flow_steps = getattr(args, "flow_steps", None)
        solver_steps = int(flow_steps) if flow_steps is not None else int(args.diff_steps)
        model = PhysicsConditionFlow(
            obs_dim=4,
            act_dim=2,
            cond_dim=6,
            nav_patch_size=args.patch_size,
            nav_emb_dim=int(nav_emb_dim),
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            hidden_dim=args.hidden_dim,
            solver_steps=int(solver_steps),
            nav_emb_scale=float(args.nav_emb_scale),
            nav_gate=str(args.nav_gate),
            nav_gate_hidden=int(getattr(args, "nav_gate_hidden", 32)),
            nav_gate_dropout=float(getattr(args, "nav_gate_dropout", 0.0)),
        )
        
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    prior_model = None
    if args.prior_checkpoint:
        prior_model = _load_baseline_prior(str(args.prior_checkpoint), device=device)
        print(f"[OK] Residual eval enabled: prior={args.prior_checkpoint}")
    
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

    pred_speed_sum = 0.0
    pred_speed_count = 0
    gt_speed_sum = 0.0
    gt_speed_count = 0
    pred_path_len_sum = 0.0
    pred_path_len_count = 0
    gt_path_len_sum = 0.0
    gt_path_len_count = 0

    gt_msd_sum = np.zeros((args.pred_len - 1,), dtype=np.float64)
    gt_msd_count = np.zeros((args.pred_len - 1,), dtype=np.int64)
    gt_rog_sum = 0.0
    gt_rog_count = 0

    save_preds = []
    save_preds_k_by_k = None  # optional: list[list[np.ndarray]] for saving all K samples
    save_targets = []
    save_start_pos = []
    
    print("Running Inference...")
    with torch.no_grad():
        tqdm_total = len(dataloader)
        if args.max_batches is not None:
            tqdm_total = min(int(tqdm_total), int(args.max_batches))
        for batch_idx, batch in enumerate(tqdm(dataloader, total=tqdm_total)):
            if args.max_batches is not None and batch_idx >= int(args.max_batches):
                break

            obs = batch['obs'].to(device)
            cond = batch['cond'].to(device)

            nav_patch = batch['nav_patch'].to(device) if args.model_type in ('physics', 'physics_flow') else None

            # CFG inference: build unconditional condition by dropping destination (d_y, d_x)
            cond_uncond = None
            cfg_scale = float(getattr(args, "cfg_scale", 0.0))
            if cfg_scale != 0.0 and args.model_type in ("diffusion", "physics", "flow", "physics_flow"):
                cond_uncond = cond.clone()
                mode = str(getattr(args, "cfg_uncond_dest_mode", "origin"))
                if mode == "origin":
                    cond_uncond[:, 4:6] = cond_uncond[:, 2:4]
                elif mode == "zeros":
                    cond_uncond[:, 4:6].zero_()
                else:  # pragma: no cover
                    raise ValueError(f"Unknown --cfg_uncond_dest_mode: {mode}")

            start_pos_norm = obs[:, -1, :2]
            start_pos = norm.denormalize_pos(start_pos_norm.cpu().numpy())

            # Deterministic prior (normalized vel)
            prior_vel_norm = None
            if prior_model is not None:
                prior_vel_norm = prior_model.sample_trajectory(obs, cond, int(args.pred_len))

            # GT future velocities
            if args.model_type in ('baseline', 'cvae'):
                gt_vel_norm = batch['target_vel'].cpu().numpy()
            else:
                gt_vel_norm = batch['action'].cpu().numpy()
            gt_vel = norm.denormalize_vel(gt_vel_norm)

            # GT step stats (per condition, single trajectory)
            s_sum, s_cnt = _speed_sum_and_count_from_vel(gt_vel)
            gt_speed_sum += s_sum
            gt_speed_count += s_cnt
            pl_sum, pl_cnt = _path_len_sum_and_count_from_vel(gt_vel)
            gt_path_len_sum += pl_sum
            gt_path_len_count += pl_cnt

            gt_pos = _integrate_positions(start_pos, gt_vel)  # (B, F, 2)

            # GT macro metrics (对照用；每个 condition 只有一条 GT)
            _accumulate_msd(gt_pos, gt_msd_sum, gt_msd_count)
            gt_rog = _accumulate_rog(gt_pos)
            gt_rog_sum += float(np.sum(gt_rog))
            gt_rog_count += int(gt_rog.shape[0])

            ade_list = []
            fde_list = []
            frechet_list = []
            dtw_list = []

            # Saving policy:
            # - Always keep backward-compatible `preds` as k=0.
            # - Optionally save all K as `preds_k` (N, K, F, 2) when --save_all_k is enabled.
            want_save = int(args.save_samples) > 0 and len(save_targets) < int(args.save_samples)
            take = 0
            if want_save:
                remaining = int(args.save_samples) - len(save_targets)
                take = min(remaining, int(gt_pos.shape[0]))
                if bool(getattr(args, "save_all_k", False)) and save_preds_k_by_k is None:
                    save_preds_k_by_k = [[] for _ in range(int(K))]

            for k in range(K):
                if args.model_type in ('physics', 'physics_flow'):
                    pred_vel_norm = model.sample_trajectory(
                        obs,
                        cond,
                        args.pred_len,
                        nav_patch=nav_patch,
                        cond_uncond=cond_uncond,
                        cfg_scale=cfg_scale,
                    )
                elif args.model_type == 'cvae':
                    pred_vel_norm = model.sample_trajectory(obs, cond, args.pred_len, z_temperature=float(args.z_temperature))
                else:
                    pred_vel_norm = model.sample_trajectory(
                        obs,
                        cond,
                        args.pred_len,
                        cond_uncond=cond_uncond,
                        cfg_scale=cfg_scale,
                    )

                # Residual mode: model predicts residual vel; add deterministic prior.
                if prior_vel_norm is not None and args.model_type in ("diffusion", "physics", "flow", "physics_flow"):
                    pred_vel_norm = pred_vel_norm + prior_vel_norm

                pred_vel = norm.denormalize_vel(pred_vel_norm.cpu().numpy())
                pred_vel = pred_vel * float(args.vel_scale)

                s_sum, s_cnt = _speed_sum_and_count_from_vel(pred_vel)
                pred_speed_sum += s_sum
                pred_speed_count += s_cnt
                pl_sum, pl_cnt = _path_len_sum_and_count_from_vel(pred_vel)
                pred_path_len_sum += pl_sum
                pred_path_len_count += pl_cnt
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

                # Save examples (k=0 for backward compatibility; optionally all K).
                if take > 0:
                    if k == 0 and len(save_targets) < int(args.save_samples):
                        save_preds.extend(pred_pos[:take].astype(np.float32, copy=False))
                        save_targets.extend(gt_pos[:take].astype(np.float32, copy=False))
                        save_start_pos.extend(start_pos[:take].astype(np.float32, copy=False))
                    if bool(getattr(args, "save_all_k", False)) and save_preds_k_by_k is not None:
                        save_preds_k_by_k[int(k)].append(pred_pos[:take].astype(np.float32, copy=False))

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
    gt_msd_curve = (gt_msd_sum / np.maximum(gt_msd_count, 1)).astype(np.float64)

    results = {
        "seed": int(args.seed),
        "split": args.split,
        "num_conditions": int(total_n),
        "K": int(K),
        "pred_type": (str(getattr(args, "pred_type", "eps")) if args.model_type in ("diffusion", "physics") else None),
        "cfg_scale": (float(getattr(args, "cfg_scale", 0.0)) if args.model_type in ("diffusion", "physics") else None),
        "cfg_uncond_dest_mode": (str(getattr(args, "cfg_uncond_dest_mode", "origin")) if args.model_type in ("diffusion", "physics") else None),
        "vel_scale": float(args.vel_scale),
        "prior_checkpoint": (str(args.prior_checkpoint) if args.prior_checkpoint else None),
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
        "pred_speed_mean": (pred_speed_sum / max(pred_speed_count, 1)),
        "gt_speed_mean": (gt_speed_sum / max(gt_speed_count, 1)),
        "pred_path_len_mean": (pred_path_len_sum / max(pred_path_len_count, 1)),
        "gt_path_len_mean": (gt_path_len_sum / max(gt_path_len_count, 1)),

        # Ground-truth macro metrics (paper-ready 对照)
        "GT_MSD_1": float(gt_msd_curve[0]) if gt_msd_curve.size > 0 else 0.0,
        "GT_MSD_5": float(gt_msd_curve[4]) if gt_msd_curve.size > 4 else 0.0,
        "GT_MSD_10": float(gt_msd_curve[9]) if gt_msd_curve.size > 9 else 0.0,
        "GT_msd_curve": gt_msd_curve.tolist(),
        "GT_Rog": (gt_rog_sum / gt_rog_count) if gt_rog_count > 0 else 0.0,
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    # 6. Save
    out_dir = Path(f"data/experiments/{args.exp_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    if save_preds:
        npz_kwargs = {
            "preds": np.stack(save_preds, axis=0),
            "targets": np.stack(save_targets, axis=0),
            "start_pos": np.stack(save_start_pos, axis=0),
        }
        if bool(getattr(args, "save_all_k", False)) and save_preds_k_by_k is not None:
            per_k = []
            for k_list in save_preds_k_by_k:
                if not k_list:
                    raise RuntimeError(
                        "save_all_k enabled but some k lists are empty; this indicates a bug in saving logic."
                    )
                per_k.append(np.concatenate(k_list, axis=0))
            preds_k = np.stack(per_k, axis=1)  # (N, K, F, 2)
            if preds_k.shape[0] != npz_kwargs["preds"].shape[0]:
                raise RuntimeError(
                    f"save_all_k enabled but N mismatch: preds_k N={preds_k.shape[0]} vs preds N={npz_kwargs['preds'].shape[0]}"
                )
            npz_kwargs["preds_k"] = preds_k.astype(np.float32, copy=False)
        np.savez(out_dir / "samples.npz", **npz_kwargs)
             
    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--model_type', type=str, choices=['baseline', 'cvae', 'diffusion', 'physics', 'flow', 'physics_flow'], required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--prior_checkpoint', type=str, default=None, help="Residual diffusion: frozen deterministic prior checkpoint (SeqBaseline last.pt)")
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--splits_dir', type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")
    
    # Physics args
    parser.add_argument('--nav_file', type=str, default=None)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument(
        '--nav_patch_channel2',
        type=str,
        choices=['speed', 'count', 'zeros'],
        default='speed',
        help="nav_patch 第3通道：speed(默认)/count/log1p(count)/zeros(置零，仅方向)",
    )
    parser.add_argument('--nav_emb_scale', type=float, default=1.0, help="Physics: nav embedding 强度缩放（<1 减弱 mean-field tether）")
    parser.add_argument('--nav_gate', type=str, choices=['auto', 'none', 'obscond'], default='auto', help="Physics: nav_gate 模式（auto: 从 checkpoint 推断；obscond: learnable gate；none: 关闭）")
    parser.add_argument('--nav_gate_hidden', type=int, default=32, help="Physics: nav gate MLP hidden dim（通常无需手动设置，auto 会从 checkpoint 对齐）")
    parser.add_argument('--nav_gate_dropout', type=float, default=0.0, help="Physics: dropout on gate scalar (训练时用；评估无影响)")
    
    # Model args
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=128) # check defaults
    parser.add_argument('--pred_type', type=str, choices=['auto', 'eps', 'v'], default='auto', help="Diffusion 参数化：auto(默认，从 checkpoint 对齐)/eps/v")
    parser.add_argument('--diff_steps', type=int, default=100)
    parser.add_argument('--flow_steps', type=int, default=None, help="Rectified Flow: ODE solver steps (Euler). 默认沿用 --diff_steps。")
    parser.add_argument('--latent_dim', type=int, default=16, help="CVAE latent dim (only for --model_type cvae)")
    parser.add_argument('--z_temperature', type=float, default=1.0, help="CVAE 采样温度（仅 cvae 生效）")

    parser.add_argument('--seed', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4, help="DataLoader workers; 0 for single-process (WSL/权限受限环境建议 0)")
    parser.add_argument('--num_samples_per_condition', type=int, default=20, help="K for diffusion/physics (baseline uses 1)")
    parser.add_argument('--save_samples', type=int, default=100, help="number of (pred,target) pairs to save")
    parser.add_argument('--save_all_k', action='store_true', help="when saving samples, also save all K predictions as preds_k (N,K,F,2)")
    parser.add_argument('--max_batches', type=int, default=None, help="limit evaluation batches for quick runs")
    parser.add_argument('--vel_scale', type=float, default=1.0, help="对预测 future vel 做整体缩放（用于修正运动幅度偏小；与温度/噪声解耦）")
    parser.add_argument('--cfg_scale', type=float, default=0.0, help="CFG guidance scale（0 关闭；>0 放大 destination 条件影响）")
    parser.add_argument('--cfg_uncond_dest_mode', type=str, choices=['origin', 'zeros'], default='origin', help="CFG uncond 分支 destination 替换方式")
    
    args = parser.parse_args()
    evaluate(args)
