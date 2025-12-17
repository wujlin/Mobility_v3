import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data.datasets_seq import SeqDataset
from src.models.seq.seq_cvae import SeqCVAE


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    _set_seed(int(args.seed))
    print(f"Using seed: {int(args.seed)}")

    print("Loading datasets...")
    traj_ids = None
    if args.split != "all":
        processed_dir = Path(args.data_path).resolve().parents[1]
        splits_dir = Path(args.splits_dir) if args.splits_dir else (processed_dir / "splits")
        split_file = splits_dir / f"{args.split}_ids.npy"
        if not split_file.exists():
            raise FileNotFoundError(split_file)
        traj_ids = np.load(split_file).astype(np.int64)
        print(f"Using split={args.split}: {len(traj_ids)} trajectories ({split_file})")

    train_dataset = SeqDataset(args.data_path, obs_len=args.obs_len, pred_len=args.pred_len, traj_ids=traj_ids)
    g = torch.Generator()
    g.manual_seed(int(args.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        generator=g,
    )

    model = SeqCVAE(
        obs_dim=4,
        act_dim=2,
        cond_dim=6,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(args.lr))

    save_dir = Path(f"data/experiments/{args.exp_name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "model_type": "cvae",
        "data_path": args.data_path,
        "split": args.split,
        "splits_dir": args.splits_dir,
        "obs_len": int(args.obs_len),
        "pred_len": int(args.pred_len),
        "hidden_dim": int(args.hidden_dim),
        "latent_dim": int(args.latent_dim),
        "beta_kl": float(args.beta_kl),
        "kl_anneal_epochs": int(args.kl_anneal_epochs),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "num_workers": int(args.num_workers),
    }

    print(f"Start training {args.exp_name}...")
    model.train()

    for epoch in range(int(args.epochs)):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        start_time = time.time()

        if int(args.kl_anneal_epochs) > 0:
            frac = min(1.0, float(epoch + 1) / float(args.kl_anneal_epochs))
            kl_weight = float(args.beta_kl) * frac
        else:
            kl_weight = float(args.beta_kl)

        for batch_idx, batch in enumerate(train_loader):
            obs = batch["obs"].to(device)
            cond = batch["cond"].to(device)
            target_vel = batch["target_vel"].to(device)

            optimizer.zero_grad()

            loss, recon, kl = model(obs, cond, target=target_vel, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Loss {loss.item():.4f} | Recon {recon.item():.4f} | KL {kl.item():.4f} | kl_w {kl_weight:.4f}"
                )

        n_batches = max(1, len(train_loader))
        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        duration = time.time() - start_time
        print(
            f"Epoch {epoch} Done. Avg Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}. Time: {duration:.1f}s"
        )

        torch.save(
            {
                "epoch": int(epoch),
                "loss": float(avg_loss),
                "recon": float(avg_recon),
                "kl": float(avg_kl),
                "kl_weight": float(kl_weight),
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": run_config,
            },
            save_dir / "last.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="cvae_v1")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="train")
    parser.add_argument("--splits_dir", type=str, default=None, help="override splits dir (default: <processed_dir>/splits)")

    parser.add_argument("--obs_len", type=int, default=8)
    parser.add_argument("--pred_len", type=int, default=12)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--beta_kl", type=float, default=0.1, help="KL 权重（可配合 kl_anneal_epochs）")
    parser.add_argument("--kl_anneal_epochs", type=int, default=0, help="KL 线性 warmup epoch 数；0 表示不 warmup")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)

