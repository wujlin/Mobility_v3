import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.style_config import set_style, PALETTE, get_color

def plot_trajectories(args):
    set_style(context='paper')
    
    # 1. Load Data
    data = np.load(args.samples_file)
    preds = data['preds']   # (N, F, 2)
    targets = data['targets'] # (N, F, 2)
    
    # Select subset
    indices = np.random.choice(len(preds), min(len(preds), args.num_plots), replace=False)
    
    # 2. Setup Figure (Grid: cols x rows)
    cols = 3
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Plot GT
        gt_traj = targets[idx]
        ax.plot(gt_traj[:, 1], gt_traj[:, 0], color=PALETTE['GT'], 
                linewidth=2, label='Ground Truth', marker='o', markersize=4, alpha=0.8)
        
        # Plot Pred
        pred_traj = preds[idx]
        ax.plot(pred_traj[:, 1], pred_traj[:, 0], color=get_color(args.method_name), 
                linewidth=2, label=args.method_name, linestyle='--', marker='x', markersize=4)
        
        # Start point
        ax.scatter(gt_traj[0, 1], gt_traj[0, 0], color='black', s=50, marker='*', zorder=10)
        
        ax.set_aspect('equal')
        ax.set_title(f'Sample #{idx}')
        if i == 0:
            ax.legend()
            
    # Hide empty axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
        
    out_path = Path(args.output_dir) / f"viz_traj_{args.method_name}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved trajectory plot to {out_path}")

def plot_spatial_heatmap(args):
    """
    Plot Kernel Density of all points to show Mesoscopic distribution.
    """
    set_style(context='paper')
    data = np.load(args.samples_file)
    preds = data['preds'] # (N, F, 2)
    targets = data['targets']
    
    # Flatten: (N*F, 2)
    pred_points = preds.reshape(-1, 2)
    gt_points = targets.reshape(-1, 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    
    # 1. Ground Truth Heatmap
    axes[0].hist2d(gt_points[:, 1], gt_points[:, 0], bins=100, cmap='Greys', density=True)
    axes[0].set_title("Ground Truth Distribution")
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis() # Usually Y is Lat, index 0. Invert for map view? 
    # Our simple projection might have Y increasing.
    
    # 2. Predicted Heatmap
    axes[1].hist2d(pred_points[:, 1], pred_points[:, 0], bins=100, cmap='Blues', density=True)
    axes[1].set_title(f"Predicted Distribution ({args.method_name})")
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    
    out_path = Path(args.output_dir) / f"viz_heatmap_{args.method_name}.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved heatmap to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_file', type=str, required=True, help="Path to samples.npz")
    parser.add_argument('--method_name', type=str, default='Model')
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--num_plots', type=int, default=6)
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_trajectories(args)
    plot_spatial_heatmap(args)
