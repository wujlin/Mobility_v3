import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.visualization.style_config import set_style, get_color

def power_law(t, a, alpha):
    return a * (t ** alpha)

def plot_msd_comparison(files_dict, output_dir):
    set_style(context='paper')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    t_lag = None
    
    for name, filepath in files_dict.items():
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'msd_curve' not in data:
            print(f"Warning: No MSD curve in {name}")
            continue
            
        msd = np.array(data['msd_curve'])
        if t_lag is None:
            t_lag = np.arange(1, len(msd) + 1)
        
        # Fit Alpha
        # Use simple log-linear regression for robust alpha est
        # log(MSD) = log(a) + alpha * log(t)
        valid = msd > 0
        p_coef = np.polyfit(np.log(t_lag[valid]), np.log(msd[valid]), 1)
        alpha = p_coef[0]
        
        # Plot Curve
        ax.loglog(t_lag, msd, label=f"{name} ($\\alpha={alpha:.2f}$)", 
                  color=get_color(name), linewidth=2.5, marker='o', markersize=5)
        
    ax.set_xlabel('Time Lag $\\tau$ (steps)')
    ax.set_ylabel('Mean Squared Displacement (MSD)')
    ax.set_title('Macroscopic Diffusion Properties')
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.4)
    
    out_path = Path(output_dir) / "viz_msd_comparison.png"
    plt.savefig(out_path, dpi=300)
    print(f"Saved MSD plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Accept pairs of Name=Path
    parser.add_argument('--metrics_files', nargs='+', required=True, 
                        help="List of Name=Path_to_metrics.json (e.g. Baseline=exp1/metrics.json)")
    parser.add_argument('--output_dir', type=str, default='.')
    
    args = parser.parse_args()
    
    files_dict = {}
    for item in args.metrics_files:
        key, val = item.split('=')
        files_dict[key] = val
        
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    plot_msd_comparison(files_dict, args.output_dir)
