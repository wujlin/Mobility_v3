import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def set_style(context='paper', font_scale=1.1):
    """
    Configure Matplotlib/Seaborn for publication-quality figures.
    Context: 'paper', 'notebook', 'talk', 'poster'
    """
    # 1. Base Seaborn Style
    sns.set_context(context, font_scale=font_scale)
    sns.set_style("ticks") # Clean white background with ticks
    
    # 2. Custom RC Params for Academic Look
    base_size = 11
    label_size = base_size * font_scale
    title_size = (base_size + 1) * font_scale
    tick_size = (base_size - 2) * font_scale
    legend_size = (base_size - 1) * font_scale

    rc_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        
        # LaTeX rendering usually gives best math, but requires external deps. 
        # We stick to standard text for robustness.
        'text.usetex': False, 
        
        'axes.linewidth': 1.2,
        'axes.labelsize': label_size,
        'axes.titlesize': title_size,
        
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.labelsize': tick_size,
        'ytick.labelsize': tick_size,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        'legend.frameon': False,
        'legend.fontsize': legend_size,
        
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
    }
    mpl.rcParams.update(rc_params)

# Color Palettes (Nature/Science inspired)
PALETTE = {
    'GT': '#333333',          # Black/Dark Grey for Ground Truth
    'Baseline': '#E64B35',    # Red (Nature Journal)
    'Diffusion': '#4DBBD5',   # Blue
    'Physics': '#00A087',     # Green
    'Obs': '#F39B7F'          # Orange/Peach for History
}

def get_color(method_name):
    # Fallback logic (heuristic name matching; keep KISS)
    name = (method_name or "").lower()

    # Ground truth
    if "gt" in name or "target" in name:
        return PALETTE["GT"]

    # Deterministic anchor / prior
    if "prior" in name or "anchor" in name:
        return PALETTE["Baseline"]

    # CFG settings (treat as physics-derived variants; use distinct shades)
    if "cfg2" in name:
        return "#00A087"  # same family as Physics
    if "cfg3" in name:
        return "#007A63"  # darker green
    if "cfg" in name:
        return PALETTE["Physics"]

    # Baselines / models
    if "baseline" in name:
        return PALETTE["Baseline"]
    if "physics" in name or "phys" in name:
        return PALETTE["Physics"]
    if "diffusion" in name or "diff" in name:
        return PALETTE["Diffusion"]

    return "#84919E"  # Grey default
