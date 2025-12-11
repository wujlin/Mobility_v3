import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

def set_style(context='paper', font_scale=1.5):
    """
    Configure Matplotlib/Seaborn for publication-quality figures.
    Context: 'paper', 'notebook', 'talk', 'poster'
    """
    # 1. Base Seaborn Style
    sns.set_context(context, font_scale=font_scale)
    sns.set_style("ticks") # Clean white background with ticks
    
    # 2. Custom RC Params for Academic Look
    rc_params = {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        
        # LaTeX rendering usually gives best math, but requires external deps. 
        # We stick to standard text for robustness.
        'text.usetex': False, 
        
        'axes.linewidth': 1.5,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        'legend.frameon': False,
        'legend.fontsize': 14,
        
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
    # Fallback logic
    if 'baseline' in method_name.lower(): return PALETTE['Baseline']
    if 'physics' in method_name.lower(): return PALETTE['Physics']
    if 'diffusion' in method_name.lower(): return PALETTE['Diffusion']
    if 'gt' in method_name.lower() or 'target' in method_name.lower(): return PALETTE['GT']
    return '#84919E' # Grey default
