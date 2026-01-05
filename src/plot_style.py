"""
Plot style re-export for convenience.

Source of truth lives in `src/visualization/plot_style.py`.
This module exists so callers can simply `import src.plot_style` as documented.
"""

from __future__ import annotations

from src.visualization.plot_style import (  # noqa: F401
    FIGSIZE_FULL,
    FIGSIZE_HALF,
    OKABE_ITO,
    PaperStyle,
    add_panel_label,
    apply_paper_style,
    despine,
    paper_rcparams,
    paper_style,
    save_figure,
)

__all__ = [
    "OKABE_ITO",
    "PaperStyle",
    "FIGSIZE_FULL",
    "FIGSIZE_HALF",
    "paper_rcparams",
    "paper_style",
    "apply_paper_style",
    "save_figure",
    "despine",
    "add_panel_label",
]

