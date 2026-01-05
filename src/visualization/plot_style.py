"""
论文级可视化统一样式（偏“子刊可投”的简洁可打印风格）。

目标：
- 统一字体（Times New Roman / STIXGeneral 兜底）、字号、线宽、配色
- 统一导出参数，降低 LaTeX 合图后的不一致感
- 提供面板标签（a/b/c/d）工具，避免 LaTeX 子图标签跑位

约定：
- 不依赖 seaborn，保持最小依赖与可控性
- rcParams 全局设置 + 可选的 rc_context
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import os
from pathlib import Path
from typing import Dict, Iterator, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm


# Okabe–Ito 色盲友好配色（推荐用于期刊图）
OKABE_ITO: Dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#777777",
}


@dataclass(frozen=True)
class PaperStyle:
    # 说明：这些字号是"以最终 LaTeX 嵌入后的可读性"为目标校准的。
    # 典型用法：
    # - 半栏子图：在 LaTeX 中以 ~0.48\\linewidth 插入（约 3.1in 宽）
    # - 全栏子图：在 LaTeX 中以 ~\\linewidth 插入（约 6.5in 宽）
    # 因此脚本中 figure 物理尺寸（figsize）应与嵌入宽度同量级，避免缩放导致字号过小。
    # 
    # 2024-12 调整：减小字号，避免半栏图中 title/label/tick 拥挤与重叠
    font_size: float = 11.0          # 基础字号：12 → 11
    axes_labelsize: float = 12.0     # 轴标签：14 → 12
    axes_titlesize: float = 11.0     # 标题字号：14 → 11（新增独立控制）
    tick_labelsize: float = 10.0     # tick 标签：11 → 10
    # 子刊常见：legend 字号约为正文的 80–90%
    legend_fontsize: float = 9.0     # 保持不变
    axes_linewidth: float = 1.2
    lines_linewidth: float = 2.4
    lines_markersize: float = 5.5
    figure_dpi: int = 150
    savefig_dpi: int = 300


FIGSIZE_FULL: tuple[float, float] = (6.5, 4.0)
"""用于整栏/全宽图（LaTeX 中接近 \\linewidth）。"""

FIGSIZE_HALF: tuple[float, float] = (3.2, 2.45)
"""用于半栏子图（LaTeX 中接近 0.48\\linewidth）。"""


def _resolve_times_font() -> tuple[str, list[str]]:
    """
    尽量使用 Times New Roman（WSL 常见 Windows 字体路径），否则退化为 STIXGeneral。
    """

    times_paths = [
        Path("/mnt/c/Windows/Fonts/times.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbd.ttf"),
        Path("/mnt/c/Windows/Fonts/timesi.ttf"),
        Path("/mnt/c/Windows/Fonts/timesbi.ttf"),
    ]
    if any(p.exists() for p in times_paths):
        for p in times_paths:
            if p.exists():
                fm.fontManager.addfont(str(p))
        return "Times New Roman", ["Times New Roman"]
    return "STIXGeneral", ["STIXGeneral", "DejaVu Serif"]


def paper_rcparams(style: PaperStyle | None = None) -> Dict[str, object]:
    style = style or PaperStyle()
    font_family, serif_fallback = _resolve_times_font()

    # 限制 BLAS 线程：减少 WSL/容器环境下的噪声（不影响数值正确性）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    return {
        "font.family": font_family,
        "font.serif": serif_fallback,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "axes.grid": False,
        "axes.linewidth": style.axes_linewidth,
        "lines.linewidth": style.lines_linewidth,
        "lines.markersize": style.lines_markersize,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "font.size": style.font_size,
        "axes.titlesize": style.axes_titlesize,  # 使用独立的 titlesize
        "axes.labelsize": style.axes_labelsize,
        "xtick.labelsize": style.tick_labelsize,
        "ytick.labelsize": style.tick_labelsize,
        "legend.fontsize": style.legend_fontsize,
        # 避免 Type3 字体（更利于 Overleaf/期刊印刷）
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": style.figure_dpi,
        "savefig.dpi": style.savefig_dpi,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }


@contextmanager
def paper_style(style: PaperStyle | None = None) -> Iterator[None]:
    """在 with 作用域内启用论文统一样式。"""

    with mpl.rc_context(paper_rcparams(style=style)):
        yield


def save_figure(
    fig: mpl.figure.Figure,
    out_path: str | Path,
    *,
    dpi: Optional[int] = None,
) -> None:
    """
    统一保存入口：不使用 bbox_inches='tight'，避免不同文本元素导致的
    bounding box 抖动，从而引发 LaTeX 子图错位。
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)


def despine(ax: mpl.axes.Axes) -> None:
    """轻量去除上/右边框，保持期刊图常见风格。"""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def apply_paper_style(style: PaperStyle | None = None) -> None:
    """直接应用全局 rcParams（适合脚本式制图）。"""

    mpl.rcParams.update(paper_rcparams(style=style))


def add_panel_label(
    ax: mpl.axes.Axes,
    label: str,
    *,
    # 以 axes 的左上角为锚点，用 points 做固定偏移（更稳，不受 tick 文本宽度影响）
    x: float = 0.0,
    y: float = 1.0,
    dx: float = -42.0,
    dy: float = 4.0,
    fontsize: float | None = None,
) -> mpl.text.Text:
    """
    在坐标轴左上角**外部**添加面板标签（a/b/c/d），并避免与 tick/title 冲突。

    设计原则（KISS + 稳定排版）：
    - 使用 axes fraction 作为锚点：默认 (x,y)=(0,1) 对应轴域左上角
    - 再用 points 做固定偏移：默认向左 42pt、向上 4pt
      这样不会因为 y 轴 tick label 变长（如 1.00 / 12）而发生重叠
    """

    t = ax.annotate(
        str(label),
        xy=(x, y),
        xycoords="axes fraction",
        xytext=(dx, dy),
        textcoords="offset points",
        ha="left",
        va="top",
        fontweight="bold",
        fontsize=fontsize or float(mpl.rcParams.get("axes.labelsize", 12.0)),
        color="black",
        annotation_clip=False,
    )
    t.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])
    return t
