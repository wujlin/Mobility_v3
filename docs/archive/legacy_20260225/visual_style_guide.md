# 可视化风格规范（emotion\_dynamics / Essay）

本规范用于在不同脚本/Notebook/论文之间统一制图风格，目标是“子刊可投”的简洁、可打印、跨面板一致。

**推荐作为唯一真源（source of truth）**：`src/visualization/plot_style.py`（并在 `src/plot_style.py` 提供同名转发，便于短 import）  
本文档是对其参数与用法的“人类可读版”说明，并补充语义约定与检查清单。

---

## 1. 字体与数学字体

- 正文字体：优先 `Times New Roman`；若环境缺失则回退 `STIXGeneral`（再回退 `DejaVu Serif`）。
- 数学字体：`mathtext.fontset = stix`（保证公式与 Times/serif 视觉协调）。
- 负号：`axes.unicode_minus = False`（避免某些字体下负号渲染异常）。

> 说明：该策略已在 `src/plot_style.py::_resolve_times_font()` 与 `paper_rcparams()` 中实现。

---

## 2. 字号（以 LaTeX 嵌入后的可读性校准）

默认字号（`PaperStyle`）：

- 基础字号：`font.size = 11`
- 轴标签：`axes.labelsize = 12`
- 标题：`axes.titlesize = 11`
- tick 标签：`xtick/ytick.labelsize = 10`
- 图例：`legend.fontsize = 9`（正文的 80–90%）

> 半栏图在双栏排版中会被缩放，低于 `8pt` 不建议使用。

---

## 3. 线宽、marker 与坐标轴

- 坐标轴线宽：`axes.linewidth = 1.2`
- 主线线宽：`lines.linewidth = 2.4`
- marker 大小：`lines.markersize = 5.5`
- tick：`major.size = 4.0`，`major.width = 1.1`
- 默认不画网格：`axes.grid = False`

参考线（常用约定）：

- 零线：`color=#666666`，`lw≈0.9–1.2`
- 临界点 `r_c`：灰色虚线 `linestyle=':'`，`lw≈1.2`，`alpha≈0.6–0.8`

---

## 4. 尺寸（物理尺寸与 LaTeX 插入宽度匹配）

为了避免 LaTeX 再缩放导致字号过小，脚本中的 `figsize` 应与最终插入宽度同量级：

- 全栏/全宽：`FIGSIZE_FULL = (6.5, 4.0)`（LaTeX `\linewidth`）
- 半栏：`FIGSIZE_HALF = (3.2, 2.45)`（LaTeX `0.48\linewidth`）

多子图建议：

- 面板内元素尽量少（尤其是 legend），优先把 legend 放到轴外统一位置。
- 不依赖 `bbox_inches='tight'` 来“自动裁剪”，否则不同面板会出现 bounding box 抖动。

---

## 5. 配色（色盲友好 Okabe–Ito）

统一使用 Okabe–Ito 调色板（`src/plot_style.py:OKABE_ITO`）：

| 名称 | Hex | 典型用途 |
|---|---|---|
| `blue` | `#0072B2` | Theory / 主曲线 / |Q| |
| `bluish_green` | `#009E73` | Activity `a` |
| `vermillion` | `#D55E00` | ABM / 对照曲线 |
| `orange` | `#E69F00` | 次要强调（谨慎使用） |
| `gray` | `#777777` | 参考线 / 次要曲线 |
| `black` | `#000000` | 文本/坐标轴 |

语义映射建议（尽量全篇一致）：

- **Polarization**：`OKABE_ITO["blue"]`
- **Activity**：`OKABE_ITO["bluish_green"]`
- **ABM vs Theory**：Theory 用 `blue` 实线；ABM 用 `vermillion` 虚线或点线

---

## 6. 置信区间/不确定性（band）

优先：`fill_between` 画 **shaded band**（而非 errorbar 或 legend 里写一长串）。

推荐参数：

- `alpha ≈ 0.12–0.22`
- `linewidth = 0`（必要时可加极细边线，但注意不要“抢戏”）
- 与主线同色系（更直观）

注意：

- 当 CI 很窄时，band 可能印刷尺度下不显眼，这是合理现象（稳定收敛）。
- caption 可说明 “bands are narrow away from the transition”，但避免过度“辩护式”语气。

---

## 7. 图例（Legend）与面板标签（a/b/c/d）

### 图例（Legend）

原则：**优先轴外、下方居中**，避免遮挡数据与 tick/标题冲突。

常用模板：

```python
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.20),
    ncol=2,
    frameon=False,
)
```

### 面板标签（a/b/c/d）

统一使用 `add_panel_label()` 放在轴域外，避免与 y 轴 tick label 重叠：

- 默认锚点：轴域左上角 `(x,y)=(0,1)`
- 默认偏移：`dx=-42pt, dy=+4pt`（稳定，不随 tick 文本宽度变化）
- 字体：加粗，黑色，带白色描边（提高可读性）

---

## 8. 导出与文件格式

- 主论文与 SI：**优先导出 PDF（矢量）**。
- 预览：可额外导出 PNG（仅用于快速 review，不用于投稿）。
- DPI：
  - `figure.dpi = 150`（屏幕显示）
  - `savefig.dpi = 300`（导出）
- 字体嵌入：`pdf.fonttype = 42`（避免 Type3 字体，利于 Overleaf/印刷）。

强制约定：

- **不使用** `bbox_inches="tight"`（避免不同文本元素导致 bounding box 抖动，引发 LaTeX 子图错位）。
- 统一用 `save_figure(fig, path)` 保存（见 `src/plot_style.py`）。

目录建议：

- 主文图：`Essay/figures/`
- 补充图：`Essay/figures_supp/`

---

## 9. Python 用法模板（推荐复制）

```python
import matplotlib.pyplot as plt

from src.plot_style import (
    OKABE_ITO,
    FIGSIZE_HALF,
    FIGSIZE_FULL,
    paper_style,
    add_panel_label,
    save_figure,
)

with paper_style():
    fig, ax = plt.subplots(figsize=FIGSIZE_HALF)
    ax.plot(x, y, color=OKABE_ITO["blue"], label="Theory")
    ax.fill_between(x, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, linewidth=0)
    ax.axvline(rc, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.2, alpha=0.7)
    ax.set_xlabel(r"Control parameter $r$")
    ax.set_ylabel(r"Polarization $|Q|$")
    add_panel_label(ax, "a")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
    save_figure(fig, "Essay/figures/example.pdf")
```

---

## 10. 出图检查清单（交付前必过）

- [ ] 字体一致（Times/STIX），数学字体一致（STIX），无中文字符混入
- [ ] legend 在轴外；不遮挡数据；字号 `≈9pt`
- [ ] 面板标签在轴外；不与 tick label 重叠
- [ ] 坐标轴线宽、主线线宽一致；虚线/参考线风格一致
- [ ] CI band 用 shaded band；不要把方法细节塞进 legend
- [ ] PDF 导出无 Type3 字体；未使用 `bbox_inches="tight"`
- [ ] 半栏图在双栏缩放后仍可读（tick≥10、legend≥9）

