# 可视化风格规范（Nature 级期刊标准）

本规范用于在不同项目/脚本/论文之间统一制图风格，以 Nature 系列期刊的 figure guidelines 为基准。

**唯一真源（source of truth）**：`src/visualization/plot_style.py`（`src/plot_style.py` 提供同名转发，便于短 import）。
本文档是其"人类可读版"说明，补充语义约定与检查清单。

> **参考标准**：[Nature Research Figure Guide](https://research-figure-guide.nature.com/)、[Springer Nature artwork guidelines](https://www.springernature.com/gp/authors/campaigns/writing-a-manuscript/figure-preparation)

---

## 1. 尺寸

Nature 只提供两档宽度，最终排版时可能进一步缩小。建议按"最小但仍清晰可读"的尺寸提交，确保字号在 5–7 pt 范围内。

| 类型 | 宽度 | figsize 建议 | 备注 |
|---|---|---|---|
| **单栏** | 89 mm (3.5 in) | `(3.5, 2.6)` | 最常用，适合单面板图 |
| **双栏/全宽** | 183 mm (7.2 in) | `(7.2, 4.5)` | 适合多面板复杂图 |

- **最大高度**：170 mm (6.7 in)——需留出底部图注空间
- 生产排版可能缩图，因此不要依赖"提交尺寸=印刷尺寸"，要确保**缩小后仍可读**
- 文件大小控制在 **≤ 50 MB**

```python
# src/visualization/plot_style.py 中的尺寸常量
FIGSIZE_HALF = (3.5, 2.6)   # 单栏
FIGSIZE_FULL = (7.2, 4.5)   # 双栏/全宽
```

---

## 2. 面板排列（Panel Arrangement）

Nature 要求面板**紧凑节省空间**，尽量减少无意义留白。

- 面板按字母顺序排列（a, b, c, …），阅读顺序从左到右、从上到下
- 面板边缘对齐，行列整齐——避免"随意拼贴"式布局
- 面板大小由**内容和可读性**决定，不必强行统一：有的面板需要更大空间，有的不需要被不成比例地放大
- 不同面板之间的间距保持一致（建议 2–4 mm）

---

## 3. 字体

### 字体族

Nature 要求 **sans-serif**（Helvetica、Arial 为首选）。所有文字必须**可编辑**（不可转曲/描边为轮廓 outline）。

| 优先级 | 字体 | 说明 |
|---|---|---|
| 1 | Helvetica | Nature 标准，macOS 内置 |
| 2 | Arial | Windows/WSL 常见替代 |
| 3 | DejaVu Sans | matplotlib 内置兜底 |
| 等宽 | Courier | 序列/代码专用 |

- 数学字体：`mathtext.fontset = "dejavusans"` 或 `"stixsans"`
- 字体嵌入：`pdf.fonttype = 42`（TrueType），**禁止** Type3 字体
- **禁止**将文字转为轮廓（outline）——Nature 需要可编辑文本

### 字号

Nature 硬性要求：**5–7 pt**。考虑到排版可能缩图，建议按 7 pt 设计。

| 元素 | 字号 | 说明 |
|---|---|---|
| 轴标签 | 7 pt | `axes.labelsize` |
| tick 标签 | 6 pt | `xtick/ytick.labelsize` |
| 图例 | 6 pt | `legend.fontsize` |
| 面板标签 (a/b/c) | 8 pt, **bold** | 见 `add_panel_label()` |
| 标题（若有） | 7 pt | `axes.titlesize`；通常不加 title，用 caption |

> **底线**：最终印刷时任何文字不小于 5 pt。如使用非 Nature 尺寸，需换算缩放后的实际字号。

---

## 4. 线宽与 Marker

Nature 要求：**最终印刷时线宽不小于 0.25 pt**（推荐 0.5–1.5 pt）。

| 元素 | 值 | 说明 |
|---|---|---|
| 数据线 | 1.0–1.5 pt | `lines.linewidth`；粗于 2 pt 在单栏图中会显得笨重 |
| 坐标轴 | 0.8 pt | `axes.linewidth` |
| tick | major.size=3.5, width=0.8 | |
| marker | 4–5 pt | `lines.markersize`；不宜过大，避免遮挡数据 |
| 参考线 | 0.6–0.8 pt, `linestyle=':'` | 灰色虚线/点线 |

---

## 5. 配色与无障碍（Accessibility）

Nature 遵循 WCAG 2.1 AA 标准，要求作者在主图与 Extended Data 中均考虑可访问性。

### 5.1 色盲友好调色板（强制要求）

Nature 官方推荐的色盲友好调色板（即 Okabe–Ito）：

| 名称 | Hex | RGB | 用途建议 |
|---|---|---|---|
| `black` | `#000000` | 0, 0, 0 | 文本 / 坐标轴 |
| `orange` | `#E69F00` | 230, 159, 0 | 强调（谨慎使用） |
| `sky_blue` | `#56B4E9` | 86, 180, 233 | 浅色背景 / 辅助 |
| `bluish_green` | `#009E73` | 0, 158, 115 | 第三组数据 |
| `yellow` | `#F0E442` | 240, 228, 66 | 高亮（慎用，对比度低） |
| `blue` | `#0072B2` | 0, 114, 178 | 主曲线 / 主方法 |
| `vermillion` | `#D55E00` | 213, 94, 0 | 对照 / baseline |
| `reddish_purple` | `#CC79A7` | 204, 121, 167 | 第六组数据 |

### 5.2 配色原则

- **不依赖颜色传递信息**：同时使用线型（实线/虚线/点线）或 marker 形状作为区分
- **避免红绿对比**：红绿色盲无法区分。用 blue vs vermillion 替代
- **避免彩虹色图**（jet / rainbow）：用 `viridis`、`cividis` 等感知均匀色图
- **语义映射全篇一致**：确定了"蓝=方法 A，红=方法 B"后，全文所有图保持一致
- **图内放色键（color key）**：颜色说明尽量在图中直接标注，不要全部放到 caption 里（色盲读者难以对应）

### 5.3 文字对比度

- 文字与背景的对比度 ≥ **4.5**（WCAG AA 标准）
- **尽量避免彩色文字**：能用黑/白就用黑/白，配高对比背景
- 若彩色文字对科学含义不重要，出版方可能会改为黑/白

### 5.4 避免装饰性图标

- 能用文字标签就用文字，不要用容易引发歧义的图标（如鼠形/人形剪影当分组标签）
- 仅在图标提供必要科学语境时才可使用

---

## 6. 面板标签（a, b, c, d）

Nature 要求：**小写粗体字母**，放在面板**左上角外侧**。

```python
from src.plot_style import add_panel_label
add_panel_label(ax, "a")  # 默认锚点 (0,1)，向左偏移 42pt
```

- 字体：加粗、黑色
- 位置：轴域外（不遮挡数据），通过固定 pt 偏移避免与 tick label 冲突
- 所有面板标签的位置和字号必须统一

---

## 7. 图例（Legend）

原则：**简洁，不遮挡数据**。

- 优先放在轴外（下方居中或右侧），避免覆盖数据区域
- `frameon=False`（不画边框）
- 每条图例项用最少的文字，避免长句
- 多面板共享同一组图例时，只放一份（通常在底部居中）

```python
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.18),
    ncol=3,
    frameon=False,
)
```

---

## 8. 不确定性（置信区间 / 误差）

优先使用 `fill_between` 画 **shaded band**（而非 errorbar）：

- `alpha ≈ 0.15–0.25`
- `linewidth=0`（band 边界不画线）
- 与主线同色系
- 当 CI 很窄时 band 可能不显眼，这是正常现象（说明结果稳定）

Errorbar 仅在数据点稀疏时使用（如 bar chart），cap 长度适中。

---

## 9. 坐标轴与网格

- **默认不画网格**：`axes.grid = False`（Nature 风格偏简洁）
- **去除上/右边框**：`despine(ax)` 保留左/下边框
- **轴标签用 sentence case**：`"Arrival rate (%)"`，不用 Title Case
- **数学符号用 LaTeX**：`r"Polarization $|Q|$"`
- **负号**：`axes.unicode_minus = False`

---

## 10. 导出与文件格式

### 主图（Main Figures）——必须矢量

Nature 主图要求**可编辑图层的矢量文件**。

| 格式 | 状态 | 说明 |
|---|---|---|
| **.pdf** | ✅ 首选 | 矢量，保留编辑能力 |
| **.eps** | ✅ 接受 | 矢量，兼容性好 |
| **.ai** | ✅ 接受 | Adobe Illustrator 原生 |
| **.svg** | ✅ 接受 | 用 plain SVG（非 Inkscape SVG） |
| .jpeg / .tiff / .png | ❌ **不接受** | 光栅格式不可用于主图 |

- 所有元素嵌入文件中，不以链接方式引用
- 文件大小 ≤ **50 MB**
- 不合并图层、不扁平化

### Extended Data 图——可用光栅

Extended Data 与主图要求不同：

| 格式 | 状态 |
|---|---|
| **.jpeg** | ✅ 首选 |
| **.tiff** | ✅ 接受 |
| **.eps** | ✅ 接受 |

- 色彩模式：**RGB**（不是 CMYK）
- 分辨率上限：**300 dpi**
- 文件大小 ≤ **10 MB**
- 命名规则：`AuthorSurname_EDfig1.jpg`

### matplotlib 关键设置

```python
"pdf.fonttype": 42,       # 嵌入 TrueType，避免 Type3 字体
"ps.fonttype": 42,
"savefig.dpi": 300,       # PNG 预览用
```

### 禁止事项

- **不使用** `bbox_inches="tight"`：会导致不同文本元素引起 bounding box 抖动，多面板组图时子图错位
- **主图不使用** JPEG / TIFF / PNG
- 统一用 `save_figure(fig, path)` 保存

---

## 11. Python 用法模板

```python
import matplotlib.pyplot as plt

from src.plot_style import (
    OKABE_ITO,
    FIGSIZE_HALF,
    FIGSIZE_FULL,
    paper_style,
    add_panel_label,
    save_figure,
    despine,
)

with paper_style():
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    ax = axes[0]
    ax.plot(x, y, color=OKABE_ITO["blue"], lw=1.2, label="Method A")
    ax.fill_between(x, lo, hi, color=OKABE_ITO["blue"], alpha=0.18, lw=0)
    ax.set_xlabel("Control parameter")
    ax.set_ylabel(r"Response $R$")
    despine(ax)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.plot(x, z, color=OKABE_ITO["vermillion"], lw=1.2, label="Baseline")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    despine(ax)
    add_panel_label(ax, "b")

    fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.02),
        ncol=2, frameon=False,
    )
    save_figure(fig, "figures/example.pdf")
```

---

## 12. 出图检查清单（交付前必过）

### 技术规范
- [ ] 物理尺寸匹配目标版面（单栏 89mm / 双栏 183mm），高度 ≤ 170mm
- [ ] 字号在 5–7 pt 范围内
- [ ] 最终印刷时最小线宽 ≥ 0.25 pt（推荐 ≥ 0.5 pt）
- [ ] PDF 导出无 Type3 字体（`pdf.fonttype = 42`）
- [ ] 文字可编辑，未转为轮廓（outline）
- [ ] 未使用 `bbox_inches="tight"`
- [ ] 主图为矢量格式（PDF/EPS），非 JPEG/TIFF/PNG
- [ ] 文件大小 ≤ 50 MB

### 排版与可读性
- [ ] 面板紧凑排列，边缘对齐，无多余留白
- [ ] 面板标签（a/b/c）小写加粗，在轴外，位置统一
- [ ] 图例不遮挡数据，无边框，字号一致
- [ ] 坐标轴标签 sentence case，数学符号用 LaTeX
- [ ] 去除上/右边框（除非有特殊理由保留）

### 色彩与无障碍
- [ ] 配色色盲友好（Okabe–Ito 或等效）
- [ ] 不仅靠颜色区分数据——同时使用线型/marker
- [ ] 语义配色全文一致（同一方法在所有图中同色）
- [ ] 未使用 rainbow/jet 色图
- [ ] 文字与背景对比度 ≥ 4.5（避免彩色文字，优先黑/白）
- [ ] 未使用装饰性图标（用文字标签替代）
- [ ] 颜色说明在图内标注（color key），不仅放在 caption

### 数据完整性
- [ ] CI/errorband 使用 shaded band，`alpha` 适中
- [ ] 所有图中的数字与实验产物文件一致
- [ ] 多面板图的坐标轴范围合理对齐（共享轴时一致）
