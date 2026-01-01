Implementation Plan, Task List and Thought in Chinese：本文件是“最后 1–2 小时冲刺写作”的最小骨架，目标是在**不夸大、不造假**的前提下，把你们已经做对的“方法论严谨性 + 可复现性 + 证据链”写成一篇得分高的期末报告（docx）。

# Essay 冲刺写作指南（基于真实结果）

> 交付要求来源：`essay/requirements.md`（docx、15 页左右、Times New Roman 12、1.5 倍行距、图表≤5页、必须有个人贡献与 AI 声明）

---

## 0) 先说底线：不要“包装/生成预期结果”

你现在时间紧、想拿高分非常合理，但**伪造或选择性误导**属于学术不端，风险远大于收益。  
更稳妥的高分策略是：把“严格 pipeline + 负结果诊断 + 专家反馈 + 迭代路线”写得清楚，让 instructor 看到你们掌握了科学研究的方法，而不是只堆分数。

---

## 1) 你这篇报告最强的主线（建议照抄到 Introduction 最后）

**主线一句话**：我们研究在已知 OD（KnownDestination）条件下的出租车未来轨迹生成；提出并评估 data-only diffusion 与 physics-conditioned diffusion（nav_field 条件），在严格的 dt-fixed=30s、train-only 产物合同与无泄漏评估协议下，系统对比了确定性回归与生成式模型在**覆盖能力（best-of-K）**与**物理一致性（MSD/Rog）**上的权衡，并对“宏观收缩/走不动”现象做了可复现的根因诊断与改进尝试。

你们的贡献不只是“结果”，更是：
- strict 数据合同（train-only nav_field & stats + hash）
- 评估协议补齐（K=20、best-of-K、Fréchet、DTW、GT MSD/Rog）
- 对 shrinkage 的排雷式诊断（vel_scale / macro loss / gating / batch norm）

---

## 2) Results：最省时间的“1 表 + 2 图”组合

### 2.1 主表（Phase B quick, test, 320 conditions）

直接引用：`docs/archive/phase_b/PHASE_B_RESULTS.md` 第 4 节的 quick 表格（已整理好）。

你在正文里只需要抓 3 句话：
- **确定性 baseline（K=1）**：`ADE_mean=5.467, FDE_mean=8.855`，宏观指标接近 GT（`Rog=5.494` vs `GT_Rog=5.247`，`MSD_10=304.099` vs `GT_MSD_10=349.740`）。
- **生成模型（K=20）**：Diffusion/Physics 的 **best-of-K 明显更好**（Diffusion `ADE_best=2.836`，Physics `ADE_best=2.510`），说明具备更强的多模态覆盖潜力。
- **但宏观收缩明显**：Diffusion `Rog=3.056 (0.582×GT)`, `MSD_10=100.948 (0.289×GT)`；Physics `Rog=3.435 (0.655×GT)`, `MSD_10=116.869 (0.334×GT)` —— physics 能缓解但不足以根治。

对应文件（可复现）：
- `data/experiments/baseline_b_dt30_eval_quick/metrics.json`
- `data/experiments/diff_b_dt30_eval_quick/metrics.json`
- `data/experiments/physics_b_dt30_eval_quick/metrics.json`

### 2.2 图 1：微观指标对比（mean + best-of-K）

建议画法：同一张图里放 `ADE_mean/ADE_best/FDE_mean/FDE_best`（或分两张）。  
如果来不及画，直接用现成脚本重出（见 `docs/archive/phase_b/PHASE_B_RESULTS.md` 第 5 节的命令），并在 caption 标注 quick=320。

### 2.3 图 2：MSD(τ) 曲线叠图（Pred vs GT）

这是“期刊风格”的硬证据图：横轴 `τ=k*30s`，纵轴 MSD(τ)，至少展示 baseline / diffusion / physics / GT 四条曲线。  
目的：把“收缩”从一句话变成一张图。

### 2.4（可选加分）图 3：地理空间可视化（地图式展示）

你想强调“城市空间意义/复杂动力学”，最省时间的加分图是：
- 将 evaluate 保存的 `samples.npz` 从 grid `[y,x]` 线性映射回经纬度 bbox，做轨迹叠图 + 密度图。

一键命令与注意事项见：`docs/GEO_VISUALIZATION.md`

---

## 3) Discussion：用“诊断链”拿分（比硬拉 SOTA 更稳）

建议分 3 小段，每段都要有“观察→解释→证据路径”：

### 3.1 为什么 ADE_mean 不该成为生成模型的唯一目标
- 生成式模型追求覆盖未来分布；确定性回归追求条件均值。
- 因此 best-of-K、Fréchet/DTW、MSD/Rog 更能反映“风险感知与物理真实”。

### 3.2 Shrinkage（走不动）的根因与证据
直接引用并压缩 `docs/archive/phase_b/ROOT_CAUSE_ANALYSIS.md`：
- 排除：统计量 mismatch / padding 污染 / 单纯欠拟合
- 剩下：高不确定性下的均值回归倾向 + nav_field 的保守先验

### 3.3 我们做了哪些“严谨但仍未完全解决”的改进尝试
这里写成 ablation（真实且高分）：
- 推理期 `vel_scale`：能拉近宏观幅度，但会放大方向误差导致 ADE/FDE 变差（说明尺度≠根治）。
- 训练期 macro loss：从 Rog 到 EPE（端到端位移），并做 timestep gate；仍存在平台区/权衡（可作为“失败的但严谨的尝试”）。
- 结构性修复（v1.1 residual）：prior+residual decomposition 把“尺度”交给 deterministic prior，把“随机性”交给 diffusion；在 fast eval 中能显著缓解 shrinkage（可作为“阶段性成功的 pivot”）。

给出一个小表（可放附录）：展示 `pred_speed/gt_speed`、`MSD10/GT`、`Rog/GT` 的 ratio。

例：Physics batch-normalized EPE（val, K=1, 51200）：
- `λ=0.01/0.03/0.06` 几乎相同：`speed_ratio≈0.597`，`MSD10_ratio≈0.261`，`Rog_ratio≈0.536`
（对应：`data/experiments/phys_ft_batchEPE_l0.{01,03,06}_t50_lr3e-4_e22_s0_eval_val_k1/metrics.json`）

结论写法（建议原句）：  
“Macro loss 在当前实现下尚未根治 shrinkage，但它为后续提供了可解释的控制旋钮；下一步的关键不是继续扫 λ，而是按 GT 位移对 macro 信号做加权/筛选以避免被低位移窗口稀释。”

补充一句（如果篇幅允许）：  
“进一步地，我们采用 residual decomposition（prior+residual）作为结构性修复，使宏观尺度从‘走不动’转为‘保守 tether’，将后续工作聚焦到 conditioning 注入方式上。”

---

## 4) Conclusion：不要写“我们解决了”，写“我们建立了可复现闭环并定位瓶颈”

结论 3 句足够：
1) 我们构建了 dt-fixed=30s、无泄漏合同的数据/训练/评估闭环；  
2) 生成式模型在 best-of-K 与分布指标上显示出更强覆盖潜力，physics 条件进一步提升最优覆盖；  
3) 发现并量化了宏观收缩瓶颈，给出可复现诊断与下一步可验证的修复路线（位移加权 macro loss / 更稳的 low-frequency supervision）。
（可选替换为更贴近最新状态）：位移加权的 deterministic prior + residual decomposition + 更合理的 nav\_field 注入（避免 mean-field tether）。

---

## 5) 你必须补齐的两段声明（按 requirements）

### 5.1 个人贡献（每个人都要写）
你可以按“数据/代码/实验/写作”四类写：
- 我负责：dt-fixed 数据制作与 strict 合同、训练/评估脚本实现、实验运行与结果分析、报告撰写（哪些章节）。
- 合作者负责：例如 nav_field 构建、文档规范、baseline 训练、可视化脚本等。

### 5.2 AI 声明（如实写）
可参考 `essay/sections/06_ai_declaration.tex` 的模板，写清楚 AI 用在：
- brainstorming/润色/代码 review/排错建议（如果有）
- 最后由作者复核并对内容负责
