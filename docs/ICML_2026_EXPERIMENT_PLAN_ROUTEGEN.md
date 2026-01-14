Implementation Plan, Task List and Thought in Chinese

# ICML 2026：Route Generation（CascadeTraj）实验思路与执行计划（PI Review Draft）

> 目标：用**可诊断的证据链**支撑论文主张——长程路线生成的核心瓶颈是“拓扑多模态导致的 mode interference/averaging”，而**决策-执行级联（先离散承诺，再连续执行）**能系统性缓解该失败模式；soft prior 提升可行性但不做 hard truncation；census 仅作为可选引导（extensibility），不把论文变成 population synthesis。

> [!IMPORTANT]
> **2026-01-14 重大更新（请新 PI 优先读这一段）**：  
> 我们已确认 window-level（F=256 滑窗）会把 route generation 降级为短距离轨迹延续，导致“走廊选择/语义条件化”相关证据链失真。当前 ICML routegen 主线已转为：  
> **segment-level route → road graph path → waypoint-level AR（3–5 步）→ A\* 连接（可选 continuous execution）**。  
> 因此，本文件中 **E0–E13（windows/cascade/diffusion）** 作为历史记录保留，但不再作为当前主线的可复现入口。  
> 当前主线入口请看：`docs/PI_BRIEF_ROUTEGEN_ICML2026.md` 与 `docs/WORKSTATION_GUIDE.md` 的 `T3/T4` 目录约定。

---

## 0) 范围与三条核心 Claim（写作/实验必须对齐）

### 0.1 本文不做什么（边界声明）
- **不做** urban rupture / avoidance field（这属于第二篇文章）。
- **不做** population synthesis 的完整闭环（个体属性生成不是本文主线）。
- census 相关只作为“可选语义引导/可控性扩展”的 **ablation**，用一段话在 Introduction 提前封堵歧义。

### 0.2 本文要证明什么（Claim → 需要的证据类型）
**Claim A（诊断发现）**：端到端连续坐标生成在 full-trip route generation 上会出现模式坍塌，其根因是 corridor-level 多模态在连续输出空间发生 mode interference，导致 destructive averaging（“直线/模糊走廊”）。
- 证据：**Fig.1/2 级别的可视化诊断** + dataset-level 的 diversity/coverage 指标崩溃（不是只报 FDE/ADE）。

**Claim B（机制性解法）**：用“决策（离散/稀疏承诺）→执行（连续生成）”的级联分解，可显式承载拓扑多模态，从机制上打破 averaging trap，恢复 corridor-level diversity，同时保持 intent consistency。
- 证据：对比端到端 baseline vs CascadeTraj，在**同一条件**下多样性显著提升，同时 intent/realism 不退化。

**Claim C（鲁棒性与可行性）**：地图信息更适合作为 **soft prior**（road proximity / dist-to-road / road_prob），而非训练期 hard mask/裁剪；hard truncation 能“看起来更可行”但会把分布上限绑定到 proxy 质量并削掉真实模态。
- 证据：soft prior vs hard mask 的 ablation：可行性提升同时多样性不被截断；并做 dilation/buffer sensitivity audit 避免被 mask 孔洞污染（见 `docs/README.md` 的踩坑提醒）。

---

## 0.3 当前主线（GraphCascade｜segment-level + graph + waypoint AR + A*）

> 这一节是 **当前可执行** 的实验路线；其余章节（E0–E13）为旧口径历史记录。

### 数据与产物（统一以 `$RAW_ROOT/experiments/icml2026_routegen/` 为根）

- segment-level routes（固定长度、epoch 时间）：`gt_segments/*_segments_route_F256_epoch_seed0.npz`
- road graph：`G1_roadgraph_*/road_graph.npz` 与多城合并的 `T3_combo_*/road_graph_combo.npz`
- GT graph paths：`T3_combo_*/paths_graph_combo.npz`
- GT waypoints（从 GT path 提取固定 K）：`T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/waypoints_graph.npz`

### Gate-1：候选覆盖诊断（为什么不用 K-shortest+classify）

目的：量化 “候选集覆盖不足” 是否是系统性问题。  
脚本：`src/data/road_graph/gate_candidate_paths_from_routes_npz.py` + `src/data/road_graph/diagnose_candidate_coverage.py`

通过标准（建议）：
- `gt_point_dist_to_road_p90` 较小（GT 与路网对齐没问题）
- 但 `best_jaccard_p50` 仍显著偏低（说明 K-shortest 覆盖不到 GT 走廊）→ 进入 AR/waypoint 路线

### Gate-2：语义信息量（time+tier 是否能区分走廊簇）

目的：验证 “context-conditioned diversity” 是否有可观测信号支撑（AUC > 0.6）。  
脚本：`src/data/road_graph/gate_semantic_informativeness_cluster.py`

### T3（已验证失败）：node-level AR（600 步）累积误差过大

结论：即使 teacher-forcing accuracy 看似不低，生成时长程累积误差会导致覆盖崩溃；不作为最终方案。

### T4（当前方案）：Waypoint AR（3–5 步） + A\*

目的：把 AR 步数从 600 降到 3–5，消除长程累积误差；把可行性交给 A\*。  
脚本：
- dump waypoint：`src/data/road_graph/dump_waypoints_from_paths_graph_npz.py`
- 训练 waypoint AR：`src/training/train_graph_ar_waypoint_bins.py`
- 评估 waypoint AR + A\*：`src/training/sample_graph_ar_waypoints_astar.py`

关键指标：
- `success_rate`（K 次采样中能否连到 D）
- `best-of-K Jaccard (edge set)`（合法路径前提下的走廊覆盖）

---

## 1) 数据与路径（按仓库合同口径，避免“路径不一致导致无法复现”）

### 1.1 外置数据根目录（推荐）
按 `docs/DATA_STRUCTURE.md`：Phase D 默认外置 `$RAW_ROOT`（不进 git）。

典型结构（示例，实际以你机器为准）：
- `$RAW_ROOT/worldtrace/<city>_core_v1/segments.parquet`
- `$RAW_ROOT/worldtrace/<city>_core_v1/osm_road_prob.npy`（或 dist-to-road/road_mask）
- `$RAW_ROOT/worldtrace/<city>_core_v1/poi_density_*.npy` / `landuse_*.npy`
- `$RAW_ROOT/census/<city>_core_v1/...`（若启用 census ablation）

> 口径真相源：`docs/DATA_CONTRACT.md`（bbox/grid/坐标/road_prob 定义）。

### 1.2 立刻要跑通的“最小可训练样本形态”
为了支持 route generation（O,D,t0,context → full route），数据至少要能提供：
- 轨迹序列（位置序列）与时间戳
- trip-level 的 origin/destination
- 可选的 context channels（OSM soft prior，POI/imagery，census covariates）

如果你当前的 WorldTrace 产物还是 parquet/segments 形态，建议先做一个**轻量转换**（不追求最优 IO，只求 48 小时内跑通 baseline）：
- 从 `segments.parquet` 采样一个可控规模的子集（例如 50k–200k segments）
- 统一投影到 `(y,x)` 栅格（bbox/grid 以合同写死）
- 导出一个“训练可直接 mmap/顺序读取”的格式（parquet 分区/arrow/npz 均可，KISS 优先）

> 避免一次性工程化：先跑通“诊断 baseline + 关键图”，再考虑 manifest/更高效格式（`docs/WORDTRACE_UNITRAJ.md` 的 IO 风险提醒）。

---

## 2) 任务定义（实验口径写死，避免审稿人觉得你在换任务）

### 2.1 条件与输出
条件（最小）：$c = (o, d, t_0)$  
可选条件：OSM soft prior、POI/imagery、（ablation）census covariates  
输出：完整路线 $\tau = (p_1,\dots,p_T)$ 或等价的 action/velocity 表示（最终都可还原到位置序列用于评估）。

### 2.2 采样设置
- 生成模型统一采样 `K` 条（建议 `K=20`，快速阶段可 `K=5`）。
- 所有模型统一随机种子集合（至少 3 个 seed），否则 diversity 对比不可比。

---

## 3) 模型与 Baseline 设计（只保留能回答 Claim 的最小集合）

### 3.1 必要 Baselines（直接对打）
1) **End-to-End AR**：自回归坐标/位移模型（检验 drift 与误差累积）。
2) **End-to-End Diffusion/Flow**：端到端连续序列生成（直接暴露 mode collapse/averaging）。
3) **Hard mask / hard support（诊断项）**：训练期或采样期的输出裁剪/masked softmax 版本（用于展示“可行性来自 truncation，并非真正学到分布”）。

> 备注：Hard mask baseline 的定位是“诊断/止损线”，不要把它写成主贡献能力（与仓库既有经验一致）。

### 3.2 我们的方法（CascadeTraj）
**Stage-1 决策（Topological Commitment）**  
- 输出：稀疏 waypoint skeleton（如 2 个 waypoint + end anchor）
- 监督信号（KISS 优先）：从 GT 路线几何**启发式抽取**固定数量的 waypoints（例如 Fixed-K RDP：largest deviation first），作为 Stage-1 的训练目标；对应实现可复用 `src/features/waypoints.py` 的 `rdp_dev`（并在 Method 中明确写 “We derive ground-truth skeleton by ...” 以避免歧义）。
- 模型：AR 或 diffusion 均可（以能稳定表达多模态为准）

**Stage-2 执行（Physical Execution）**  
- 输入：waypoints + (o,d,t0,context)
- 输出：连续路线（全分辨率）
- 重点：执行层的目标是“像真 + 可行”，但不应承担“创造拓扑模态”的责任（模态应在 Stage-1）。

### 3.3 分阶段 Go/No-Go（避免 10 天内陷入工程黑洞）
- **Gate-0（数据可用）**：能生成 GT 轨迹可视化 + 能采样若干 OD 的候选样本（哪怕是 dummy baseline）。
- **Gate-1（诊断成立）**：端到端 diffusion 在关键 OD 上出现平均化/走廊混叠（Fig.1 的左半边成立）。
- **Gate-2（级联有效）**：仅加入“离散承诺（waypoints）”就能让多走廊样本显著分离（Fig.1 右半边成立），且 diversity 指标显著上升。
- **Gate-3（soft prior 提升可行性）**：soft prior 提升 on-road proxy 且不过度削多样性；hard mask 的“提升”需伴随分布截断的证据。

---

## 4) 指标体系（必须覆盖“多模态”，否则 Claim A/B 站不住）

### 4.1 Intent consistency（长程一致性）
- 终点误差（FDE/endpoint error）
- 到达率/终止一致性（是否到达目的地邻域）

### 4.2 Realism（几何与形状）
- DTW / Fréchet 类形状距离（比单点 FDE 更能抓“形状像不像”）
- 路径长度/绕行比的分布一致性（避免“看似到达但走直线”）
- **多 GT 的匹配规则（必须写死）**：同一条件下若存在多条 GT 路线（本身多模态），则对每条生成样本 $\hat{\tau}$ 计算 `min-DTW`（或 `min-Fréchet`）到该条件的 GT 集合；再对 $K$ 个生成样本汇总（例如 `mean(min-DTW)` 与 `best-of-K` 两个口径同时报告，避免只看单一样本导致误判）。

### 4.3 Feasibility proxies（可行性：必须做敏感性审计）
- on-road ratio / off-road rate：基于 OSM road mask 或 dist-to-road
- **敏感性审计**：buffer/dilation/threshold 扫描，避免“mask 孔洞”把模型冤死或把 hard mask 美化（对应 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md` 的经验）

### 4.4 Diversity / Coverage（核心卖点）
我们需要一个“能区分**一条好路线** vs **多条不同走廊的好路线**”的指标族。建议采用两层口径（KISS + 可复现）：

**(a) 条件内多样性（intra-condition diversity）**  
给定同一条件 $c=(o,d,t_0,ctx)$，对生成的 $K$ 条路线：
- **Pairwise Jaccard Distance（占用栅格集合）**：把路线 rasterize 到低分辨率网格（例如 64×64 或 128×128）得到占用集合 $S(\tau)$，定义
  - $D_{\text{Jacc}} = 1 - \frac{|S(\tau_i)\cap S(\tau_j)|}{|S(\tau_i)\cup S(\tau_j)|}$
  - 汇总：均值/分位数（越大表示越多样）。
- **Self-BLEU（离散 token 序列）**：将路线压缩为网格 token 序列（去重/下采样），计算 self-BLEU（越低越多样）。

**(b) 走廊覆盖（corridor coverage）**  
核心是“是否覆盖到 GT 的主要走廊模态”。最小可行做法：
- 在每个条件下，对 GT 路线做聚类得到走廊簇 $\{\mathcal{C}_m\}$。KISS 版本建议先复用 `src/evaluation/od_multimodality_gate.py` 的几何特征（signed deviation / progress / length ratio）做 `k=2` 聚类来定义“两个走廊模态”，确保流程可跑通；更复杂的 occupancy/embedding 聚类作为后续增强。
- 生成样本覆盖率：生成集合与 GT 簇的匹配比例（例如每个 GT 簇是否至少被一个生成样本命中，或按簇权重算 recall）。
- **Corridor clustering audit（E0 必交付）**：对 3–5 个关键 OD case，画出 GT 路线叠图并按簇着色，人工确认簇确实对应“肉眼可辨的走廊”；并做一次参数敏感性扫描（KMeans 的 `k` 或 DBSCAN 的 `eps`）确认结论不依赖拍脑袋参数。若高度敏感，需在论文中诚实声明限制。

> 通过这两层指标，我们能把“端到端模型看似 FDE 还行但其实 collapse 到单走廊/平均走廊”的问题量化出来，从而支撑 Claim A/B。

---

## 5) 关键图与可视化交付物（按 `docs/visual_style_guide.md` 执行）

### 5.1 风格与工程规范（必须统一）
- 统一入口：`src/plot_style.py`（source-of-truth 在 `src/visualization/plot_style.py`）
- 配色：Okabe–Ito（`OKABE_ITO`），线宽/字号/figsize 按 `PaperStyle` 与 `FIGSIZE_HALF/FULL`
- 输出：主图 PDF（矢量），预览 PNG 可选
- **禁止** `bbox_inches="tight"`（避免 bbox 抖动导致 LaTeX 子图错位）

### 5.2 Fig 1 / Fig 2 级别“必出图”（支撑 Claim）
**Fig 1：Mode collapse 诊断图（主文级）**  
同一组 $(o,d,t_0)$ 条件下：
- 左：End-to-End baseline 的 $K$ 样本（平均化/直线/走廊混叠）
- 右：CascadeTraj 的 $K$ 样本（多走廊清晰分离）
- 视觉编码建议：
  - GT：黑色粗线（或灰色）
  - 样本：蓝/绿系 + alpha（多条叠加）
  - 可加 waypoint 标记（点/十字）强调“离散承诺”确实在分离模态

**Fig 2：Diversity–Realism Tradeoff（主文级）**  
优先用**散点图**：x=realism（DTW/Fréchet），y=diversity（Jaccard/self-BLEU/coverage），不同模型用不同颜色/形状标记；Pareto 曲线仅用于“同一模型多超参变体”的补充展示。

> 其余 ablation 图（soft prior / hard mask / semantics / census）放 Fig 3/4 或 SI。

---

## 6) 实验矩阵（最小闭环优先；每项都回答一个 Claim）

### E0（必做）数据与评估管线自检
目标：确保每个指标/图都能在小样本上跑通，避免最后两天才发现口径问题。
通过标准（写死交付物）：
- **GT baseline（优先级最高）**：对同一条件下的 GT 多轨迹集合，计算 4.4 的 diversity/coverage 指标，作为后续模型结果的“参照上限/基准”（PI 提醒：没有 GT baseline，后面很难判定是否真的 collapse）。
- **诊断 OD case 选择**：用 GT 数据先筛出至少 3–5 个“肉眼可辨存在多走廊”的 OD case（可复用 `src/evaluation/od_multimodality_gate.py` 的 OD 分桶 + 2-means gate 作为自动筛选器），并将这些 case 固定为 Fig1/2 的诊断集（避免每次换 case 导致结论漂移）。
- **Corridor clustering audit**：对诊断 OD case 输出“GT 走廊聚类可视化审计图”，并记录聚类参数（见 4.4b）。
- **Waypoint audit（若启用 Stage-1 监督）**：把抽取到的 GT waypoints 叠加在 GT 路线上（同一风格规范），人工确认它们确实编码了走廊选择（否则 Stage-1 监督会变成噪声）。
- 能生成 Fig1 的“同条件多样本叠图”；能输出四类指标（intent/realism/feasibility/diversity）。

#### E0（工作站A可执行模板｜产物用于后续 Gate）

> 说明：以下以“窗口级 GT samples.npz”作为输入（必须包含 `start_pos/targets/traj_idx/start_t`，可选 `dest_pos`）。  
> Legacy 深圳可参考 `src/evaluation/dump_gt_windows_npz.py`；Phase D（WorldTrace×Detroit）请先用你现有的导出脚本生成同口径 npz（不要在这里工程化）。

> [!IMPORTANT]
> **WorldTrace 路线（Detroit/Columbus）如果要启用时间条件（hour/dow）或对齐 `open_hours`，务必保证 `start_t` 是 Unix epoch seconds。**
> 我们的 `src/data/worldtrace/dump_route_windows_from_segments.py` 默认会把 `start_t` 写成 segment 内的 window offset（不是时间戳），需要显式加 `--use_epoch_start_t` 才会写入 epoch 秒；否则训练侧 `temporal_mode=auto` 会退化为全 0。

```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export E0_DIR="$EXP_ROOT/E0_gt_baseline"
mkdir -p "$E0_DIR"

# 选择多走廊 OD case + 生成 case_XX/gt_case.npz（用于后续固定窗口对齐）
python -m src.evaluation.route_gt_baseline \
  --samples_npz "<PATH_TO_GT_WINDOWS_SAMPLES_NPZ>" \
  --out_dir "$E0_DIR" \
  --od_bin 8 \
  --min_bucket_n 30 \
  --sep_thr 2.5 \
  --num_cases 5 \
  --save_case_npz \
  > "$E0_DIR/route_gt_baseline.stdout.json"

# JSON-only 摘要（方便贴给 PI review）
python - <<'PY'
import json, os, pathlib
e0_dir = pathlib.Path(os.environ["E0_DIR"])
p = e0_dir / "report.json"
r = json.loads(p.read_text(encoding="utf-8"))
out = {
  "E0_dir": str(e0_dir),
  "N": (r.get("stats") or {}).get("N"),
  "F": (r.get("stats") or {}).get("F"),
  "num_buckets_multimodal": (r.get("stats") or {}).get("num_buckets_multimodal"),
  "selected_cases": [
    {
      "case_id": c.get("case_id"),
      "n_used": c.get("n_used"),
      "gt_jaccard_mean": ((c.get("gt_jaccard_distance") or {}) or {}).get("mean"),
      "gt_corridor_pdf": ((c.get("paths") or {}) or {}).get("gt_corridor_clusters_pdf"),
    } for c in (r.get("selected_cases") or [])
  ],
}
print(json.dumps(out, ensure_ascii=False))
PY
```

### E1（必做）端到端 baseline 诊断（支撑 Claim A）
目标：证明 collapse 真实发生，且是拓扑多模态造成的。
输出（写死判定标准）：
- **视觉标准**：在 Fig1 左半边（K=20），样本无法分离出多于一条清晰走廊，或收敛为接近 O–D 直线的模糊带状区域，即判定出现 collapse。
- **量化标准（相对 GT baseline）**：对诊断 OD case，若
  - `mean pairwise Jaccard` 明显低于 GT baseline（例如 $D_{\text{model}} < 0.5 \cdot D_{\text{GT}}$），且
  - corridor coverage 未覆盖到 GT 的多走廊簇（例如 GT 有 2 簇但生成样本仅命中 1 簇），
  则判定为 mode/corridor collapse。
- 备注：在 E0 先跑 GT baseline 的原因是把阈值锚定到“数据自身的多模态强度”，避免凭空设定绝对阈值。

#### E1/E2（工作站A可执行模板｜同一批 windows 对齐）

> 关键点：所有对比必须在**同一批窗口集合**上进行（`--windows_npz`），否则 diversity/coverage 不可比。  
> 做法：直接用 E0 产物 `case_XX/gt_case.npz` 作为固定窗口集合。

```bash
export RAW_ROOT="$HOME/data/geoexplicit_data"
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export E0_DIR="$EXP_ROOT/E0_gt_baseline"

# 选择一个 case（先肉眼看 case_XX/gt_corridor_clusters.pdf 决定）
export CASE_DIR="$E0_DIR/case_00"

# 训练输入：E0 同口径的 GT windows npz（Detroit）
export GT_WINDOWS_NPZ="<PATH_TO_GT_WINDOWS_SAMPLES_NPZ>"

# --- E1: 端到端 baseline（从 windows npz 直接训练 + 在固定 case 上采样，产物为 samples.npz）---
export E1_TRAIN_DIR="$EXP_ROOT/E1_end2end_diffusion_npz_seed0"
export E1_CASE_DIR="$EXP_ROOT/E1_end2end_diffusion_npz_case00_seed0"

PYTHONUNBUFFERED=1 python -u -m src.training.train_route_e2e_diffusion_npz \
  --train_npz "$GT_WINDOWS_NPZ" \
  --out_dir "$E1_TRAIN_DIR" \
  --hidden_dim 128 \
  --diff_steps 100 \
  --epochs 30 \
  --batch_size 64 \
  --seed 0 |& tee "$E1_TRAIN_DIR/run.log"

python -m src.training.sample_route_e2e_diffusion_npz \
  --checkpoint "$E1_TRAIN_DIR/last.pt" \
  --case_npz "$CASE_DIR/gt_case.npz" \
  --out_dir "$E1_CASE_DIR" \
  --num_samples_per_condition 20 \
  --seed 0 \
  > "$E1_CASE_DIR/sample.stdout.json"

# --- E2: Decision→Execution（oracle skeleton bank + residual diffusion execution）---
# 说明：这是最快能闭环验证 Claim-B 的最小版本：
# - 先从 GT case 构建 waypoint bank（两走廊混合），作为“离散承诺”的采样来源；
# - 再用 waypoint-conditioned residual diffusion 做执行层生成，避免端到端平均化。
export E2_TRAIN_DIR="$EXP_ROOT/E2_exec_diffusion_wp_residual_npz_seed0"
export E2_CASE_DIR="$EXP_ROOT/E2_exec_diffusion_wp_residual_npz_case00_seed0"

PYTHONUNBUFFERED=1 python -u -m src.training.train_route_exec_diffusion_wp_npz \
  --train_npz "$GT_WINDOWS_NPZ" \
  --out_dir "$E2_TRAIN_DIR" \
  --waypoint_mode rdp_dev \
  --num_waypoints 2 \
  --hidden_dim 128 \
  --diff_steps 100 \
  --epochs 30 \
  --batch_size 64 \
  --seed 0 |& tee "$E2_TRAIN_DIR/run.log"

python -m src.training.sample_route_exec_diffusion_wp_npz \
  --checkpoint "$E2_TRAIN_DIR/last.pt" \
  --case_npz "$CASE_DIR/gt_case.npz" \
  --out_dir "$E2_CASE_DIR" \
  --num_samples_per_condition 20 \
  --seed 0 \
  > "$E2_CASE_DIR/sample.stdout.json"

# --- collapse 指标（JSON-only）---
python -m src.evaluation.route_mode_collapse_metrics \
  --gt_report_json "$E0_DIR/report.json" \
  --gt_windows_npz "$CASE_DIR/gt_case.npz" \
  --model_samples_npz "$E1_CASE_DIR/samples.npz" \
  --out_json "$E1_CASE_DIR/baseline_collapse.json" \
  > "$E1_CASE_DIR/baseline_collapse.stdout.json"

python -m src.evaluation.route_mode_collapse_metrics \
  --gt_report_json "$E0_DIR/report.json" \
  --gt_windows_npz "$CASE_DIR/gt_case.npz" \
  --model_samples_npz "$E2_CASE_DIR/samples.npz" \
  --out_json "$E2_CASE_DIR/cascade_collapse.json" \
  > "$E2_CASE_DIR/cascade_collapse.stdout.json"

# --- Fig1 风格可视化（PDF + JSON-only）---
python -m src.evaluation.plot_route_mode_collapse_figure \
  --gt_case_npz "$CASE_DIR/gt_case.npz" \
  --model "End2End=$E1_CASE_DIR/samples.npz" \
  --model "CascadeTraj=$E2_CASE_DIR/samples.npz" \
  --out_pdf "$E1_CASE_DIR/fig_mode_collapse.pdf" \
  --out_json "$E1_CASE_DIR/fig_mode_collapse.json" \
  > "$E1_CASE_DIR/fig_mode_collapse.stdout.json"
```

### E2（必做）CascadeTraj（仅 Stage-1 改动）验证（支撑 Claim B）
目标：只靠“离散承诺”就能让走廊模态分离（避免把功劳归因给执行层）。
输出：
- Fig1 右半边：在相同诊断 OD case 下，多走廊样本可视化“清晰分离”。
- 指标：diversity/coverage 显著提升，并接近 GT baseline（至少不再处于 collapse 区间）；intent/realism 不显著变差。
- 训练信号说明：Stage-1 的 GT skeleton 采用 E0 中审计过的启发式抽取策略（例如 `rdp_dev`），避免 reviewer 质疑“waypoints 从哪来”。

### E3（中）执行层细化（支撑“physical execution”叙事）
目标：在已分离走廊模态的前提下，提升 realism/feasibility（形状/速度纹理/局部几何）。
输出：realism 指标提升，同时 diversity 不回落。

### E4（中）soft prior vs hard mask（支撑 Claim C）
目标：展示 hard mask 的“可行性提升”伴随分布截断/多样性损失；soft prior 更平衡。
输出（包含两种叙事分支，提前预案）：
- 必交付：dilation/buffer 扫描曲线（Feasibility & Diversity vs dilation/buffer），证明 hard mask 的结果对 proxy 质量更敏感，从而支撑 soft prior 的鲁棒性价值。
- 情况 A（预期）：Hard mask 的 feasibility 更高但 diversity/coverage 更低 → 叙事为 “truncation trades diversity for feasibility; soft prior offers a better balance.”
- 情况 B（挑战）：Hard mask 在所有指标上都更好 → 叙事改为 “framework is compatible with hard constraints when reliable; soft priors are preferred under uncertain map quality / for generalization”，并用敏感性审计展示其潜在脆弱性。

### E5（可选）语义通道与 census guidance（不抢主线）
目标：展示可控性/扩展性，而不是把论文变成 population synthesis。
成功标准（写死，不引入新任务）：
- 只报告核心指标（Diversity/Realism/Feasibility/Intent）是否保持或小幅改善，证明“加入 census guidance 不破坏核心能力，并展示可扩展性”。
- **不引入** census-specific 的语义准确性指标（例如“生成职业分布是否匹配 census”），避免论文叙事滑向 population synthesis。

---

## 7) 结果落盘与可复现约定（避免“跑完找不到产物”）

建议统一外置实验根目录（示例）：
- `$RAW_ROOT/experiments/icml2026_routegen/<exp_name>/`

每个实验目录最少包含：
- `config.json`：所有超参/数据口径/seed
- `metrics.json`：四类指标 + 置信区间/多 seed 汇总
- `samples/`：用于 Fig1 的同条件多样本可视化（PDF+PNG）
- `audit/`：feasibility proxy 的敏感性审计结果（不同 dilation/buffer 的曲线/表）

日志习惯（来自 `docs/README.md` 踩坑）：优先 `python -u ... |& tee logs/xxx.log`，不要后台重定向导致“看似没进度”。

---

## 8) 风险与止损线（10 天窗口下必须严格止损）

- **数据/指标先于模型**：48 小时内必须跑通 E0/E1 的 Fig1 左半边，否则先停模型开发。
- **避免 hard support 变成主路线**：hard mask 只作为诊断对照，结果呈现必须带“截断证据”与敏感性审计。
- **多进程/锁风险**：如遇 HDF5/多进程卡死，优先 `HDF5_USE_FILE_LOCKING=FALSE` 或 `--num_workers 0`（见 `docs/README.md`）。

---

## 9) 10 天时间盒（建议节奏；可按实际压缩）

- Day 1–2：E0（数据/评估/作图管线跑通）+ baseline 代码链路确认（至少能启动训练/采样）+ Fig1 样例框架定稿（遵循 `docs/visual_style_guide.md`）
- Day 3–4：E1（端到端 baseline collapse 证据 + diversity 指标）
- Day 5–6：E2（CascadeTraj 的 Stage-1 版本，优先把 Fig1 右半边做“扎实”）
- Day 7–8：E3/E4（执行层提升 realism + soft prior/hard mask 对照 + proxy 敏感性审计）
- Day 9：E5（可选语义/census ablation）+ 主文图表清理
- Day 10：复现实验/重跑关键 seed + 论文整合与查漏补缺
