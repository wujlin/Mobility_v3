# Hierarchical 路线验证协议（硬约束版，避免走偏）

> 目的：把 “trip-level 决策 + segment-level 执行” 的分层路线，变成**可执行、可止损、可归因**的工程流程。  
> 原则：不为赶时间做 trivial 设计；不做无意义烧卡；所有止损线必须可复现（固定统计口径 + CI + 噪声地板）。

---

## 主线结论（可证伪承诺）

- **分层扩散最可能解决的问题**：`p(τ|o,d)` 是多路线模态的混合分布，端到端 score 学到的是“责任加权平均梯度场”→ 低频拓扑模态（detour）被平均掉，表现为 **Destination Gravity / 直线坍缩 + 抖动**。  
- **分层扩散不保证解决的问题**：若“不物理”来自**条件信息缺失（缺地图/可行域/规则）**或 micro 执行本身物理建模不足，单靠分层并不能凭空补齐约束。

因此我们把分层写成一个**可证伪承诺**（止损线）：
- **只有当 Oracle(z) 条件下**（z=waypoints/anchors），micro 能在“不破坏 coarse 拓扑”的前提下显著改善物理纹理指标，分层才是主线；
- 否则优先转向：weak-map/graph 或更强的 micro 物理约束/表示（而不是继续堆层级扩散）。

---

## 主线三步 Go/No-Go（最短实验序列）

> 这三步的设计目标是：**先把积分打开（Oracle z）**，把风险压到最低，再决定是否进入训练。

### Go/No-Go 1：OD 条件下是否存在“稳定多模态混合”？

**目的**：确认直线坍缩是否真的来自“混合分布的梯度平均”。  
**做法**（GT-only，CPU-only）：把相近 OD 做桶（需要 coarse bin 才有足够重复），对桶内 GT 轨迹做粗尺度几何聚类/分裂，检查是否出现稳定的 2+ 簇。

- 脚本：`src/evaluation/od_multimodality_gate.py`
- 推荐做法：扫一遍 `od_bin`（例如 8/12/16），看结论是否稳定；`od_bin` 越小越“严格”（但重复越少）。

```bash
~/miniconda3/envs/emotion/bin/python -m src.evaluation.od_multimodality_gate \
  --samples_npz data/experiments/prior_geo_density_test/samples.npz \
  --od_bin 12 --min_bucket_n 50 --sep_thr 2.0 \
  --out_json data/experiments/od_gate/gt_odbin12.json
```

**判读**（经验口径，避免过拟合阈值）：
- 若在相对严格的 `od_bin`（如 8/12）下，多模态桶占比接近 0：说明“同 OD 多路线混合”不是主要矛盾，分层仍可做但预期收益变小（更可能是 micro/可行域问题）。
- 若在较严格 bin 下仍存在稳定多模态桶：强证据支持“混合导致平均场”解释，分层值得作为主线。

### Go/No-Go 2：Oracle waypoints 的 skeleton-only 能否承载 coarse 拓扑模态？

**目的**：验证 “少量 waypoint 能表达拓扑” 这个压缩瓶颈是否成立。  
**做法**：从每条 GT 提取 2–3 个 waypoint（建议 RDP fixed-K），生成 skeleton，并用 Phase 0 的空间尺度指标评估（不看 MSE）。

1) 先做 **硬可行性**：`Waypoint Gate`（碰撞率/越界率）  
2) 再做 **coarse detour 指标对齐**：`detour_validity`（overall + detour subset）

生成 skeleton-only 的 `samples.npz`（便于统一后续评估）：
```bash
~/miniconda3/envs/emotion/bin/python -m src.evaluation.make_oracle_skeleton \
  --samples_npz data/experiments/prior_geo_density_test/samples.npz \
  --out_npz data/experiments/skeleton_oracle/rdp_k2_linear.npz \
  --waypoint_mode rdp_dev --num_waypoints 2 --skeleton linear
```

对照（0 waypoint 的 straight skeleton）：
```bash
~/miniconda3/envs/emotion/bin/python -m src.evaluation.make_oracle_skeleton \
  --samples_npz data/experiments/prior_geo_density_test/samples.npz \
  --out_npz data/experiments/skeleton_oracle/straight_k0_linear.npz \
  --waypoint_mode time --num_waypoints 0 --skeleton linear
```

评估（空间尺度 turn + max_dev_ratio + len_ratio；含 CI+noise floor）：
```bash
~/miniconda3/envs/emotion/bin/python -m src.evaluation.detour_validity \
  --inputs "StraightK0:data/experiments/skeleton_oracle/straight_k0_linear.npz" \
           "RDPK2:data/experiments/skeleton_oracle/rdp_k2_linear.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 10 --bootstrap 200 --noise_splits 200
```

**Go/No-Go**：
- 若 `RDPK2` 在 detour subset 上显著优于 `StraightK0`，且 `Waypoint Gate` 的碰撞率 < 10%：通过（waypoint 瓶颈成立）。
- 若 skeleton-only 仍对不齐 coarse detour（或碰撞率高）：说明 “少点 waypoint 承载拓扑” 不成立，应改 K/改表征或直接转 weak-map/graph。

### Go/No-Go 3：Oracle waypoints 条件下，micro 是否“可执行且不破坏拓扑”？

**目的**：回答“分层能不能修复不物理”的关键。  
**必须三对照**（同一批 Oracle waypoints / 同一 detour subset）：
1) skeleton-only（不训练）
2) deterministic residual（小网络，先证伪 residual 任务不可学）
3) diffusion residual（主方案）

**硬条件（必须同时满足）**：
- (A) coarse 拓扑不被破坏：final 的 `turn@L / max_dev_ratio / len_ratio` **不差于 skeleton-only**
- (B) 物理纹理显著改善：`speed/accel/DCV`（或更强的 kinematic 耦合指标）显著优于 skeleton-only

若 (A) 失败：优先怀疑 micro 表示/条件注入（或 residual 偷跑 macro）；不要继续堆层级。  
若 (A) 成功但 (B) 失败：说明分层拆拓扑是对的，但 micro 物理建模不足（需要 Frenet residual / band-limit / kinematic 约束等）。

---

## 0) 三条必须写死的硬条款（写进所有计划/README）

1. **多尺度以空间尺度为主**：弧长重采样/插值 + spatial lag（而非 time stride），并做 **multi-offset 聚合**（消除 alias/相位偏置）。
2. **detour 判据至少二元联合**：粗尺度转向（turn@L） + `max deviation`（或 `length ratio`）同时报告，防止“靠乱转角度”骗过 turn 直方图。
3. **Phase 2 必须有三对照**：`skeleton-only` vs `deterministic residual` vs `diffusion residual`，且 residual 需结构性约束防止偷跑 macro（否则失败无法归因）。

---

## 1) 跨 Phase 最容易走偏的 6 个漏洞（以及补丁）

### 漏洞 1：time stride ≠ 空间尺度
- 速度分布不一致时，stride 低通掉的是“时间高频”，不等价于“几何高频”，会误判 detour。
- **补丁**：统一用弧长参数化，再在空间 lag（如 L=1/2/4/8）上算 turn/曲率分布。

### 漏洞 2：stride 的相位偏置（alias + offset bias）
- stride=4 只取 offset=0 会系统性跳过转折点。
- **补丁**：对每个尺度做 **multi-offset 聚合**（offset=0..stride-1 或 ds 的分数偏移）。

### 漏洞 3：TurnAngle 直方图不看序列结构，可被“乱转圈”骗过
- **补丁**：至少加一个几何位移类量：`max lateral deviation`（建议归一化到 chord）或 `path length ratio`。

### 漏洞 4：JSD 口径不锁死（bin/样本量/圆周变量）
- turn-angle 的直方图 bin 边界会显著影响数值；样本少时方差更大。
- **补丁（Phase 0 写死）**：
  - 固定 turn 的 bin（例如 `[0,π]` 等宽），并把 bin edges 写入 JSON；
  - 用 bootstrap 给 95% CI；
  - 计算 `GT vs GT (split-half)` 的 JSD 作为 **噪声地板**。

### 漏洞 5：detour 是低频且低频事件（尾部稀缺），overall 会被淹没
- **补丁**：定义 `detour subset`（只用于评估），所有 Go/No-Go 必须同时报告：
  - overall
  - detour subset

### 漏洞 6：用 KDE/密度当 “feasibility gate” 会把 detour 过滤掉
- KDE 是典型性过滤，尾部 detour 很可能被压死。
- **补丁**：
  - gate 只做硬可行性（速度/加速度界、越界/飞天等），不要把训练密度当可行性；
  - 若必须用密度：降级为 soft penalty，并在 detour subset 上单独报告通过率。

---

## 2) Phase 0：Detour Validity 指标体系（CPU-only）

### 2.1 指标（固定口径）

1) **多尺度转向（空间尺度）**  
给定空间 lag `L`：
\[
\Delta\theta^{(L)} = \left|\mathrm{wrap}\big(\angle(p(s+L)-p(s))-\angle(p(s)-p(s-L))\big)\right|
\]
统计 `Δθ(L)` 的分布，用固定 bin 的 `JSD` 衡量 Pred vs GT。

2) **Max deviation（防作弊）**  
相对 chord（start→end）最大横向偏离，并用 chord 长度归一化：
`max_dev_ratio = max_dev / chord_len`

3) **Length ratio（防回环/乱转）**  
`len_ratio = path_len / chord_len`

> 注意：turn 分布“像”不代表拓扑正确；`max_dev_ratio/len_ratio` 是最便宜的反作弊约束。

### 2.2 统计流程（可止损）
- 固定 bin（turn 使用 `[0,π]` 等宽；标量 bin 由 GT 分位数确定并写入报告）。
- bootstrap 得到 95% CI。
- **噪声地板**：`GT split-half` 的 JSD（同口径）作为 floor。

### 2.3 detour subset（只评估）
- 默认用 GT 的 `max_dev_ratio` 取 top-10%（或按需求调阈值），并写入阈值与子集大小。

### 2.4 工具脚本

- 脚本：`src/evaluation/detour_validity.py`

示例：
```bash
~/miniconda3/envs/emotion/bin/python -m src.evaluation.detour_validity \
  --inputs "Prior:data/experiments/prior_geo_density_test/samples.npz" \
           "Ours(CFG2):data/experiments/phys_cfg2_geo_density_test/samples.npz" \
  --ds 0.5 --lags 1 2 4 8 --offset_fracs 0 0.25 0.5 0.75 \
  --detour_pct 10 --detour_score max_dev_ratio \
  --bootstrap 200 --noise_splits 200 \
  --out_json data/experiments/detour_validity/prior_vs_cfg2.json
```

---

## 3) Phase 1：Waypoints / Skeleton 是否能承载拓扑模态（不看 MSE）

核心判据：在 detour subset 上，`skeleton-only` 是否能对齐：
- `turn@L(粗尺度)` 分布
- `max_dev_ratio` 分布
- `len_ratio` 分布

补强建议（仍然不烧卡）：
- 时间分位点基线改为 **弧长分位点**（避免速度污染）。
- `max_dev` 选点建议用固定-K 的 RDP 两步版（S 型 detour 至少需要 2 个点）。
- 必须包含对照：**0 waypoint（仅 start/end）** 的 skeleton。

同时跑 `Waypoint Gate` 的硬可行性（碰撞率/越界率），避免 coarse 表征本身不可行：  
见 `src/evaluation/waypoint_gate.py`。

---

## 4) Phase 2：Micro-only Executability（最关键止损点）

必须三对照（同一批 Oracle waypoints / 同一 detour subset）：
1) `skeleton-only`（不训练）
2) `deterministic residual`（小网络，先证伪 “residual 任务不可学”）
3) `diffusion residual`（主方案）

结构性补丁（避免 residual 偷跑 macro 或被锁死）：
- residual 建议在 **Frenet frame** 里生成（法向偏移为主），并在 waypoint 处施加边界条件（残差为 0）。
- 或显式去掉 residual 的低频分量（band-pass），保证拓扑只能由 skeleton 承担。

止损线（示例口径）：
- detour subset 上：最终轨迹的 `turn@L(粗尺度)` **不能差于 skeleton-only**（不能被 residual 搞坏）
- 同时：`speed/accel`（或微观纹理指标）应显著优于 skeleton-only（证明 micro 不是摆设）

---

## 5) Phase 3：Macro waypoint 可学性（仅在 Phase 2 通过后）

评估不要只看 best-of-K（容易被 K 大小骗）：
- coverage（best-of-K 命中）
- **detour subset 的概率质量/占比**（detour 模态是否真的被分配到足够概率）

训练侧避免尾部被忽略：
- detour subset oversample / reweight（成本敏感）
- feasibility gate 不要用 KDE 硬过滤尾部

---

## 6) Phase 4：端到端组合（小规模闭环 + 可归因）

每个 case 必须同时产出/保存三条轨迹（同条件）：
1) skeleton-only
2) skeleton + deterministic residual
3) skeleton + diffusion residual

并在同一套 Phase 0 指标（overall + detour subset）下出报表，锁死归因边界：
- waypoint（拓扑决策）
- skeleton（几何生成器）
- micro（纹理/可行性执行）

---

## 7) 最小实验序列（按“最省卡、最强止损”排序）

1) Phase 0：定义 detour subset + 空间尺度 turn + max_dev_ratio/len_ratio + CI + noise floor  
2) Phase 1：Oracle waypoints → skeleton-only 是否保住 coarse detour（含 0 waypoint 对照）  
3) Phase 2：skeleton-only vs deterministic residual vs diffusion residual（三对照）  
4) Phase 3：macro 先只评 skeleton 指标（别急着接 micro），通过后再组合  
5) Phase 4：小 K、小子集闭环（可归因产物齐全）
