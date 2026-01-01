# Phase C 结果解读（Macro Hard Support + AR + Micro DetRes）

> **用途**：给 PI/组会的一份“事实链条 + 归因锚 + 下一步决策”的可读版本。  
> **核心原则**：不靠“调参直觉”，只靠可证伪审计与可复现指标。
>
> **重要说明（数据口径）**：本文件记录的是旧数据（深圳出租车 Passenger Trip，dt=30s）上的 Phase C 结果，用于复现与方法论验证；当前主线数据已切换到 WorldTrace×Detroit（1Hz matched），新主线口径见 `docs/DATA_CONTRACT.md` 与 `docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`。

---

## 0. 一句话结论（目前能拍板的）

- **G1（可行性）已通过**：Hard Support + 自回归（AR）把 `WP_ANY` 压到 0，并把 `COLL/CUT` 压到 `6%~8.5%`（K=1，非 best-of-k/事后筛选）。
- **G2（真实性）主要卡在“宏观绕路幅度不足（under-detour）”**：MacroSkeleton 明显更直/更短；DetRes 能把它往 GT 拉近，但无法凭空创造绕路拓扑。
- **ORACLE_PROJ 也会 CUT 的主因不是“直线骨架理论下界”，而是 weak-map `count>=1` mask 孔洞/过严**：轻微 dilation 可把 CUT 从 `~10%` 拉到 `<1%`。

---

## 1. 当前技术路线（主线）

1) **Macro（决策层）**：预测 `wp1 → wp2 → end_anchor`（patch 内像素分布），输出被 `masked softmax` 约束在 strict drivable（`count>=thr`）内。  
2) **Macro → Skeleton（粗执行）**：用直线连点生成 skeleton（供下游 executor）。  
3) **Micro（执行层）**：DetRes（确定性残差 executor）基于 skeleton prior 输出未来速度/轨迹细节。  

> 关键分工：Macro 负责“走哪条走廊/绕不绕”；Micro 负责“怎么顺滑地走”。  
> DiffRes（随机残差扩散）已止损，不进入主线。

---

## 2. G1：可行性（合法性）验证与解读

### 2.1 指标含义（不用符号）

- `WP_ANY`：三个位点（wp1/wp2/end）里是否有任意一个落在 offroad（`count<thr`）上。  
  - 这是“点位合法性”，Hard Support 直接保证它接近 0。
- `CUT_ONLY`：三个点都合法，但“连线段”仍然穿过 offroad 的比例。  
  - 这是“连通性/走廊一致性”，**不是** Hard Support 自动能保证的，需要模型学会“点与点之间要可达”。
- `COLL`：总体碰撞率（这里基本等于 CUT_ONLY，因为 WP_ANY=0）。

### 2.2 结果（detour-hard，N=400，K=1）

- `argmax`：`COLL=0.0600, CUT=0.0600, WP_ANY=0.0000`，seg0/1/2=`0.0425/0.0225/0.0125`
- `multinomial`：`COLL=0.0850, CUT=0.0850, WP_ANY=0.0000`，seg0/1/2=`0.0600/0.0250/0.0175`

### 2.3 这说明 Macro “学会了什么”

- **“在路上”不是学出来的**：`WP_ANY=0` 主要来自 hard support（输出空间被限制）。  
- **“连通性”是学出来的**：如果只是硬约束，`CUT` 不会自然降到 `6%~8.5%`。AR 把“后续点要看着前序点生成”写回模型，才把 CUT 压下来。

---

## 3. G2：真实性（行为像不像真的）验证与解读

### 3.1 detour_validity（多尺度转向 + 两个宏观标量）

你们现在的 G2 由两类量组成：

1) **turn@4/8（多尺度转向强度）**：在“更粗的空间尺度（4m/8m）”上统计轨迹转向角分布，和 GT 做分布距离（JSD）。  
   - 直线 + 抖动：细尺度角度可能大，但粗尺度角度会塌。  
   - 真绕路：粗尺度角度也会显著。
2) **dev/len（绕路幅度与长度）**：  
   - `max_dev_ratio`：相对起终点直线的最大横向偏离（越大越绕）。  
   - `len_ratio`：路径长度/直线距离（越大越绕/越长）。

> 仅看 “JSD 有多大”不够：你必须知道偏差方向（更直还是更绕）。这就是 scalar-direction 审计要补的关键。

### 3.2 scalar-direction 审计（方向性：更直还是更绕）

在 detour-hard（N=400，detour_pct=100）上得到：

- GT 中位数：`max_dev_ratio p50=0.4950`，`len_ratio p50=1.6304`
- MacroSkel：`Δp50 dev=-0.3742`，`Δp50 len=-0.5736`
- Macro+DetRes：`Δp50 dev=-0.2429`，`Δp50 len=-0.3969`

解读（把符号翻译成人话）：
- 负号表示**比 GT 更直、更短**。  
- MacroSkel 在“绕路幅度”和“路径长度”两项上都明显不足（under-detour）。  
- DetRes 能把两项往 GT 拉回一截，但仍然不足（说明 Micro 不能替代 Macro 做绕路决策）。

### 3.3 物理统计（PHY：speed/accel/turn 分布 + DCV）

在 detour-hard（N=400）上：

- MacroSkel → Macro+DetRes：`JSD_turn` 显著下降（更像 GT 的转向统计），`JSD_accel` 显著下降（加速度统计更像 GT）
- 但 `JSD_speed` 可能出现轻微变差（这是一个可观察 trade-off：形状更像、速度分布未必同步更像，需要后续监控）
- `DCV`：`argmax` 基本为 0；`sample1` 有小比例违规（提示“采样随机性 + 骨架尖角”可能导致局部动力学更紧张）

结论：Micro DetRes 在做“局部动力学质感”，但它无法凭空创造“绕路幅度”（这是 Macro 的职责）。

---

## 4. ORACLE_PROJ 也 CUT：根因不是骨架上限，而是 mask 孔洞

### 4.1 观测（N=2000，oracle_cut_cause_audit）

- dilate_0（strict mask）：`CUT≈0.1055`
- dilate_1（轻微膨胀）：`CUT≈0.0065`
- dilate_2：`CUT≈0.0015`
- `resolved_by_dilate_1≈0.9384`

### 4.2 解读

- 这说明绝大多数 “直线连点切墙” 不是因为路网天然弯，而是因为 `count>=1` 的 weak-map 在道路上存在**孔洞/缺采样**。  
- 因此：G1/G2 的 CUT 在当前设定下，包含一部分“数据代理误差”（proxy error），不完全是模型错误。

> 对外口径建议：主报告仍按 strict mask（与训练/评估一致）；附录同时给 dilation 敏感性，证明 CUT 的下界受 weak-map 质量支配。

---

## 5. mask 内分布是否“乱选”？（回答 Hard Support 的核心质疑）

Hard Support 不等于“模型学会道路识别”；但模型**仍然需要在 mask 内学会选点**。

在 detour-hard，AR sample1（N=400，K=1）：

- `JSD_pref(pred vs gt_proj)`：`wp1=0.4796`, `wp2=0.6360`, `end=0.7903`
- random baseline：`~0.91`

解读：
- 模型在 mask 内并非随机（尤其 wp1 明显优于随机）；  
- 但 `end` 的像素级偏好仍弱（接近随机），这会直接拖累 G2 的拓扑/绕路质量。

---

## 6. end “不精”的具体类型（应该怎么修）

end_imprecision_audit（N=400，use_gt_proj=True）：

- 类型占比：`fine≈0.482`，`dist≈0.133`，`corridor≈0.213`，`both≈0.173`

解读：
- “纯距离不对”不是主因（≈13%），更大的问题是 **corridor（选错平行走廊/道路）**（≈21%）以及 both（≈17%）。
- 这类错误通常需要更强的“目的地附近区分能力”（更高分辨率语义/目的地中心视野/拓扑信息），而不只是“朝着目的地走”。

---

## 7. 当前瓶颈与下一步（不烧卡的 Go/No-Go）

### 7.1 当前瓶颈（已定性）

- **瓶颈 1（G2 主矛盾）**：Macro under-detour（更直/更短）。  
- **瓶颈 2（G2 关键子因）**：end 在 mask 内选点偏弱（corridor error 明显）。  
- **瓶颈 3（评估代理误差）**：strict mask 孔洞导致 CUT 被抬高（oracle 也 CUT）。

### 7.2 Go/No-Go 实验（可归因对照集，不用“先后顺序”讲故事）

这 3 个实验不是“谁先谁后”，而是为了把三个潜在根因拆开验证：

1) **训练分布修正**：Macro 训练对 detour-hard 子集加权/重采样  
   - 要回答：under-detour 是不是主要来自训练分布（detour 事件太稀，模型学到“走直线更稳”）？  
   - 判据：`detour_scalar_direction_audit` 的 `Δp50 dev/len` 明显向 0 靠近。  
2) **目的地附近辨别（destination-aware）**：最小条件增强（例如 dest-centered patch 或 per-pixel dest-delta 通道）  
   - 要回答：end 的 corridor error 是否主要来自“目的地附近缺少可辨识信号”而不是语义不足？  
   - 判据：`end_imprecision_audit` 里 `corridor_error_rate` 显著下降（目标 <0.10）。  
3) **语义信息是否带来可证伪增益**：最小语义注入（例如：目的地周边 POI density / 功能区 one-hot）  
   - 要回答：end 的 mask 内分布偏弱、以及 under-detour 是否真的需要“城市语义”才能改善？  
   - 判据：`macro_mask_alignment` 的 end `JSD_pref` 明显下降，并且 G2 的方向性/物理指标同步改善（避免只“看起来像”但不真实）。

---

## 8. 论文定位（必须写清楚的 trade-off）

- **若接受 Hard Support 作为建模假设**：论文应定位为“weak-map 条件下的 trip-level 决策 + micro 执行”。  
- **若要 claim ‘道路识别/城市语义理解’**：必须引入 OSM/遥感/POI 的可学习语义编码（这会是下一阶段工作量级）。

---

## 9. Phase D（新转折点）

本阶段已完成“可归因 baseline”（Hard Support + AR + DetRes）与关键审计；下一阶段转向为：
- **OSM 不再作为 hard support**：把 OSM 道路信息作为**软先验特征**输入模型，同时把 `count_proxy` 与 `osm_proxy` 两套口径都纳入审计输出（避免 mask 黑箱与 proxy 污染）。
- **道路拓扑（corridor selection）**：以距离场/可达性等形式输入，专门打 `corridor_error`。
- **城市语义（POI/功能区/建成环境）**：作为解释“为什么要绕”的信息源，目标是把 under-detour 拉回。
- **Diffusion 只负责多模态**：不再背“落点合法/走廊正确”的锅；多样性建立在单条路线质量已足够的前提上。

详见：`docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`

---

## 10. 一键汇总（不写自然语言，只输出关键数值）

> 目的：把一次实验跑完后的关键信息统一抽出来，避免手工翻 JSON。

```bash
# 例：把一次 ARGMAX 实验的 4 份 JSON 汇总成一段固定格式输出
export DIR=data/experiments/phys_macro_hardsupport_ar_detourhard_macro_hardsupport_ar_p64_thr1_s0_argmax

python -m src.evaluation.summarize_phase_c_table \
  --tag ARGMAX \
  --g1_json    "$DIR/macro_waypoint_gate.json" \
  --align_json "$DIR/mask_alignment.json" \
  --detour_validity_json data/experiments/phys_macro_hardsupport_ar_detourhard_g2_detour_validity.json \
  --phy_json   essay/figures/physical_stats/fig_physical_stats_macro_hs_ar_g2_validity.json
```
