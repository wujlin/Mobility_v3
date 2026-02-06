# ICML 2026 RouteGen：Way-CASD 主线实验记录（给新 PI 的快速背景）

> 口径声明：本文只记录**已跑过且在本地 `_sync/wsa/` 下可查到的结果**（主要在 `_sync/wsa/icml2026_routegen/`、`_sync/wsa/pi_verify/`、`_sync/wsa/baselines_sota_s0/`），并把关键实验的**设置/结果/结论**压缩成可审阅版本。  
> 工作站真实落盘与环境约定见：`docs/WORKSTATION_GUIDE.md`。  
> legacy（Map-free / raster / segment）E 系列实验索引见：`docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`。

> ⚠️ 2026-02-01 更新（min_hops=5 的“论文口径”）：我们新增了一条“过滤短路线(min_hops=5)→重训AE→Oracle Decode(beam)→granularity(米)”的结果链路，产物在 `_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/`（以及对应可视化 `_sync/wsa/paper_figures/waycasd_v1/min5_s0/`）。  
> 这条链路 **不在** `_sync/wsa/icml2026_routegen/` 下，因此单独在第 0 节补充，避免 PI 误把旧口径（未过滤短路线 / 不同 ckpt）当作当前主结果。
>
> ⚠️ 2026-02-05 补充（min_hops=5 的 end-to-end generation）：已完成 Flow→z_flow→Decoder 的端到端评测（含 per-route + 形状指标），并形成 D4 空间审计（hit_wall 热点）。产物主要在 `_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/` 与 `_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/`。
>
> ⚠️ 2026-02-06 补充（评测公平性）：已完成 OD-disjoint split（精确 OD：city,start_way,dest_way 不重叠）并重训 RNN-AR / Tr-AR / Way-CASD，输出 paired McNemar 与失败模式对比。产物在 `_sync/wsa/pi_verify/20260206_od_disjoint_s0/`。

---

## 0. 一句话结论（当前状态）

### 0.0 论文口径（min_hops=5；oracle 上界 + generation 现状）

**核心结论**：过滤短路线后，任务更“planning-like”，但 greedy 显著变难；beam=10 仍能带来稳定提升；并且我们已把“way-level success 的地理精度”用 meters 量化清楚（成功中位误差≈65m）。

- **过滤比例（min_hops=5, max_way_len=160）**：keep=5022/7502=66.9%（Detroit 61.2%，Columbus 69.5%）。  
  产物：`_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/filter_routes_min5_max160.json`  
  备注：该 JSON 为旧格式，包含 `p25=-1`（存在无效路线）；新版脚本 `tools/waycasd_filter_routes_stats.py` 已支持输出 valid_keep（后续建议补跑 v2 产物以免口径歧义）。
- **Oracle Decode（GT→Encoder→latent→Decoder，上界；n=200/城，共 400）**：  
  - Greedy：Detroit 0.32，Columbus 0.56，Overall 0.44  
  - Beam=5：Detroit 0.56，Columbus 0.805，Overall 0.6825  
  - Beam=10：Detroit 0.66，Columbus 0.85，Overall 0.755  
  产物：`_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/oracle_decode_*_n200.json`
- **Granularity & 终点误差（meters）**：  
  - way_len 分布：median=74m（p25=26m, p75=196m, p95=638m）  
  - Beam-10 终点误差（pred_last_way_center→dest_pos）：成功 median=65m，p95=662m；失败 median=5.19km，p95=17.6km  
  产物：`_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/metrics/waycasd_eval_granularity_stats.json`  
  论文 snippet：`_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/metrics/waycasd_eval_granularity_paper_snippet.md`
- **可视化（min5 口径）**：  
  - Micro（每城 easy/recovered/hard，带 Err@dest 角标）：`_sync/wsa/paper_figures/waycasd_v1/min5_s0/micro/waycasd_city_micro_case_study.png`  
  - Macro（三列：beam10 failure density / greedy success rate / beam gain）：`_sync/wsa/paper_figures/waycasd_v1/min5_s0/macro/waycasd_city_macro_overview.png`

#### 0.0.1 end-to-end generation（Flow→z_flow→Decoder；min5 口径，n=200/城）

当前“可复现最佳链路”（对齐 `checklist_exp.md`）：
- Flow：`W11_train_flow_past16_regionseq_xattn_s0/ckpt_best.pt`
- Decoder：E2 joint fine-tune 续训 e60：`_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/ckpt_best.pt`
- Decode：beam=10，`decode_max_candidates=0`（全后继），soft anti-loop `P=2.0,K=4`，Region constraint=AR relaxed + dest_region fallback

结果（Overall success_rate，beam）：
- **E2（未续训）**：69.0%（`_sync/wsa/pi_verify/E2_joint_finetune_s0/binned_eval_flow_n200pc.json`）
- **E2 续训 e60（当前主结果）**：**74.5%**（`_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/binned_eval_flow_n200pc.json`）
  - 分桶 success_rate（[5,10)→[60,+)）：`91.7 / 86.6 / 69.0 / 80.8 / 51.9 / 64.7`

失败模式（E2 续训 e60，per-route；n=400）：
- hit_wall=24.5%，loop=21.0%，dead_end=1.0%  
  产物：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/eval/per_route_flow_n200pc.json`

#### 0.0.2 D4：hit_wall 空间审计（[40,60) bin）

目的：定位最严重 bin（[40,60)）的 hit_wall 是否呈现空间聚集，并量化“卡住点”的局部拓扑难度。
- 输出：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/audit/hit_wall_spatial.png`
- 审计 JSON：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/audit/hit_wall_spatial_audit.json`
- 关键事实（hit_wall 子集，last_outdeg）：两城均 `p50=1, p90=2`（并非高分叉路口），更像“早期选错 corridor 后一路走到底”。

#### 0.0.3 Flow 改进实验索引（min5 口径，n=200/城）

> 目的：避免对“+2/+4 条路径”的噪声做过度解读；尽量用可复现的对照 +（可用时）paired 统计来支持结论。

- **B3：Best-of-K（K=8）= 多样本 + 选择**  
  产物：`_sync/wsa/pi_verify/B3_bestofK_s0/`  
  - success_rate（beam）：dest 选择与 oraclebest 选择 **相同**：Overall 82.25%（因为两者都“success-first”）；但在 **成功样本内**，oraclebest 可显著改善 shape（例如 [40,60) DTW p50：21.7km→0.73km，Fréchet p50：1.02km→0.265km，len_ratio p50：1.27→1.006）。  
  - 含义：z_flow 的“可达性/多样性”并不差，瓶颈更可能在 **如何选到好样本**（final_error 对成功样本几乎无区分度）。
- **A2：Flow CFG（cond dropout=0.1 + CFG 推理）**  
  产物：`_sync/wsa/pi_verify/A2_flow_cfg_s0/`  
  - sweep（cfg_scale）：1.0/1.5/2.0/3.0，其中 cfg=2.0 的 overall success≈73.25%（本轮未超过主结果 74.5%）。  
  - 备注：当前只保存 binned JSON，未 dump per-route，因此不做显著性结论；若要对“是否有改进”下判断，建议补跑 paired（输出 per_route）。
- **A1：Flow 更深（n_layers=8）**  
  产物：`_sync/wsa/pi_verify/A1_flow_deeperL8_s0/`  
  - overall success≈73.75%（本轮未超过主结果 74.5%）；同样建议补 per-route 才能做 paired 结论。

### 0.1 历史口径（未过滤短路线 / strict_sem5 旧主线，2026-01）

1. **Way-level 表示是主线**：用 WorldTrace 的 `osm_way_id` 构造 way 序列（p50≈24, p90≈54），相比 node/raster 表示能把序列长度压到文献可比的量级。
2. **Strict 数据过滤后仍可训练**：默认严格 gate 后保留 `N=5353` routes（Detroit=1448, Columbus=3905）。
3. **Decision AE 在 GT latent 下可用，但 greedy 不够**：在“全后继（`decode_max_candidates=0`）+ first 策略”下，GT latent 的 greedy 到达率：Detroit 0.47 / Columbus 0.725；beam=10 提升到 0.795 / 0.945。
4. **Flow 采样端到端仍弱**：在同一口径下，Flow 单样本成功率约 0.12~0.15；多采样 any-success（N=20）约 0.33~0.35（n=100/城）。
5. **On-road 指标必须归一化解读**：Detroit `road_prob` 覆盖率高（coverage@0.5≈0.777），导致 raw on-road 容易虚高；Columbus coverage@0.5≈0.106，容错空间极小。

---

## 1. 名词与评估口径（避免“同名不同义”）

### 1.1 关键对象

- **way 序列**：`[way_1, ..., way_L]`，由 `osm_way_id` 连续去重得到。
- **GT latent / z_enc**：把 GT way 序列送入 AE encoder（WayEncoder + Perceiver）得到的 latent tokens。
- **Flow latent / z_flow**：从 Flow 模型采样得到的 latent tokens（输入只含 condition）。

### 1.2 关键评估开关（本页所有数字都写清这些口径）

- `decode_max_candidates`：
  - `0`：使用**全部 successors**（最宽松口径，用于“能力上限/口径对齐”）。
  - `-1`：使用模型默认 `max_candidates`（常见为 32）。
- `decode_candidate_policy`：
  - `first`：按图 CSR 顺序取候选。
  - `destdist`：按离终点距离排序取候选（有潜在 shortcut 风险，需谨慎解读）。
- `greedy` vs `beam`：beam 能显著降低 dead-end，但仍可能 hit-wall（走满 `max_decode_len`）。

---

## 2. 数据：来源、产物与严格过滤（Strict v1）

### 2.1 数据底座

- WorldTrace 原始包（工作站）：`$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip` + `Meta.zip`
- 每城输入（Way-CASD 主线）：
  - Detroit：`$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet`
  - Columbus：`$RAW_ROOT/worldtrace/columbus_core_v1/segments_with_wayid.parquet`

相关代码入口：
- `src/data/worldtrace/build_detroit_segments.py`（已支持写入 `osm_way_id`）
- `src/data/way_graph/build_way_routes_from_segments_parquet.py`

### 2.2 Way routes 质量审计与 Strict gate（事实）

来自：`_sync/wsa/icml2026_routegen/A_way_routes_quality_rustbelt_v1/report.json`

- labeled routes 总量：`n_routes=7462`（Detroit 2271 / Columbus 5191）
- 长度分布（全量）：`way_seq_len p50=26, p90=57, max=146`
- `valid_transition_ratio`：全局 `min=1.0`（转移都在图上；dead_end_frac=0）
- 默认严格 gate（`max_step_m<=2000, max_loop_ratio<=0.3, max_missing_frac=0.0, min_valid_transition_ratio>=0.9`）：
  - **保留**：`n_routes=5353`（Detroit=1448, Columbus=3905）

Strict 数据集（后续训练/评估都用它）：
- routes：`$RAW_ROOT/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W5_way_routes_strict/way_routes_strict_masklen0.npz`
- graph：`.../W3_way_graph_strict/way_graph.npz`

### 2.3 Way features（含语义采样）与覆盖率

way features（含 `way_semantic` 5 通道）：
- `.../W4_way_features_sem/way_features.npz`
- `way_semantic` 通道：`['road_prob_major','road_prob_minor','road_prob_service','entropy','poi_total']`

覆盖率审计（route-level）结论（来自 strict coverage audit 输出）：
- used ways 缺失率约 `13.09%`，但 **route contamination 仅 `5/7502 = 0.07%`**（几乎不影响训练集可用性）。

相关代码入口：
- `src/data/way_graph/build_way_features_from_osm_pbf.py`
- `src/data/way_graph/audit_way_features_npz.py`
- `tools/audit_strict_route_feature_coverage.py`

---

## 3. 模型：Way-CASD（Decision / Flow / Execution）

### 3.1 Decision AutoEncoder（当前主线：PastCtx + semantic5）

模型文件：
- `src/models/way_casd/way_casd.py`
- `src/models/way_casd/way_encoder.py`
- `src/models/way_casd/way_decoder.py`（包含 PastContextEncoder）

训练入口：
- `src/training/train_way_casd_autoencoder.py`

当前使用的 AE（PastCtx k=8）训练结果（事实）：
来源：`_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/W6_train_ae_pastctx_k8/report.json`

- best epoch（by val loss）：`epoch=53`
- best val loss：`0.1871`
- 对应 val acc：`0.9333`

> 注：val acc 之后可继续升到 ~0.95，但 val loss 上升（过拟合/过置信风险）；我们后续更关心序列级到达率。

### 3.2 z_enc 信息性（Encoder 是否“有用”）

来源：`.../W6_train_ae_pastctx_k8/zenc_informativeness_v2/report.json`（n_routes=100/城，`decode_max_candidates=0`, policy=first）

- true：success `0.58`，jaccard_mean `0.660`
- shuffle：success `0.17`
- zero：success `0.185`
- 分城：
  - Detroit true：`0.47`
  - Columbus true：`0.69`

结论（事实）：**z_enc 携带有效路径信息，且 decoder 确实在用它（true >> shuffle/zero）。**

### 3.2.1 cand_query ablation（候选感知 cross-attention 是否必要）

> 目的：直接回答 PI 的“核心贡献是否有 ablation 证据”。  
> 注意：这是 **旧口径（未做 min_hops=5 过滤）** 的 oracle 上界诊断；若论文主结果采用 min_hops=5，需要重跑同口径的 candq0 模型。

来源（可复现，n=200/城，共 400）：
- candq=0：`_sync/wsa/icml2026_routegen/WAYCASD_AB_candquery_strict_sem5_seed0_e100/WAYCASD_AB_candq0_pastctx_k8_strict_sem5_seed0_e100/W8_diag/zenc_info_n200.json`
- candq=1：`_sync/wsa/icml2026_routegen/WAYCASD_AB_candquery_strict_sem5_seed0_e100/WAYCASD_AB_candq1_pastctx_k8_strict_sem5_seed0_e100/W8_diag/zenc_info_n200.json`

结果（true z_enc，greedy decode）：

| 配置 | Overall success | Detroit | Columbus | Jaccard(mean) |
|---|---:|---:|---:|---:|
| candq=0 | 58.25% | 47.0% | 69.5% | 0.6738 |
| candq=1 | **82.25%** | 79.5% | 85.0% | 0.8672 |

解释：candq=1 允许每个候选用自己的 query 从 `z_enc` 抽取候选相关信息；candq=0 相当于把 `z_enc` 当“全局 bias”，对候选区分能力弱。

### 3.3 Oracle step 诊断（失败机制概览）

来源：`.../W6_train_ae_pastctx_k8/oracle_step_diagnose/report.json`

- 评估规模：`n_eval=400`（n_routes=200/城）
- 口径：`decode_max_candidates=-1`（使用模型默认 max_candidates），policy=first
- 总体：
  - success_rate：`0.5975`
  - success_exact_rate（成功中完全匹配占比）：`0.7448`
  - success_diverged_rate：`0.2552`
- 分叉规模：`first_div_outdeg_gt32_frac=0.0`（首次偏离点不涉及 >32 超高分叉）

结论（事实）：成功里“完美复现”为主，但确实存在一定的偏离后恢复能力。

### 3.4 Beam Search（GT latent 上限）——大样本结果（n=200/城）

来源：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_beam_gt_pastctxfix_seed0_n200pc/bs*.report.json`  
口径：`decode_max_candidates=0`（全后继）+ policy=first

| beam | Detroit success | Columbus success | 备注 |
|---:|---:|---:|---|
| 1（greedy） | 0.47 | 0.725 | dead_end 较高 |
| 3 | 0.67 | 0.855 | dead_end 大幅下降 |
| 5 | 0.74 | 0.89 | dead_end≈0，仍有 hit_wall |
| 10 | 0.795 | 0.945 | dead_end=0，Detroit hit_wall 仍显著 |

结论（事实）：**beam 能显著提升到达率并几乎消灭 dead_end**；剩余主要失败是 hit_wall（走满 max_decode_len 未到 dest），Detroit 更明显。

### 3.5 Flow 多采样 any-success（端到端瓶颈）

来源：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_flow_anysucc_pastctxfix_seed0_N*_n200/report.json`（n_routes=100/城）  
口径：`decode_max_candidates=0`（全后继）+ policy=first + decode=greedy

| N（每条 route 采样数） | Detroit any-success | Columbus any-success | 单样本 success（Det/Col） |
|---:|---:|---:|---:|
| 1 | 0.12 | 0.15 | 0.12 / 0.15 |
| 5 | 0.28 | 0.28 | 0.134 / 0.140 |
| 10 | 0.27 | 0.28 | 0.129 / 0.136 |
| 20 | 0.35 | 0.33 | 0.124 / 0.146 |

结论（事实）：Flow 的单样本成功率基本稳定在 ~0.12–0.15，多采样 any-success 会提升但很快平台化（存在“怎么采都难成功”的 route）。

> Flow 训练曾遇到 “PastCtx ckpt strict load 失败导致无 ckpt_best” 的工程问题；已在工作站修复并产出 `ckpt_best.pt`（`.../WAYCASD_FLOW_v1_pastctx_k8_strict_sem5_seed0_fix2/...`）。

### 3.6 Execution（GPS-level diffusion）

代码：
- `src/models/way_casd/gps_diffusion.py`
- `src/training/train_way_casd_gps_diffusion.py`
- 评估：`src/evaluation/way_casd_exec_eval.py`

On-road prior 的“数据侧核验”（用于解释跨城差异）：
来源：`_sync/wsa/icml2026_routegen/WAYCASD3_xattn_nL64_rustbelt_seed0_e1000/A_{city}_roadprob_audit/report.json`

- Detroit：
  - coverage@0.5：`0.7773`
  - GT onroad@0.5 mean：`0.9594`
- Columbus：
  - coverage@0.5：`0.1059`
  - GT onroad@0.5 mean：`0.8361`（p10≈0.415）

结论（事实）：raw on-road 需要结合 coverage 解读；Detroit coverage 高导致“随机也高”，Columbus coverage 低导致“容错极小”。

---

## 4. Corridor/OD 扫描（数据探索支线，供参考）

### 4.1 Michigan+Ohio multimodal OD 扫描（v2）

来源：`_sync/wsa/icml2026_routegen/A_mm_od_mioh_v2_bin02_sep50/report.json`

- 扫描：`total_files_scanned=2,451,298`
- 过滤后保留：`files_kept_after_filter=139,943`
- unique OD bins：`68,378`
- multimodal OD bins：`594`（`od_bin_deg=0.02`, min_routes_per_od=5）

用途：为后续“多模态走廊评估/训练子集”提供候选 OD 列表（当前主线训练数据仍以 Rust Belt strict routes 为主）。

---

## 5. 复现与查询入口（给新 PI）

- 工作站跑法与目录口径：`docs/WORKSTATION_GUIDE.md`
- legacy E 系列同步索引：`docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`
- 本地结果索引：
  - 旧主线（strict_sem5）：`_sync/wsa/icml2026_routegen/`
  - PI/Partner 验证与论文口径（min5 / flow / region / fairness）：`_sync/wsa/pi_verify/`
  - baseline & sota（unified eval）：`_sync/wsa/baselines_sota_s0/`
- Way-CASD 关键目录（本地同步）：
  - AE+诊断：`_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/`
  - Beam 大样本：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_beam_gt_pastctxfix_seed0_n200pc/`
  - Flow any-success：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_flow_anysucc_pastctxfix_seed0_N*_n200/`
  - min5 generation 主结果（E2续训）：`_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/`
  - D4 空间审计（hit_wall）：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/`
  - OD-disjoint split（重训 + paired）：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/`

---

## 6. 与 PI review 的“进度口径”对齐（建议写给 PI 的版本）

> 目的：避免“PI 讨论的是旧口径/旧 ckpt，而我们已经切到 min_hops=5 新口径”的信息错位。

### 6.1 我们当前 **已经有** 的（可复现证据）

- **cand_query ablation 证据**：candq=0→candq=1 在 oracle 上界下 +24pp（第 3.2.1 节，且有 `_sync` 文件可追溯）。
- **beam 的价值**：在 min_hops=5 的 oracle 口径下，beam=10 相对 greedy Overall +31.5pp（第 0.0 节）。
- **granularity 诚实披露**：已经给出 way_len 分布与终点误差（meters）（第 0.0 节），并在可视化里展示 recovered/hard case。

### 6.2 PI review 关注项：当前完成情况

1) **Flow end-to-end generation（min_hops=5 口径）** ✅ 已补齐  
   - 主结果（E2续训 e60）：`_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/binned_eval_flow_n200pc.json`  
   - per-route（paired/显著性基础设施）：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/eval/per_route_flow_n200pc.json`  
   - hit_wall 空间审计：`_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/audit/hit_wall_spatial.png`

2) **Baseline（Shortest Path / RNN-AR / Tr-AR / GTG / DiffTraj）** ✅ 已跑（但需强调公平性）  
   - 注意：在 “success=到达 dest way” 的定义下，Shortest Path 很可能在过滤后的集合上接近 100% success（因为 GT 已证明可达且长度≤160）。  
   - 因此 baseline 更应与 **路径质量指标** 绑定汇报（例如 Jaccard / DTW / length ratio / final error），否则会造成“成功率被 trivial baseline 统治”的误解。
   - 现有工具：
     - `src/evaluation/shortest_path_baseline.py`：length-weighted Dijkstra（meters），输出 `detour_gt_over_sp` + DTW/Fréchet（按 gt_hops 分桶）。
     - `src/evaluation/way_casd_vs_sp_shape_compare.py`：Way-CASD vs SP 的分桶 shape 对比汇总（输出 json/markdown 表）。
   - 产物（随机 split 的统一评测结果，供参考）：`_sync/wsa/baselines_sota_s0/`  
   - ⚠️ 公平性修复（OD-disjoint 重训 + paired McNemar）：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/`

3) **路径质量指标补全（除 success/final error 外）**  
   - 已有：Jaccard（oracle failures & zenc_info），len_ratio（decision_eval 里已有 best/mean 统计）。  
   - 已补：DTW/Fréchet（meters，`src/evaluation/shape_metrics.py` + binned eval / SP baseline）。  
   - 可选补：Hausdorff（已实现但未系统汇报），以及 “detour over shortest”（SP baseline 已输出 `detour_gt_over_sp`）。

4) **min_hops=5 口径下的 cand_query ablation 是否仍成立**  
   - 现状：已有的 +24pp ablation 是旧口径（未 min5 过滤）。  
   - TODO：若论文主表采用 min_hops=5，应训练一版 candq=0(min5) 并复现对比（否则只能把旧 ablation 放在 appendix/讨论里并标注口径差异）。

### 6.3 评测公平性（OD-disjoint；2026-02-06，seed0）

目的：验证“RNN/Tr 的高成功率是否来自 random split + way_id embedding 的背诵”，并给出 paired 显著性结论。

- split + 审计：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/od_split_min5_max160_seed0.json`、`_sync/wsa/pi_verify/20260206_od_disjoint_s0/od_overlap_audit_min5_max160_seed0.json`
  - train/test **OD overlap=0**（严格 disjoint），route overlap=0  
  - 但 transition coverage 仍高：test transitions in train ≈0.839（提示“背诵风险”会弱化但不会归零）
- 评测样本：test 可用 routes 为 Detroit 132 + Columbus 200 = 332（`n_routes=200/城` 时 Detroit 不足）
- paired 结论（beam）：
  - Way-CASD vs Tr-AR：+11.4pp，p=0.0001（显著）  
    产物：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/paired_tr_vs_waycasd_beam.md`
  - Way-CASD vs RNN-AR：-3.0pp，p=0.2888（不显著，统计上打平）  
    产物：`_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/paired_rnn_vs_waycasd_beam.md`
- 失败模式强对比：RNN 主要 dead_end（≈34%），Way-CASD 主要 hit_wall/loop（hit_wall≈36%）——两者瓶颈不同，后续优化应对准 Way-CASD 的 hit_wall（而不是 dead_end）。
