# ICML 2026 RouteGen：Way-CASD 主线实验记录（给新 PI 的快速背景）

> 口径声明：本文只记录**已跑过且在本地 `_sync/wsa/` 下可查到的结果**（主要在 `_sync/wsa/icml2026_routegen/`、`_sync/wsa/pi_verify/`、`_sync/wsa/baselines_sota_s0/`），并把关键实验的**设置/结果/结论**压缩成可审阅版本。  
> 工作站真实落盘与环境约定见：`docs/WORKSTATION_GUIDE.md`。  
> legacy（Map-free / raster / segment）E 系列实验索引见：`docs/archive/legacy_20260225/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`。

> ⚠️ 2026-02-01 更新（min_hops=5 的“论文口径”）：我们新增了一条“过滤短路线(min_hops=5)→重训AE→Oracle Decode(beam)→granularity(米)”的结果链路，产物在 `_sync/wsa/pi_verify/20260201_min5_candq1_past8_len160_s0/`（以及对应可视化 `_sync/wsa/paper_figures/waycasd_v1/min5_s0/`）。  
> 这条链路 **不在** `_sync/wsa/icml2026_routegen/` 下，因此单独在第 0 节补充，避免 PI 误把旧口径（未过滤短路线 / 不同 ckpt）当作当前主结果。
>
> ⚠️ 2026-02-05 补充（min_hops=5 的 end-to-end generation）：已完成 Flow→z_flow→Decoder 的端到端评测（含 per-route + 形状指标），并形成 D4 空间审计（hit_wall 热点）。产物主要在 `_sync/wsa/pi_verify/E2_joint_finetune_s0_cont_e60/` 与 `_sync/wsa/pi_verify/D4_hit_wall_spatial_e2cont_s0/`。
>
> ⚠️ 2026-02-06 补充（评测公平性）：已完成 OD-disjoint split（精确 OD：city,start_way,dest_way 不重叠）并重训 RNN-AR / Tr-AR / Way-CASD，输出 paired McNemar 与失败模式对比。产物在 `_sync/wsa/pi_verify/20260206_od_disjoint_s0/`。
>
> ⚠️ 2026-02-22 口径修正（关键）：已补齐 Porto 的 teacher-forcing 逐步准确率诊断，并确认“93%”与“47%”来自不同口径。  
> - `0.9333` 仅对应 Rustbelt AE 训练验证口径（`_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/W6_train_ae_pastctx_k8/report.json`，best-val-loss epoch 的 `val.acc`）。  
> - Porto 同口径 TF-stepwise（`decode_max_candidates=32`, `K=1`, deterministic, n=5000）结果：  
>   - E2 e100 ckpt：`step_accuracy_overall_mean=0.4692`（`_sync/wsa/pi_verify/20260222_porto_tf_stepwise_accuracy_s0/tf_stepwise_k1_cand32_n5000.json`）  
>   - P1 e20 ckpt：`step_accuracy_overall_mean=0.4581`（`_sync/wsa/pi_verify/20260222_porto_tf_stepwise_e20_probe_s0/tf_stepwise_k1_cand32_n5000_e20.json`）  
> - 结论：当前 Porto 主线讨论单步准确率时，必须使用 ~46-47% 口径，不得引用 Rustbelt 训练验证的 0.9333。

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

#### 补充：Porto TF-stepwise 口径校准（2026-02-22）

目的：修复“单步准确率=93%”的口径误用，给 Porto 主线提供可复现实测基线。

- 评测脚本：`src/evaluation/way_casd_teacher_forcing_coverage.py`
- 核心口径（两次完全一致）：`K=1`、`decode_max_candidates=32`、`decode_candidate_policy=first`、`latent_noise_std=0`、`decode_stochastic=false`、`n_routes=5000`、`split_part=test`
- 结果：
  - e100 ckpt（`_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/ckpt_best.pt`）：
    - `step_accuracy_overall_mean=0.4692`
    - `sample_arrival_rate=0.4466`
    - 产物：`_sync/wsa/pi_verify/20260222_porto_tf_stepwise_accuracy_s0/tf_stepwise_k1_cand32_n5000.json`
  - e20 ckpt（`_sync/wsa/pi_verify/20260213_porto_p0p1_stepemb_s0/P1_stepemb_e20/ckpt_best.pt`）：
    - `step_accuracy_overall_mean=0.4581`
    - `sample_arrival_rate=0.4318`
    - 产物：`_sync/wsa/pi_verify/20260222_porto_tf_stepwise_e20_probe_s0/tf_stepwise_k1_cand32_n5000_e20.json`

结论（事实）：Porto 当前同口径单步准确率约 46-47%，而非 93%。

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

---

## 8. 新数据集：Porto Taxi（WAYCASD0，2026-02-08）

> 背景：Rustbelt（Detroit/Columbus）存在 “GT≈ShortestPath” 的退化风险，且样本量很小（5k）。Porto Taxi 提供更丰富拓扑与更强 OD 多模态，适合验证 latent diversity 与 best-of-K。

**数据根目录（工作站落盘）**：
- `OUT_BASE=/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD0_waydata_porto_seed0`

**Way-CASD 标准产物（W1–W4）**：
- `W2_way_graph/way_graph.npz`：`out_deg p50=4, p90=18`；连通性 `largest_cc=100%`
- `W4_way_routes_labeled/way_routes_labeled.npz`：routes=1,630,527；`way_seq_len p50=24, p90=46`

**诊断与关键结论**（产物同步到仓库，便于 review）：
- 诊断目录：`_sync/wsa/pi_verify/20260208_porto_diagnose/`
- OD corridor 多样性（coarse OD bin，>=5 routes）：抽样 5000 bins，multimodal=99.6%，mean LCS dist=0.742（支持“同一 OD 存在结构性不同走法”）。
- 质量风险：`max_step_m` 重尾（p95≈14.9km），提示存在 teleport edge/异常段；在引用 SP/detour 等指标前，必须先做 strict gate。

**P0 / Blocking（严格过滤 + split）**：
- bad ids（默认阈值）：`_sync/wsa/pi_verify/20260208_porto_diagnose/way_routes_bad.json`
- default gate 保留：`1,350,143 / 1,629,126 = 82.9%`（len∈[3,160]）
- 脚本：`tools/porto/run_porto_strict_gate_and_split.sh`（生成 `W5_way_routes_strict_gate/*` 与 `od_split_min3_max160_seed0.json`）

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
- 对应 val acc：`0.9333`（仅 Rustbelt 训练验证口径）

> 注：val acc 之后可继续升到 ~0.95，但 val loss 上升（过拟合/过置信风险）；我们后续更关心序列级到达率。  
> 另见第 0 节“Porto TF-stepwise 口径校准”小节：Porto 同口径单步准确率约 46-47%，与本节 Rustbelt 训练验证指标不可混用。

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
- legacy E 系列同步索引：`docs/archive/legacy_20260225/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`
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

---

## 7. Porto 主线补录（2026-02-10 至 2026-02-22）

> 说明：本节补录此前未写入正文但已在 `_sync/wsa/pi_verify/` 落盘的实验。  
> 统一口径：除非特别声明，success/hit_wall/loop 为 `binned*.json` 按 `overall.greedy.cells` 的加权均值。

### 7.1 Phase-1/2（早期主线，dev10p）

- `20260210_porto_phase1_s0`
  - Flow（xattn，K1，n=1000）：
    - `succ=0.0230, hit_wall=0.9670, loop=0.9910`
    - 文件：`_sync/wsa/pi_verify/20260210_porto_phase1_s0/E3_flow_xattn_dev10p_s0/eval/binned_flow_xattn_k1_beam10_n1000.json`
  - E2（K1，n=1000）：
    - `succ=0.0860, hit_wall=0.9030, loop=0.9260`
    - 文件：`_sync/wsa/pi_verify/20260210_porto_phase1_s0/E2_joint_ft_dev10p_s0/eval/binned_e2_flow_k1_beam10_n1000.json`
  - E2（K16-dest，n=1000）：
    - `succ=0.3190, hit_wall=0.6780, loop=0.8670`
    - 文件：`_sync/wsa/pi_verify/20260210_porto_phase1_s0/E2_joint_ft_dev10p_s0/eval/binned_e2_flow_k16_dest_greedy_n1000.json`

- `20260211_porto_phase2_s0`
  - E2-fulltrain（带 anti-loop，K1，实际 n=200）：
    - `succ=0.0800, hit_wall=0.8850, loop=0.8900`
    - 文件：`_sync/wsa/pi_verify/20260211_porto_phase2_s0/step2_e2_fulltrain_bs1024_s0/eval/binned_e2full_flow_k1_beam10_antiloop_n1000.json`
  - E2-fulltrain（带 anti-loop，K16-dest，实际 n=200）：
    - `succ=0.2850, hit_wall=0.7100, loop=0.7600`
    - 文件：`_sync/wsa/pi_verify/20260211_porto_phase2_s0/step2_e2_fulltrain_bs1024_s0/eval/binned_e2full_flow_k16_dest_greedy_antiloop_n1000.json`
  - 早期 Phase-C（n=1000，OD 组较少）：
    - `Way-CASD_E2_K16: arrival=0.394, cov=0.1554, div=0.6522, n_od=8`
    - 文件：`_sync/wsa/pi_verify/20260211_porto_phase2_s0/phaseBC_n1000_s0/phaseC/od_coverage_diversity_k16_n1000.json`

### 7.2 n=5000 主线演进（StepEmb 链）

- `20260212_porto_phaseBC_n5000_s0`（旧 E2）
  - `Way-CASD_E2_K16: arrival=0.3834, cov=0.0092, div=0.7699`
  - `Oracle_K1: arrival=0.7064, cov=0.4709, div=0.5754`
  - `RNN_b10: arrival=0.1904, cov=0.0271, div=0.2181`
  - `Transformer_b10: arrival=0.2322, cov=0.0200, div=0.2616`
  - 文件：`_sync/wsa/pi_verify/20260212_porto_phaseBC_n5000_s0/phaseC/od_coverage_diversity_k16_n5000.json`

- `20260213_porto_e2e60_p1_s0`（K4 快速口径）
  - `Way-CASD_E2e60_K4: arrival=0.3390, cov=0.0350, div=0.6350`
  - 文件：`_sync/wsa/pi_verify/20260213_porto_e2e60_p1_s0/P1_phaseC_k4_n5000/od_coverage_diversity_k4_n5000.json`

- `20260213_porto_e2e80_k16_n5000_s0`
  - `Way-CASD_E2e80_K16: arrival=0.4940, cov=0.0400, div=0.6478`
  - 文件：`_sync/wsa/pi_verify/20260213_porto_e2e80_k16_n5000_s0/phaseC_k16_n5000/od_coverage_diversity_k16_n5000.json`

- `20260214_porto_p1_stepemb_cont_e40_s0`
  - 该文件同时记录了 E2e80 / StepEmbE20 / StepEmbE40 三条曲线：
    - `StepEmbE20_K16: arrival=0.6156, cov=0.0850, div=0.6690`
    - `StepEmbE40_K16: arrival=0.6122, cov=0.0738, div=0.6508`
  - 文件：`_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e40_s0/phaseC_k16_n5000/od_coverage_diversity_k16_n5000_stepemb_e40.json`

- `20260214_porto_p1_stepemb_cont_e100_s0`（当前 StepEmb 主线）
  - `Way-CASD_E2e100_K16: arrival=0.6480, cov=0.0592, div=0.6608`
  - 文件：`_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/phaseC_k16_n5000/od_coverage_diversity_k16_n5000.json`

### 7.3 Region 侧链补录

- `20260212_porto_region_ar_res5_s0`
  - Region AR 训练：`best val_acc=0.7866 (epoch=27), val_loss=0.6263`
  - 文件：`_sync/wsa/pi_verify/20260212_porto_region_ar_res5_s0/report.json`
  - Region AR rollout（n=1000）：
    - `reach_dest_rate=1.0, exact_match_rate=0.533, has_backtrack_rate=0.035`
    - 文件：`_sync/wsa/pi_verify/20260212_porto_region_ar_res5_s0/eval_region_ar_n1000.json`

- `20260212_porto_flow_xattn_regionseq_dev10p_s0`
  - Flow 训练：`best_val_loss=0.2031 (epoch=58)`
  - 文件：`_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/report.json`
  - 直解（No-E2）：
    - K1：`succ=0.0140`
    - K16-dest：`succ=0.1840`
    - 文件：`_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/eval/binned_flow_regionseq_k1_beam10_n1000.json`、`_sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/eval/binned_flow_regionseq_k16_dest_greedy_n1000.json`

### 7.4 直接解码 Flow z（No-E2）补录

- 早期 strict-flow（pure AE + strict flow ckpt，非 regionseq）：
  - `20260208/09/10` 的 K16-dest 结果在 `0.237~0.274`（n=1000~2000）
  - 文件：
    - `_sync/wsa/pi_verify/20260208_porto_strict_s0/bestofk_dest/binned_flow_bestof16_dest_beam10_n2000.json`
    - `_sync/wsa/pi_verify/20260209_porto_strict_diag_n1000_s0/d2_flow_k16_dest/binned_flow_bestof16_dest_beam10_n1000.json`
    - `_sync/wsa/pi_verify/20260210_porto_strict_diag_n1000_s0/d3_flow_k16_dest_fast/binned_flow_bestof16_dest_greedy_n1000.json`

- regionseq-flow（pure AE + regionseq flow）quick：
  - `20260219_porto_noe2_antiloop_quick_s0`
  - K1 noAL/AL：`0.0200 -> 0.0320`
  - K8 noAL/AL：`0.0840 -> 0.1320`
  - 文件：`_sync/wsa/pi_verify/20260219_porto_noe2_antiloop_quick_s0/binned_noe2_k*_n1000.json`

### 7.5 RL Dense 两轮补录

- 第一轮（`20260215_porto_rl_dense_from_e100_s0`）
  - K1：`succ=0.2698, hit_wall=0.6862, loop=0.8684`
  - K16-dest：`succ=0.6676, hit_wall=0.3166, loop=0.7990`
  - 文件：`_sync/wsa/pi_verify/20260215_porto_rl_dense_from_e100_s0/eval/binned_rl_dense_k1_beam10_n5000.json`、`_sync/wsa/pi_verify/20260215_porto_rl_dense_from_e100_s0/eval/binned_rl_dense_k16_dest_n5000.json`

- 第二轮（`20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0`）
  - K1：`succ=0.2976`
  - K16-dest：`succ=0.6752`
  - 文件：`_sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/eval/binned_rl_dense_sched_k1_beam10_n5000.json`、`_sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/eval/binned_rl_dense_sched_k16_dest_n5000.json`

### 7.6 SIB / Bypass / C1 / n_latent / cached-z / force_past_k / graphdist / DAgger

- `20260217_porto_sib_n2_e2_chain_s0`
  - N2-AE oracle K1：`succ=0.8160`
  - N2-E2 K1：`succ=0.2372`
  - N2-E2 K16-dest：`succ=0.5746`
  - 文件：`_sync/wsa/pi_verify/20260217_porto_sib_n2_e2_chain_s0/eval/binned_n2_*.json`

- `20260218_porto_bypassdrop_only_reuseflow_s0`
  - quick K1：`0.1220`
  - quick K8-dest：`0.3040`
  - 文件：`_sync/wsa/pi_verify/20260218_porto_bypassdrop_only_reuseflow_s0/B3_eval_quick_n2000/binned_bdrop_e2_*.json`

- `20260218_porto_sib_optionA_clean_fast_s0` 与 `20260218_porto_sib_optionA_flowv2_s0`
  - clean-fast：K8 `0.3045`
  - flowv2：K8 `0.3300`
  - 文件：`_sync/wsa/pi_verify/20260218_porto_sib_optionA_clean_fast_s0/A4_quick_n2000/binned_sib_e2_k8_dest_n2000.json`、`_sync/wsa/pi_verify/20260218_porto_sib_optionA_flowv2_s0/A4_quick_n2000/binned_sib_e2_k8_dest_n2000.json`

- `20260218_porto_c1_scorer_only_s0`
  - zenc T-S：`0.0734`
  - 文件：`_sync/wsa/pi_verify/20260218_porto_c1_scorer_only_s0/eval/zenc_info_c1_n5000.json`

- `20260219_porto_d1_nlatent8_s0`
  - AE zenc T-S：`0.6212`
  - E2 zenc T-S：`0.0864`
  - E2 quick：K1 `0.1215`，K8 `0.3235`
  - 文件：`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/D1b_ae_zenc/zenc_info_ae_nL8_n5000.json`、`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/D4a_e2_zenc/zenc_info_e2_nL8_n5000.json`、`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/D4b_eval/binned_e2_nL8_*.json`

- `20260219_porto_e2_cachedz_baseline64_s0`
  - zenc：`true=0.1284, shuffle=0.0534, T-S=0.0750`
  - K1：`0.1045`；K16-dest：`0.3466`
  - 文件：`_sync/wsa/pi_verify/20260219_porto_e2_cachedz_baseline64_s0/eval/zenc_info_e2_cachedz_n5000.json`、`_sync/wsa/pi_verify/20260219_porto_e2_cachedz_baseline64_s0/eval/binned_e2_cachedz_*.json`

- `20260219_porto_forcepk16_from_e40_e60_s0`
  - K16-dest_efficient：`succ=0.6130`
  - K16-best：`succ=0.6130`
  - 文件：`_sync/wsa/pi_verify/20260219_porto_forcepk16_from_e40_e60_s0/eval/binned_e2_forcepk16_k16_*.json`

- `20260220_porto_dagger_sp_p1p0_from_e40_s0`
  - K8-dest_efficient：`succ=0.3668, hit_wall=0.6286, loop=0.9316`
  - success-only：`len_ratio p50=2.433, loop_rate=0.815`
  - 文件：`_sync/wsa/pi_verify/20260220_porto_dagger_sp_p1p0_from_e40_s0/e20_bs256_cachetbl/eval/binned_dagger_k8_dest_efficient_n5000.json`、`_sync/wsa/pi_verify/20260220_porto_dagger_sp_p1p0_from_e40_s0/e20_bs256_cachetbl/eval/per_route_dagger_k8_dest_efficient_n5000.json`

- `20260221_porto_p0_graphdist_from_e40_s0`
  - K1：`0.2125`
  - K8-dest_efficient：`0.4994`
  - success-only（K8）：`len_ratio p50=2.073`
  - 文件：`_sync/wsa/pi_verify/20260221_porto_p0_graphdist_from_e40_s0/eval/binned_p0_graphdist_*.json`、`_sync/wsa/pi_verify/20260221_porto_p0_graphdist_from_e40_s0/eval/per_route_p0_graphdist_k8_dest_efficient_n5000.json`

### 7.7 质量瓶颈与选择策略补录

- `20260219_porto_quality_bottleneck_k16_n5000_s0`
  - 三种 sample_select（K16）成功率相同：`0.6480`
  - success-only len_ratio p50：
    - `dest: 3.143`
    - `dest_efficient: 1.887`
    - `best: 1.938`
  - success-only loop_rate：
    - `dest: 0.704`
    - `dest_efficient: 0.552`
    - `best: 0.565`
  - 文件：`_sync/wsa/pi_verify/20260219_porto_quality_bottleneck_k16_n5000_s0/per_route_e2e100_k16_*.json`

### 7.8 Baseline 质量补录

- `20260221_porto_baseline_quality_n5000_s0`
  - 当前目录内文件包含两种候选口径（`decode_max_candidates=32` 与 `-1`），不要混算。
  - 已落盘口径（`cand=32`）：
    - RNN b10：`succ=0.0752`
    - Transformer b10：`succ=0.0778`
  - 文件：`_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/binned_rnn_beam10_n5000.json`、`_sync/wsa/pi_verify/20260221_porto_baseline_quality_n5000_s0/binned_transformer_beam10_n5000.json`

### 7.9 P1 vs RL 对比补录（修复版）

- `20260221_porto_p1_vs_rl_od_n5000_s0`
  - OD 级覆盖/多样性：
    - P1_K16：`arrival=0.6482, cov=0.0914, div=0.5384`
    - RL_K16：`arrival=0.6786, cov=0.0918, div=0.5103`
  - success-only 质量（修复版）：
    - P1：`len_ratio p50=1.934`
    - RL：`len_ratio p50=1.819`
  - 文件：`_sync/wsa/pi_verify/20260221_porto_p1_vs_rl_od_n5000_s0/od_coverage_diversity_k16_p1_vs_rl_vs_ar_n5000.json`、`_sync/wsa/pi_verify/20260221_porto_p1_vs_rl_od_n5000_s0/success_only_quality_summary_p1_rl_ar_fixed.json`

### 7.10 诊断链补录（coverage / fallback / corridor-zsim）

- `20260221_porto_diag_k4_antiloop_s0`
  - fallback 率：`3260/5000 = 0.6520`
  - Leaflet 诊断页：`loop_cases_city0_k4_antiloop.html`
  - 文件：`_sync/wsa/pi_verify/20260221_porto_diag_k4_antiloop_s0/fallback_k4_dest_efficient_antiloop_n5000.json`

- `20260221_porto_tf_coverage_probe_s0`
  - TF 覆盖探针（K16）：`arrival=0.4348, coverage_mean=0.2079, diversity_mean=0.2954`
  - 文件：`_sync/wsa/pi_verify/20260221_porto_tf_coverage_probe_s0/tf_coverage_summary_n5000_k16.json`

- `20260222_porto_ae_corridor_zsim_s0`
  - 同 OD 与跨 OD 的 z 相似度：
    - `within_od_cos mean=0.8318`
    - `cross_od_cos mean=0.6602`
    - `delta=+0.1716`
  - 文件：`_sync/wsa/pi_verify/20260222_porto_ae_corridor_zsim_s0/ae_corridor_zsim_n5000.json`

### 7.11 遗漏目录索引（已检出未展开）

> 说明：以下目录已检出存在 `report.json` / `binned*.json` / `zenc_info*.json` 等产物，但尚未在主线章节展开。  
> 本节给出“每目录 1 行摘要”，用于快速定位与后续补录。

#### 7.11.1 早期诊断/消融（Detroit 或跨城）

- `20260202_region_constraint_diagnose_s0`：`binned_regionAR_relaxed_destreg_n200pc.json`（succ=0.4425, hw=0.2900, loop=0.3925, len=2.6939）；路径：`_sync/wsa/pi_verify/20260202_region_constraint_diagnose_s0/`
- `20260203_flow_e2e_relaxed_s0`：`binned_flow_regionAR_relaxed_destreg_n200pc_s0.json`（succ=0.1200, hw=0.4675, loop=0.5875, len=4.3063）；路径：`_sync/wsa/pi_verify/20260203_flow_e2e_relaxed_s0/`
- `20260203_flow_experiments_s0`：`binned_flow_regionseq_add_relaxed_destreg_n200pc_s0.json`（succ=0.2200, hw=0.4175, loop=0.5400, len=3.9300）；路径：`_sync/wsa/pi_verify/20260203_flow_experiments_s0/`
- `20260203_flow_micro_seqdump_s0`：`binned_flow_add_seqdump_s0.json`（succ=0.2250, hw=0.4175, loop=0.5450, len=4.3113）；路径：`_sync/wsa/pi_verify/20260203_flow_micro_seqdump_s0/`
- `20260204_E2_stepemb_s0`：`binned_flow_xattn.json`（succ=0.1975, hw=0.2950, loop=0.5725, len=4.2600）；路径：`_sync/wsa/pi_verify/20260204_E2_stepemb_s0/`
- `20260204_E3_regions_res2p0_s0`：`binned_oracle.json`（succ=0.4425, hw=0.3100, loop=0.3975, len=2.8068）；路径：`_sync/wsa/pi_verify/20260204_E3_regions_res2p0_s0/`
- `20260204_E5_pastk16_s0`：`binned_E5_pastk16_oracle.json`（succ=0.7000, hw=0.1050, loop=0.2025, len=1.8629）；`flow_retrain_v2/report.json`（best_epoch=59, best=1.2833）；路径：`_sync/wsa/pi_verify/20260204_E5_pastk16_s0/`
- `20260204_E6_antiloop_sweep_s0`：`binned_E6a_softP2p0_K8.json`（succ=0.2325, hw=0.2750, loop=0.4600, len=3.4878）；路径：`_sync/wsa/pi_verify/20260204_E6_antiloop_sweep_s0/`
- `20260204_E7_decoder_rl_flow_s0`：`binned_E7_rl_flow_arconstraint.json`（succ=0.1825, hw=0.2525, loop=0.4525, len=3.8119）；`report.json`（best_epoch=5, best_score=0.0906）；路径：`_sync/wsa/pi_verify/20260204_E7_decoder_rl_flow_s0/`
- `20260204_E8a_multiscale_ae_s0`：`binned_E8a_oracle.json`（succ=0.4475, hw=0.1275, loop=0.3525, len=2.2356）；`report.json`（best_epoch=59, best=0.1763）；路径：`_sync/wsa/pi_verify/20260204_E8a_multiscale_ae_s0/`
- `20260204_checklist_exp_s0`：`binned_flow_maxcand0.json`（succ=0.1825, hw=0.3075, loop=0.4800, len=3.9489）；路径：`_sync/wsa/pi_verify/20260204_checklist_exp_s0/`
- `20260204_flow_antiloop_ablation_s0`：`binned_E1_hardK4.json`（succ=0.2300, hw=0.2475, loop=0.4225, len=3.2238）；路径：`_sync/wsa/pi_verify/20260204_flow_antiloop_ablation_s0/`
- `D4_hit_wall_spatial_s0`：`binned_eval_flow_n200pc.json`（succ=0.2125, hw=0.2975, loop=0.4875, len=4.0100）；路径：`_sync/wsa/pi_verify/D4_hit_wall_spatial_s0/`
- `EXPBIAS_baseline_flow_s0`：`binned_eval_flow_n200pc.json`（succ=0.3675, hw=0.1900, loop=0.3375, len=3.0465）；路径：`_sync/wsa/pi_verify/EXPBIAS_baseline_flow_s0/`
- `SS_p0p3_s0`：`report.json`（best_epoch=2, best_val_loss=0.1905）；路径：`_sync/wsa/pi_verify/SS_p0p3_s0/`
- `SS_p0p5_s0`：`binned_eval_flow_n200pc.json`（succ=0.3450, hw=0.1875, loop=0.3400, len=3.1099）；`report.json`（best_epoch=1, best=0.1898）；路径：`_sync/wsa/pi_verify/SS_p0p5_s0/`
- `VF_beta0p5_s0`：`binned_eval_flow_n200pc.json`（succ=0.3725, hw=0.1900, loop=0.3400, len=2.9256）；路径：`_sync/wsa/pi_verify/VF_beta0p5_s0/`
- `VF_beta1p0_s0`：`binned_eval_flow_n200pc.json`（succ=0.3650, hw=0.1650, loop=0.3300, len=2.9363）；路径：`_sync/wsa/pi_verify/VF_beta1p0_s0/`
- `VF_beta2p0_s0`：`binned_eval_flow_n200pc.json`（succ=0.3625, hw=0.1750, loop=0.3375, len=3.0234）；路径：`_sync/wsa/pi_verify/VF_beta2p0_s0/`
- `VF_from_flow_beam_s0`：`report.json`（best_epoch=1, best=0.5249）；路径：`_sync/wsa/pi_verify/VF_from_flow_beam_s0/`

#### 7.11.2 Porto 侧链（已落盘但未展开）

- `20260208_porto_strict_baseline`：存在 `report.json`（无可用 best 指标字段）；路径：`_sync/wsa/pi_verify/20260208_porto_strict_baseline/`
- `20260210_porto_phase0_s0`：`binned_flow_k1_rescale_p50_greedy_n1000.json`（succ=0.0130, hw=0.9780, loop=0.9910, len=9.0100）；路径：`_sync/wsa/pi_verify/20260210_porto_phase0_s0/`
- `20260210_porto_strict_diag_n1000_s0_v2`：`binned_flow_bestof16_dest_beam10_n1000.json`（succ=0.2150, hw=0.7700, loop=0.9290, len=6.2127）；路径：`_sync/wsa/pi_verify/20260210_porto_strict_diag_n1000_s0_v2/`
- `20260212_porto_e2_joint_regionseq_s0`：`binned_e2_flow_k16_dest_greedy_n1000.json`（succ=0.4420, hw=0.5310, loop=0.8180, len=6.6894）；`report.json`（best_epoch=20, best=1.0863）；路径：`_sync/wsa/pi_verify/20260212_porto_e2_joint_regionseq_s0/`
- `20260212_porto_followup_p0p1p2_s0`：`binned_e2cont_k16_dest_greedy_n1000.json`（succ=0.4730, hw=0.5110, loop=0.8010, len=6.6818）；`P2_e2_cont_e40/report.json`（best_epoch=40, best=0.9980）；路径：`_sync/wsa/pi_verify/20260212_porto_followup_p0p1p2_s0/`
- `20260214_porto_flow_gap_diag_k16_n5000_s0`：`binned_flow_k16_best_n5000.json`（succ=0.6122, hw=0.2890, loop=0.6930, len=4.2530）；路径：`_sync/wsa/pi_verify/20260214_porto_flow_gap_diag_k16_n5000_s0/`
- `20260214_porto_flow_gap_diag_v2_k16_n5000_s0`：`binned_flow_k16_dest_n5000.json`（succ=0.6122, hw=0.3744, loop=0.7956, len=5.6391）；路径：`_sync/wsa/pi_verify/20260214_porto_flow_gap_diag_v2_k16_n5000_s0/`
- `20260214_porto_rl_from_e100_lenratio_s0`：`binned_rl_k16_dest_n1000.json`（succ=0.3510, hw=0.1080, loop=0.4240, len=3.0406）；`report.json`（best_epoch=5, best_score=-0.5236）；路径：`_sync/wsa/pi_verify/20260214_porto_rl_from_e100_lenratio_s0/`
- `20260217_porto_zenc_informativeness_batched_n5000_s0`：`zenc_info_baseline_ae_n5000.json`（T-S=0.6228）；路径：`_sync/wsa/pi_verify/20260217_porto_zenc_informativeness_batched_n5000_s0/`

### 7.12 Reconstruction-Guided Flow（Porto，2026-02-22）

- 目录：`_sync/wsa/pi_verify/20260222_porto_flow_recon_guided_from_base_s0/`
- 代码路径：
  - 训练实现：`src/training/train_way_casd_flow.py`（新增 `--recon_guided`、`recon_lambda_*`、one-step `z_hat1` 重建分支）
  - 对齐诊断：`tools/flow_z_alignment_probe.py`

#### 7.12.1 训练记录（resume from flow best）

- 训练目录：`_sync/wsa/pi_verify/20260222_porto_flow_recon_guided_from_base_s0/train_recon_guided/`
- 训练区间：`epoch 59 -> 70`（12 epochs，`lambda_recon=0.1`）
- `report.json`：`best_val_loss=0.203079 @ epoch=58`（best 来自 resume 起点）
- `history.jsonl` 与 `report.json` 中后段趋势：
  - `train_flow_loss: 0.3060 -> 0.1971`
  - `val_flow_loss: 0.2285 -> 0.1802`
  - `train_recon_loss: 0.7924 -> 0.7050`
  - `val_recon_loss: 0.7539 -> 0.7044`
  - `train_recon_acc: 0.7387 -> 0.7632`
  - `val_recon_acc: 0.7487 -> 0.7629`

#### 7.12.2 Flow z 对齐诊断（训练前后）

- 训练前：`flow_z_alignment_before_n5000.json`
  - `cos_mean=0.4268, cos_p50=0.4210, l2_mean=357.30, l2_per_dim_mean=2.7914`
- 训练后：`flow_z_alignment_after_n5000.json`
  - `cos_mean=0.4176, cos_p50=0.4112, l2_mean=367.67, l2_per_dim_mean=2.8724`
- 变化量（after - before）：
  - `cos_mean=-0.0091`
  - `cos_p50=-0.0098`
  - `l2_mean=+10.37`
  - `l2_per_dim_mean=+0.0810`

#### 7.12.3 生成评估（K=8，dest_efficient，n=5000）

- 评估文件：
  - `eval/binned_reconflow_k8_dest_efficient_n5000.json`
  - `eval/per_route_reconflow_k8_dest_efficient_n5000.json`
- 口径（cfg）：
  - `latent_source=flow`
  - `n_samples_per_route=8`
  - `sample_select=dest_efficient`
  - `split_part=test`
  - `decode_max_candidates=0, decode_candidate_policy=first`
  - `anti_loop_penalty=2.0, anti_loop_penalty_k=4`
- 全局结果（按 per_route 聚合）：
  - `success_rate=0.1738`（869/5000）
  - `hit_wall_rate=0.8096`
  - `loop_rate=0.9152`
  - `sample_select_fallback_rate=0.8262`
  - `success_only_len_ratio: mean=1.3492, p50=1.0341, p75=1.5621, p95=3.3157`

- 分 bin SR（来自 binned）：
  - `[5,10)=0.2422`
  - `[10,20)=0.1736`
  - `[20,30)=0.1251`
  - `[30,40)=0.1238`
  - `[40,60)=0.2972`
  - `[60,+)=0.2552`
### 7.13 n_latent=8 降维实验（Porto，2026-02-19 ~ 2026-02-23）

> **核心假说**：n_latent=64 → 16384 维 z 空间对 108K 训练样本而言过度参数化（6.6 samples/dim），
> 导致 Flow Matching 无法学到有意义的 z 分布。n_latent=8 → 2048 维（53 samples/dim）应大幅改善 Flow z 质量。

- 目录：`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/`

#### 7.13.1 已完成阶段

| 阶段 | 目录 | 核心指标 | 对比 n_latent=64 |
|---|---|---|---|
| D1: AE | `D1_ae_nL8/` | val_acc=**0.870**, val_loss=0.360, best_epoch=100 | 0.884 → **-1.4pp**，z 表达能力几乎无损 |
| D1b: AE zenc 探针 | `D1b_ae_zenc/` | true=**63.4%**, shuffle=1.2%, T-S=**+62.1pp** | +62.3pp → **持平**，z 信息完整保留 |
| D2: Flow | `D2_flow_nL8/` | val_loss=**0.129**, best_epoch=53 | 0.203 → **-36.5%**，Flow 学到更好的分布 |
| D3: E2 | `D3_e2_nL8/` | val_loss=0.938, best_epoch=30 (=max), lr=1e-5 | 0.925 → 无改善 |
| D4a: E2 zenc 探针 | `D4a_e2_zenc/` | true=**14.7%**, shuffle=6.0%, T-S=**+8.6pp** | +10pp → **posterior collapse 依旧** |
| D4b: E2 eval | `D4b_eval/` | K1-beam=19.1%, K8-dest=32.4%, K16=缺失 | K16=64.8% → 不可比（K 不同，无 step_emb） |

**D3 E2 配置差异**（与 n_latent=64 E2 不同）：
- `decoder_use_step_emb=false`（n_latent=64 E2 为 true）
- 只训练 30 epochs（n_latent=64 E2 为 100 epochs）
- `force_decoder_past_k=0`（不覆盖，沿用 AE 的 past_k=16）

#### 7.13.2 D5 补充实验结果（2026-02-23）

- 目录：`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/D5_pureAE_flowz/`
- 口径：n_routes=5000, test split, seed=0, sample_select=dest_efficient, decode_max_candidates=0

**D5a: Flow z 对齐诊断**

| 度量 | n_latent=64 (16384d) | n_latent=8 (2048d) | 变化 |
|---|---|---|---|
| cos_mean | **0.427** | **0.334** | **-0.093 ↓** |
| cos_p50 | 0.421 | 0.330 | -0.091 ↓ |
| cos_p90 | — | 0.440 | — |
| l2_mean | 357.3 | 59.1 | -83% ↓ |
| l2_per_dim (RMSE) | **2.79** | **1.31** | **-53% ↓** |

> **发现**：cosine 下降（0.427→0.334），但 l2_per_dim 下降 53%。这揭示了 cosine 作为跨维度对比指标的局限性——低维空间中 z 向量范数更小，相同绝对误差造成更大角度偏差。l2_per_dim 是更准确的跨维度对比：每个 latent 元素上 Flow z 距 GT z 的距离显著缩小。

**D5b: Pure AE + Flow z 生成评估**

| 配置 | K | SR | len_ratio p50 | loop_rate | hit_wall | fallback_rate |
|---|---|---|---|---|---|---|
| nL64 Pure AE + Flow z | 8 | **8.4%** | — | — | — | — |
| **nL8 Pure AE + Flow z** | **8** | **19.6%** | **0.977** | 89.8% | 78.5% | 80.3% |
| **nL8 Pure AE + Flow z** | **16** | **27.2%** | **0.957** | 86.7% | 71.1% | 74.6% |

分 bin SR（K=16）：

| Bin | n | SR | success_only loop | success_only len_ratio p50 | success_only jaccard |
|---|---|---|---|---|---|
| [5,10) | 289 | 37.7% | 32.1% | 0.887 | 0.350 |
| [10,20) | 1573 | 28.9% | 51.8% | 0.943 | 0.276 |
| [20,30) | 1351 | 21.0% | 67.6% | 1.025 | 0.229 |
| [30,40) | 969 | 22.0% | 59.2% | 1.046 | 0.242 |
| [40,60) | 673 | 36.3% | 39.3% | 0.947 | 0.264 |
| [60,+) | 145 | 37.9% | 45.5% | 0.788 | 0.189 |

**D5c: E2 + Flow z K=16（补全缺失）**

| 配置 | K | SR | len_ratio p50 | loop_rate | hit_wall |
|---|---|---|---|---|---|
| nL8 E2 + Flow z | 16 | **37.9%** | **1.473** | 78.8% | 61.5% |
| nL64 E2 + Flow z（主线） | 16 | **64.8%** | **1.887** | ~55% | — |

分 bin SR（E2 K=16）：[5,10)=37.4%, [10,20)=38.9%, [20,30)=34.1%, [30,40)=35.3%, [40,60)=45.9%, [60,+)=43.5%

#### 7.13.3 关键发现

**发现一：cos 下降但 SR 上升 — 解码器鲁棒性才是关键杠杆**

cos(flow_z, gt_z) 从 0.427 降至 0.334，但 Pure AE+Flow z SR 从 8.4% 升至 19.6%（**2.3倍**）。原因：低维 z 空间让解码器对 z 扰动更鲁棒。per-element L2 下降 53%（2.79→1.31），在更紧凑的空间中，给定绝对误差覆盖的"邻域"更大，更容易落入有效走廊的吸引域。

**发现二：Pure AE 路径质量远优于 E2**

nL8 Pure AE K16 的 success-only len_ratio p50 = **0.957**（接近完美），而 nL64 E2 K16 为 **1.887**（膨胀近 2 倍）。成功的路线几乎都是 GT 级别的高效路径。E2 靠 K=16 暴力搜索 + 忽略 z 来提升 SR，代价是路径质量严重退化。

**发现三：E2 posterior collapse 与维度无关**

nL8 E2 T-S=+8.6pp（vs nL64 +10pp）— collapse 在任何维度下都发生。这是 E2 joint finetune 的结构性问题。

**发现四：长路线 SR 呈 U 型**

nL8 Pure AE K16 按 bin：短程 [5,10)=37.7% → 中程 [20,30)=21.0% → 长程 [60,+)=37.9%。长路线受益于 past_context（k=16）的自纠正能力，中程路线（20-40 hops）最困难。

#### 7.13.4 全景对比表

| 配置 | K | SR | len_ratio p50 | T-S | 备注 |
|---|---|---|---|---|---|
| nL64 Pure AE + GT z | 1 | 63.7% | ~1.0 | +62pp | AE oracle 天花板 |
| nL8 Pure AE + GT z | 1 | 63.4% | ~1.0 | +62pp | 降维几乎无损 |
| nL64 Pure AE + Flow z | 8 | 8.4% | — | — | 灾难性 |
| **nL8 Pure AE + Flow z** | **8** | **19.6%** | **0.977** | — | **2.3x 提升** |
| **nL8 Pure AE + Flow z** | **16** | **27.2%** | **0.957** | — | **质量最优** |
| nL8 E2 + Flow z | 16 | 37.9% | 1.473 | +8.6pp | E2 仍 collapse |
| nL64 E2 + Flow z（主线） | 16 | 64.8% | 1.887 | +10pp | SR 最高但质量差 |
| Corridor diverse oracle | 16 | 75.9% | — | — | 天花板 |

#### 7.13.5 遗留缺口

| 实验 | 目的 | 优先级 |
|---|---|---|
| nL8 Pure AE K16 OD 覆盖/多样性 | 确认 coverage 是否优于 nL64 E2 的 9.14% | **P0** |
| nL8 latent noise augmentation AE | 在 AE 训练阶段注入噪声提升 decoder 鲁棒性 | **P1** |
| n_latent=4 完整管线 | 验证降维趋势是否继续 | **P2** |

#### 7.13.6 D5 复跑口径锁定（2026-02-23）

为避免 D5 再次出现“参数默认值漂移/beam 分支混入”的口径问题，本轮使用统一脚本复跑并锁定配置：

- 脚本：`run_nL8_pureAE_flowz_probe.sh`
- 输出目录：`_sync/wsa/pi_verify/20260219_porto_d1_nlatent8_s0/D5_pureAE_flowz/`
- 关键运行参数（来自输出 JSON `cfg` 与脚本）：
  - `n_routes=5000`, `split_part=test`, `seed=0`
  - `sample_select=dest_efficient`
  - `decode_max_candidates=0`, `decode_candidate_policy=first`
  - `compare_beam=false`（显式 `--no_compare_beam`）
  - `eval_batch_size`: K8=256, K16=192
  - alignment probe：`batch_size=512`, `num_workers=16`

本轮复跑后，D5 关键文件均已齐备：
- `flow_z_alignment_nL8_n5000.json`
- `binned_pureAE_nL8_k8_dest_efficient_n5000.json`
- `binned_pureAE_nL8_k16_dest_efficient_n5000.json`
- `binned_e2_nL8_k16_dest_efficient_n5000.json`
- 对应 `per_route_*.json` 与 `run_d5*.log`

按 `per_route` 的 route-level 聚合统计（同口径）：

| 配置 | SR | hit_wall | loop | fallback_rate | success-only len_ratio p50 | success-only len_ratio p95 | success-only jaccard mean |
|---|---|---|---|---|---|---|---|
| nL8 Pure AE + Flow z (K8) | 19.56% | 78.48% | 89.76% | 80.44% | 0.977 | 2.780 | 0.2665 |
| nL8 Pure AE + Flow z (K16) | 27.18% | 71.10% | 86.72% | 72.82% | 0.957 | 2.692 | 0.2610 |
| nL8 E2 + Flow z (K16) | 37.88% | 61.48% | 78.76% | 62.12% | 1.473 | 7.089 | 0.2142 |

这版统计可作为后续 nL8 分支的“冻结口径基线”。

#### 7.14 Beta-VAE + Flow(mu) 修复与 B1/B2/B3（2026-02-24，进行中）

本节记录 2026-02-24 的代码修复与新一轮三实验流水线，目标是避免再次出现口径漂移与评估加载失败。

##### 7.14.1 代码修复（已完成）

本轮针对 `beta-vae + flow_target=vae_mu` 评估链路，修复了两个已复现故障：

1. Flow 条件编码器维度错配（`64 vs 256`）
- 现象：`way_casd_binned_eval.py` 在加载 `A2_flow_on_mu/ckpt_best.pt` 时，`cond_enc.*` 出现 size mismatch。
- 修复：评估端构建 Flow 条件编码器时不再沿用 AE decoder 的 `d_model`，改为按 Flow ckpt 配置（含 `d_model/n_latent`）构建。

2. 评估端缺少 `ConditionEncoderCfg` 导入
- 现象：`NameError: ConditionEncoderCfg is not defined`
- 修复：在 `src/evaluation/way_casd_binned_eval.py` 补齐导入并统一 `flow_target` 分支。

相关代码路径：
- `src/models/way_casd/way_casd.py`（β-VAE bottleneck：`vae_dim/vae_beta`）
- `src/training/train_way_casd_autoencoder.py`（β warmup 与日志）
- `src/training/train_way_casd_flow.py`（`--flow_target=vae_mu`）
- `src/evaluation/way_casd_binned_eval.py`（Flow ckpt 形状适配 + `flow_target` 解码路径）
- `src/evaluation/way_casd_zenc_informativeness.py`（AE 构造兼容 `vae_dim`）

##### 7.14.2 A1/A2/A3（k8）基线结果（已完成）

目录：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/`

- A1 (`A1_beta_vae_ae/report.json`)
  - best_epoch=99
  - best_val_loss=0.7671
  - 最终 val_acc≈0.748
- A2 (`A2_flow_on_mu/report.json`)
  - flow_target=`vae_mu`
  - flow_shape=`d_model=64, n_latent=1`
  - best_epoch=59, best_val_loss=0.4350
- A3-k8 (`A3_eval_k8/binned_betaVAE_flowmu_k8_dest_efficient_n5000.json`)
  - success_rate=0.6944
  - loop_rate=0.5720
  - success_only_len_ratio_mean=2.0295

##### 7.14.3 B1/B2/B3 流水线（进行中）

执行脚本：`run_beta_vae_b1_b2_b3.sh`

执行顺序（固定口径，GPU重任务串行）：
1. B1：现有 A1+A2，`K=16, sample_select=dest_efficient`（无 anti-loop）
2. B2：现有 A1+A2，`K=16, dest_efficient + anti_loop(k=4, penalty=2.0)`
3. B3：高容量 β-VAE
   - AE：`vae_dim=128, vae_beta=0.005, beta_warmup=30, n_epochs=150`
   - Flow：`flow_target=vae_mu`（μ∈R^128）
   - Eval：`K=16, dest_efficient`

统一评估口径（B1/B2/B3）：
- `split_part=test`, `n_routes=5000`, `seed=0`
- `decode_max_candidates=0`, `decode_candidate_policy=first`
- `city_grid_meta` 必传（meters 口径）
- `no_compare_beam`, `eval_batch_size=256`

输出目录：
- B1：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B1_eval_k16/`
- B2：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/`
- B3：`_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/`

实时日志：
- B1：`.../B1_eval_k16/run_eval_k16_dest_efficient.log`
- B2：`.../B2_eval_k16_antiloop/run_eval_k16_dest_efficient_antiloop.log`
- B3-AE：`.../A1_beta_vae128_ae/run_train_beta_vae128.log`
- B3-Flow：`.../A2_flow_on_mu128/run_train_flow_mu128.log`
- B3-Eval：`.../A3_eval_k16/run_eval_k16_dest_efficient.log`

> 状态：已完成（2026-02-24），下述为最终结果。

##### 7.14.4 B1/B2/B3 最终结果（已回填）

结果文件：
- B1：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B1_eval_k16/binned_betaVAE_flowmu_k16_dest_efficient_n5000.json`
- B2：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/binned_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json`
- B3：`_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/A3_eval_k16/binned_betaVAE128_flowmu_k16_dest_efficient_n5000.json`

> 说明：该批 `binned` 指标位于 `overall.greedy.cells`（按 bin 存储），全局值为按 `n` 加权。之前终端打印为 0 的原因是汇总脚本误读 `global` 字段。

| 配置 | success_rate | hit_wall_rate | loop_rate | fallback_rate | len_ratio_mean |
|---|---:|---:|---:|---:|---:|
| A3-k8（vae64基线） | 0.6944 | 0.2972 | 0.5720 | 0.3056 | 3.6602 |
| **B1: vae64, K16** | **0.7018** | **0.2928** | **0.6014** | **0.2982** | **3.2701** |
| **B2: vae64, K16 + anti-loop** | **0.7834** | **0.2086** | **0.4126** | **0.2166** | **3.2143** |
| **B3: vae128, K16** | **0.5350** | **0.4502** | **0.7350** | **0.4650** | **4.5517** |

按目标对照：
- B1 目标（预期 SR>80%）：**未达成**（70.18%）。
- B2 目标（零训练 anti-loop 提升）：**达成**，SR 显著提升到 **78.34%**，loop 从 0.6014 降到 **0.4126**。
- B3 目标（SR≥70%, len_ratio_p50≤1.2）：**未达成**。B3 SR=53.5%，len_ratio 全体中位数约 3.08（success-only 中位数约 1.25）。

B3 训练端（供定位）：
- AE128：`A1_beta_vae128_ae/report.json`，best_epoch=148, best_val_loss=0.7099, 最终 val_acc≈0.786
- Flow(mu128)：`A2_flow_on_mu128/report.json`，best_epoch=77, best_val_loss=0.5726（显著差于 mu64 的 0.4350）

结论（本轮）：
- 当前最优是 **B2（vae64 + K16 + anti-loop）**。
- 提升主要来自推理约束（anti-loop），不是增大 `vae_dim`。

##### 7.14.5 统计口径补充（per_route success-only + 汇总脚本修正）

为避免后续口径混淆，本节补充 `per_route` 聚合指标与脚本修正点。

`per_route` 文件：
- A3-k8：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/A3_eval_k8/per_route_betaVAE_flowmu_k8_dest_efficient_n5000.json`
- B1-k16：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B1_eval_k16/per_route_betaVAE_flowmu_k16_dest_efficient_n5000.json`
- B2-k16+AL：`_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json`
- B3-k16(vae128)：`_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/A3_eval_k16/per_route_betaVAE128_flowmu_k16_dest_efficient_n5000.json`

`per_route` 计算逻辑：
- 路由级样本来自 `per_route[*].greedy`
- `success_only_*` 指标在 `success=true` 子集上统计
- `p50/p95` 由 success 子集的 `len_ratio` 分位数计算

| 配置 | n_success | success_only_len_ratio_p50 | success_only_len_ratio_p95 | success_only_loop_rate | success_only_jaccard_mean |
|---|---:|---:|---:|---:|---:|
| A3-k8 (vae64) | 3472 | 1.3314 | 5.6259 | 0.3888 | 0.2075 |
| B1-k16 (vae64) | 3509 | 1.2380 | 4.4907 | 0.4349 | 0.2183 |
| B2-k16+AL (vae64) | 3917 | 1.2304 | 5.2196 | 0.2548 | 0.2114 |
| B3-k16 (vae128) | 2675 | 1.2538 | 5.8410 | 0.5110 | 0.2186 |

汇总脚本修正：
- 文件：`run_beta_vae_b1_b2_b3.sh`
- 修正点：汇总阶段从误读 `global.*` 改为读取 `overall.greedy.cells` 并按 `n` 加权聚合；当结构异常时输出 `PARSE_FAIL` 而非 `0.0000`。

##### 7.14.6 C1/C2/C3 推进（2026-02-24）

本轮新增执行脚本：
- `run_beta_vae_c2_c3.sh`

###### C1（已完成）：B2 的 OD coverage/diversity

输入：
- `per_route`: `_sync/wsa/pi_verify/20260223_porto_beta_vae_flowmu_s0/B2_eval_k16_antiloop/per_route_betaVAE_flowmu_k16_dest_efficient_antiloop_n5000.json`

输出：
- `json`: `_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C1_od_coverage_b2/od_coverage_diversity_b2_k16_n5000.json`
- `log`: `_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C1_od_coverage_b2/run_od_coverage_diversity_b2_k16_n5000.log`

结果（`k=16, min_routes_per_od=3, jaccard_threshold=0.5`）：
- Arrival: **0.7834**
- GT Coverage@K: **0.1366**
- Self-Diversity@K: **0.5316**
- n_OD: **154**

###### C2（待执行）：β-VAE GT μ oracle（K=1）

目标：
- 用 `latent_source=gt` 给出 β-VAE decoder 在 GT latent 下的上限口径（K=1）。

脚本中对应步骤：
- `C2_gt_mu_oracle_k1`（见 `run_beta_vae_c2_c3.sh`）

输出目录（约定）：
- `_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C2_gt_mu_oracle_k1/`

###### C3（待执行）：修复 vae128 Flow（n_layers=6）+ K16 eval

目标：
- 在不重训 AE128 的前提下，重训 Flow on μ128（`n_layers=6`）并评估 K16。

脚本中对应步骤：
- `C3_train_flow_mu128_l6`
- `C3_eval_k16_dest_efficient`

输出目录（约定）：
- Flow训练：`_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C3_flow_on_mu128_l6_fix/`
- 评估：`_sync/wsa/pi_verify/20260224_porto_beta_vae128_flowmu_s0/C3_eval_k16_l6_fix/`

##### 7.14.7 Baseline Eval + 可视化打包（2026-02-24）

新增脚本：
- `run_porto_baseline_eval_and_viz.sh`

脚本定位：
- 基线评估与可视化一体化流水线（Porto）。
- 先复用已有 `RNN/Transformer` 的 `n=5000` 结果；缺失才补跑。
- 输出 `Phase-C`、`success-only` 质量、`Hero OD`、`Loop Leaflet`、`len_ratio` 直方图。
- 前台实时日志，不后台运行，不因单步失败直接退出（记录 `FAILED STEPS`）。

本轮已完成并落盘的数据结果：

1) Phase-C（B2 vs RNN vs Transformer，K=16）  
输出：
- `_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/phaseC/od_coverage_diversity_b2_rnn_tr_k16_n5000.json`

结果（`min_routes_per_od=3, jaccard_threshold=0.3`）：
- `BetaVAE64_FlowMu_K16_AL`：Arrival **0.7834**, Coverage@K(mean) **0.3854**, Diversity@K(mean) **0.5316**, n_OD=154
- `RNN_AR_b10`：Arrival **0.1928**, Coverage@K(mean) **0.0774**, Diversity@K(mean) **0.1857**, n_OD=154
- `Transformer_AR_b10`：Arrival **0.2322**, Coverage@K(mean) **0.0807**, Diversity@K(mean) **0.2616**, n_OD=154

2) success-only 质量汇总（从 per_route 直接聚合）  
输出：
- `_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/success_only_quality_b2_vs_baselines_n5000.json`

结果：
- `BetaVAE64_FlowMu_K16_AL`：success=**0.7834**, success-only len_ratio p50=**1.230**, success-only loop_rate=**0.255**
- `RNN_AR_b10`：success=**0.1928**, success-only len_ratio p50=**1.372**, success-only loop_rate=**0.235**
- `Transformer_AR_b10`：success=**0.2322**, success-only len_ratio p50=**1.493**, success-only loop_rate=**0.399**

说明：
- 本节已完成的是“评估数值与汇总产物”。
- `Hero/Leaflet/Histogram` 图像产物由 `run_porto_baseline_eval_and_viz.sh` 在 WSL `dpl` 环境执行生成（依赖 `matplotlib`/地图元数据路径）。

###### 7.14.7-a Coverage-τ 与 threshold-free 指标补充（2026-02-25）

代码更新：
- `src/evaluation/od_coverage_diversity_eval.py`
  - 新增 `--tau_values`（默认 `0.1..0.9`）
  - 新增输出：
    - `mean_max_jaccard_at_k`（threshold-free 主指标）
    - `coverage_vs_tau`（Coverage-τ 曲线，含 mean/p25/p50/p75）
    - `coverage_vs_tau_auc`（曲线 AUC，归一化到 `[0,1]`）

结果文件：
- `_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/phaseC/od_coverage_diversity_b2_rnn_tr_k16_n5000_taucurve.json`
- `_sync/wsa/pi_verify/20260224_porto_baseline_eval_viz_bundle_s0/phaseC/coverage_tau_keypoints_b2_rnn_tr_k16_n5000.json`

关键数值（K=16）：
- `BetaVAE64_FlowMu_K16_AL`
  - Mean Max-Jaccard: **0.2821**
  - Coverage-τ AUC: **0.2162**
  - Coverage@τ=0.3: **0.3854**
  - Coverage@τ=0.5: **0.1366**
- `RNN_AR_b10`
  - Mean Max-Jaccard: **0.0941**
  - Coverage-τ AUC: **0.0452**
  - Coverage@τ=0.3: **0.0774**
  - Coverage@τ=0.5: **0.0227**
- `Transformer_AR_b10`
  - Mean Max-Jaccard: **0.0991**
  - Coverage-τ AUC: **0.0517**
  - Coverage@τ=0.3: **0.0807**
  - Coverage@τ=0.5: **0.0200**
