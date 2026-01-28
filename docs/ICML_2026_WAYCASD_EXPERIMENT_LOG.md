# ICML 2026 RouteGen：Way-CASD 主线实验记录（给新 PI 的快速背景）

> 口径声明：本文只记录**已跑过且在本地 `_sync/wsa/icml2026_routegen/` 可查到的结果**，并把关键实验的**设置/结果/结论**压缩成可审阅版本。  
> 工作站真实落盘与环境约定见：`docs/WORKSTATION_GUIDE.md`。  
> legacy（Map-free / raster / segment）E 系列实验索引见：`docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md`。

---

## 0. 一句话结论（当前状态）

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
- 本地结果索引：`_sync/wsa/icml2026_routegen/`
- Way-CASD 关键目录（本地同步）：
  - AE+诊断：`_sync/wsa/icml2026_routegen/WAYCASD_PASTCTX_strict_sem5_rustbelt_seed0/`
  - Beam 大样本：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_beam_gt_pastctxfix_seed0_n200pc/`
  - Flow any-success：`_sync/wsa/icml2026_routegen/WAYCASD_DIAG_flow_anysucc_pastctxfix_seed0_N*_n200/`

