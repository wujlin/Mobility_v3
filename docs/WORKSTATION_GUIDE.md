# 工作站/本地环境使用说明（运行与数据落盘口径）

> 目标：把“在哪跑、用哪个环境、数据落哪里、日志怎么留、常见故障怎么定位”写成一份可复现口径，避免跨机器/跨人协作时反复踩坑。  
> 说明：**数据契约**以 `docs/DATA_CONTRACT.md` 为准；**数据目录结构**以 `docs/DATA_STRUCTURE.md` 为准；本文只记录运行环境与操作习惯的约定。

---

## 0) 一句话约定（最重要）

- 仓库只放代码与小体量产物；大数据一律落到外置目录 `$RAW_ROOT`（不进 git），用软链/环境变量引用。

---

## 1) 机器与路径约定

### 1.1 工作站 A（训练/大规模处理）

- **设备信息（最小可复现口径）**：
  - hostname/别名：`wsa`（ssh config；兼容旧别名 `wsA`）
  - OS/Kernel：以 `uname -a` 为准
  - CPU：24C/48T（已知；以 `lscpu` 为准）
  - GPU：以 `nvidia-smi` 为准
  - 内存：以 `free -h` 为准

- **仓库路径**：`~/projects/Mobility_v3`
- **外置数据根目录（唯一口径）**：`$RAW_ROOT=/home/jinlin/data/geoexplicit_data`
- **常用数据目录**（示例）：
  - WorldTrace：`$RAW_ROOT/worldtrace/`
  - OSM：`$RAW_ROOT/osm/`
  - SafeGraph：`$RAW_ROOT/safegraph/`
  - Wayback：`$RAW_ROOT/wayback/`
  - Census：`$RAW_ROOT/census/`

建议把这两个变量写进 `~/.bashrc`（只做本机约定，不要写进仓库脚本）：

```bash
export REPO="$HOME/projects/Mobility_v3"
export RAW_ROOT="$HOME/data/geoexplicit_data"
```

设备信息快速自检（仅打印，不写盘）：

```bash
uname -a
lscpu | head
free -h
nvidia-smi || true
df -h | head
```

#### 1.1.1 工作站 A 数据内容（当前主线：WorldTrace Detroit/Columbus）

> 口径：以下路径是当前主线（Detroit story）在工作站 A 上的**实际落盘结构**；不涉及 legacy 深圳分析。

| 数据 | 目录 | 关键文件（示例） | 说明 |
|---|---|---|---|
| WorldTrace 原始包 | `$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/` | `Trajectory.zip`, `Meta.zip` | 主数据底座（大文件，不进 git） |
| Detroit segments（Way-CASD） | `$RAW_ROOT/worldtrace/detroit_core_v1/` | `segments_with_wayid.parquet` | Way-CASD 主线输入（包含 `osm_way_id`） |
| Columbus segments（Way-CASD） | `$RAW_ROOT/worldtrace/columbus_core_v1/` | `segments_with_wayid.parquet` | 同上 |
| segments.parquet（legacy） | `$RAW_ROOT/worldtrace/<city>_core_v1/` | `segments.parquet` | 旧版不含 `osm_way_id`；除非明确需要，否则不要用它做 Way-level pipeline |
| OSM（Detroit） | `$RAW_ROOT/osm/` | `michigan-latest.osm.pbf` | 用于生成 `road_prob/dist_to_road`（soft prior） |
| OSM（Columbus） | `$RAW_ROOT/osm/` | `ohio-latest.osm.pbf` | 同上 |
| OSM 软先验产物（每城） | `$RAW_ROOT/worldtrace/<city>_core_v1/` | `osm_road_prob.npy`, `osm_dist_to_road_m.npy`, `osm_road_prob_meta.json` | **唯一口径以 meta 为准**（variant/sigma/buffer/tier_weights 等） |
| SafeGraph（POI shards） | `$RAW_ROOT/safegraph/safegraph_unzip/` | `Global_Places_POI_Data-*.csv` | 目前为 Places Base 分片；Rich/Geometry 若缺失需在契约里标注 |
| POI 栅格化产物（可选） | `$RAW_ROOT/worldtrace/<city>_core_v1/` | `poi_density_*.npy`, `poi_raster_meta.json` | 用于 POI heatmap / 语义通道；不保证每城都有（Detroit/Columbus 当前已生成；其他城市若缺失需额外生成） |
| Wayback 遥感（Detroit） | `$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6/` | `wayback_scan_meta.json`, `z16/.../rid_<release_id>.jpg` | 以 `release_id` 作为快照标识；不以 release_date 做时间证据 |
| Census/ACS（Detroit） | `$RAW_ROOT/census/detroit_core_v1/` | `acs_tract_*.csv`, `tract_covariates_*.parquet` | TIGER 若遇 403 可手动下载；注意 GeoParquet vs 普通 parquet 读取方式 |

> [!NOTE]
> `poi_density_*.npy/landuse_entropy.npy` **不是 Way-CASD 必需输入**（Decision/Execution 训练不依赖它）；主要用于 **POI heatmap 可视化** 与（可选）语义通道实验。  
> 若某城市缺失这些文件，可用 SafeGraph base shards 生成到对应的 `$RAW_ROOT/worldtrace/<city>_core_v1/`：
>
> ```bash
> python -m src.data.safegraph.build_poi_rasters \
>   --base_dir "$RAW_ROOT/safegraph/safegraph_unzip" \
>   --base_glob "Global_Places_POI_Data-*.csv" \
>   --out_dir "$RAW_ROOT/worldtrace/<city>_core_v1" \
>   --bbox <min_lon> <min_lat> <max_lon> <max_lat> \
>   --grid_h <H> --grid_w <W> \
>   --vintage 2024-01
> ```
>
> 口径要求：`--bbox/--grid_h/--grid_w` 必须与该城市的 `osm_road_prob_meta.json` 里的 grid 一致（否则 POI 栅格与 road_prob 无法对齐）。

#### 1.1.2 数据快照（仅用于排错；不作为论文证据）

> 目的：当“跑不动/找不到文件/数据不一致”时，用一组可复现检查项快速判断是否为**路径/缺文件/版本不同**导致。  
> 说明：数值会随数据版本变化；论文正文不直接引用。

- Detroit（WorldTrace core, Way-CASD）：`$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet`（约 2.3k segments）
- Columbus（WorldTrace core, Way-CASD）：`$RAW_ROOT/worldtrace/columbus_core_v1/segments_with_wayid.parquet`（约 5.2k segments）
- （legacy）兼容旧脚本：`$RAW_ROOT/worldtrace/<city>_core_v1/segments.parquet`（不含 `osm_way_id`）
- SafeGraph Places（Base shards）：`$RAW_ROOT/safegraph/safegraph_unzip/Global_Places_POI_Data-*.csv`（当前 64 分片）
- Wayback Detroit（z=16，多 release）：`$RAW_ROOT/wayback/detroit_core_z16_fixed_multi_r6/`（目标 6×3472=20832 tiles）
- Census Detroit（tract covariates）：`$RAW_ROOT/census/detroit_core_v1/tract_covariates_detroit_core.clean.parquet`（约 419 tracts）

### 1.2 本地 WSL（写作/轻量分析/拉图）

- **仓库路径（WSL）**：`/mnt/e/newdesktop/HKUST/GeoExplicit_SFM/v3`
- 若需要从工作站拉取图/JSON：使用 `rsync -avP wsa:... local/...`（见第 6 节）

---

## 2) Python/环境口径

仓库代码默认按 `python -m src...` 运行，要求：

- 当前工作目录在仓库根（即 `pwd == $REPO`）
- `python` 对应的环境里安装了 `requirements.txt` 所需依赖

建议实践（不是硬要求）：

- 训练/评估：用一个统一 conda env（例如 `dpl`）
- 地理处理（OSM/Wayback/Census）：可以单独一个 env（例如 `geo`），但**避免同一任务跨 env 混跑**

快速自检（跑任何长任务前）：

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch; print('cuda', torch.cuda.is_available())"
```

---

## 3) 日志与可观测性（避免“跑了但不知道在干嘛”）

### 3.1 推荐的运行方式

所有长任务都用**非缓冲输出**并落盘日志：

```bash
PYTHONUNBUFFERED=1 python -u -m <module> <args...> |& tee "<out_dir>/run.log"
```

另开窗口看进度：

```bash
tail -f "<out_dir>/run.log"
```

> 不要用 `>log 2>&1 &` 把输出藏起来；这会让“卡住/报错/被杀”无法定位。

### 3.2 `set -euo pipefail` 是什么？为什么看起来像“窗口断开”？

它是 bash 的“快速失败”开关：

- `-e`：任一命令失败（非 0）就立刻退出
- `-u`：引用未定义变量就报错退出
- `pipefail`：管道里任一环节失败都算失败

**它不会主动断开 ssh/tmux**；但如果脚本中间有一条命令失败，整个脚本会立即结束，ssh 远程命令结束后你会回到本地 shell，看起来像“刚跑完/突然没了”。因此：

- 交互式排错阶段不建议加这句
- 批处理脚本阶段可以加，但要确保日志完整落盘

---

## 4) tmux 使用约定（工作站 A 必备）

典型流程：

```bash
tmux new -s detroit
# 里面跑任务…
# Ctrl-b d 退出
tmux attach -t detroit
```

建议：

- 每个长任务输出到独立 `out_dir`，不要共享一个 `run.log`
- tmux 里跑多进程任务时，优先保证日志可追踪（见第 3 节）

---

## 5) 多进程/并发（常见坑与可复现实践）

### 5.1 ProcessPool 的两类典型错误

- `BrokenProcessPool`：子进程被异常杀死（内存/句柄/依赖/数据损坏）
- `OSError: handle is closed`：常见于 `spawn` 启动方式下的句柄序列化/关闭时序问题

### 5.2 建议的跑法（以 WorldTrace segments 构建为例）

在 Linux 上默认优先用 `fork`；若确有需要再切 `spawn`：

```bash
python -m src.data.worldtrace.build_detroit_segments \
  --trajectory_zip "$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip" \
  --out_parquet "$RAW_ROOT/worldtrace/<city>_core_v1/segments_with_wayid.parquet" \
  --bbox <min_lon> <min_lat> <max_lon> <max_lat> \
  --require_way_id \
  --num_workers 24 \
  --chunk_size 5000 \
  --mp_start fork \
  ...
```

如果出现不稳定（随机崩/句柄错误），只调整两件事（保持 KISS）：

- 降低 `--num_workers`
- 增大 `--chunk_size`（减少进程间通信频率）

---

## 6) 跨机器传输（图/JSON/小产物）

### 6.0 代码仓库同步（本地 → wsa）

> 目标：**本地改代码 → 同步到工作站跑 → rsync 结果回本地给 PI review**。  
> 原则：只同步“代码/脚本/小文档”，**不要**把 `$RAW_ROOT/` 或大产物同步到 git 仓库目录里。

**推荐做法（wsa 可访问 git remote）**：直接在工作站拉取最新代码

```bash
ssh wsa "cd ~/projects/Mobility_v3 && git pull"
```

**备选做法（wsa 无法 git pull：无网/权限）**：从本地用 `rsync` 推送代码

```bash
# 本地：在仓库根目录执行（pwd == <repo_root>）
rsync -avP \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".pytest_cache/" \
  --exclude ".mypy_cache/" \
  --exclude ".ruff_cache/" \
  --exclude ".venv/" \
  --exclude "_sync/" \
  --exclude "data/" \
  ./ wsa:"~/projects/Mobility_v3/"
```

> [!WARNING]
> 不要默认加 `--delete`。如果你确实需要镜像同步：先 `--dry-run` 预演，再确认远端目录有备份/在版本控制中。

从工作站 A 拉图到本地示例（建议先落到 `_sync/wsa/...`，再用软链接接入论文/报告目录）：

```bash
rsync -avP wsa:"$RAW_ROOT/worldtrace/detroit_core_v1/story/" \
  "_sync/wsa/worldtrace_detroit/story/"
```

从工作站拉回某次实验 `out_dir`（通用模板）：

```bash
# 约定：远端 out_dir 位于 $RAW_ROOT/experiments/...；本地统一落到 _sync/wsa/...
rsync -avP wsa:"$RAW_ROOT/experiments/<proj>/<EXP_DIR>/" \
  "_sync/wsa/<proj>/<EXP_DIR>/"
```

建议：

- 只拉 `png/pdf/json` 等小文件，不要拉 zip/pbf 大文件到本地
- 若网络不稳：加 `--partial --append-verify` 断点续传

### 6.1 本地 `_sync` 目录口径（ICML 2026 RouteGen）

> 目的：让“跑完→rsync→写论文引用”这一链路可持续，不因目录命名漂移而丢证据或找不到文件。

- **统一落点**：本地一律同步到 `_sync/wsa/icml2026_routegen/<EXP_DIR>/`。
- **不移动原始同步目录**：避免下一次 rsync 把目录“同步回去”导致冲突；若发现命名不一致，优先用**软链接别名**解决。
  - 例：`E22a_audit_... -> E20a_audit_...`（保证旧路径仍可增量同步，同时论文引用用新名字）。
- **同步结果索引**：每次同步后可生成一份可检索清单（便于 PI review / 写作引用）：

```bash
python tools/gen_routegen_sync_manifest.py \
  --root _sync/wsa/icml2026_routegen \
  --out_json docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.json \
  --out_md docs/ICML_2026_ROUTEGEN_SYNC_MANIFEST.md
```

### 6.2 ICML 2026 RouteGen（Graph）最短复现命令（T3/T4）

> 口径：所有 `--out_dir` 都落到工作站 `$RAW_ROOT/experiments/icml2026_routegen/...`，本地用 `rsync` 拉到 `_sync/wsa/...`。

**(0) 先做一次“同 OD / 同 OD-bin 多实例”审计（避免 corridor 定义踩坑）**：

> 说明：很多可视化是“单条 GT（黑）vs 多次采样（蓝）”，这只能说明 \emph{single-trajectory match}（覆盖/重建），不能直接支撑 \emph{corridor-level multi-modality}。  
> 这一步的输出会告诉你：数据里到底有没有足够的“同 OD（或同 OD-bin）多条 GT”可以用来定义/评估 corridor diversity。

```bash
python -m src.data.road_graph.od_group_stats_paths_graph_npz \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz" \
  --out_json "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/od_group_stats_od128_mt5_seed0.json" \
  --od_bin 128 --min_traj_per_od 5 --multimodal_dist_thr 0.3 \
  --max_groups 200 --max_pairs 200 --seed 0
```

**(1) 构建 road graph（每城一次）**：

```bash
# Raster（legacy）：Bresenham 像素化到 grid cell，粒度会非常细（edge_len p50≈grid 分辨率）
python -m src.data.road_graph.build_road_graph_from_osm \
  --osm_pbf "$RAW_ROOT/osm/michigan-latest.osm.pbf" \
  --semantic_dir "$RAW_ROOT/worldtrace/detroit_core_v1" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/G1r_roadgraph_detroit_raster" \
  --city detroit --road_types B

# Native（推荐）：OSM node/edge 原生图（仍可能很细，常见 edge_len p50≈10–20m；需要后续 degree-2 collapse 才接近“intersection-to-intersection segment”）
python -m src.data.road_graph.build_road_graph_native_from_osm \
  --osm_pbf "$RAW_ROOT/osm/michigan-latest.osm.pbf" \
  --semantic_dir "$RAW_ROOT/worldtrace/detroit_core_v1" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/G1n_roadgraph_detroit_native" \
  --city detroit --road_types B

# 快速审计（node-to-node 的 edge_len 仅做 sanity check；真正关心的是后续 segment_graph 的 seg_len）
python -m src.data.road_graph.audit_road_graph_npz \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/G1n_roadgraph_detroit_native/road_graph.npz"
```

### 6.3 CASD（Corridor-Aware Segment Diffusion）最短复现命令（S1-S4）

> 口径：先用 combo（Detroit+Columbus）保证 corridor diversity；segment token 有两种口径：
> - `SEG_MODE=collapse`：degree-2 chain collapse（推荐；native OSM 的 edge 仍然是 OSM node-to-node，长度常见 10–20m，需要先 collapse 才接近“intersection-to-intersection segment”）
> - `SEG_MODE=edge`：每条 directed edge 作为一个 segment（仅建议 debug 或你确认 edge_len 已足够粗时使用）
>
> Native 图的经验值（KISS）：service/unclassified 分支会让“degree==2”判断失效，导致 segment 仍很短。推荐在 `SEG_MODE=collapse` 时加：
> - `SEG_COLLAPSE_DEGREE_MODE=undir`（用 undirected degree 定义 junction）
> - `SEG_COLLAPSE_TIER_MAX=1`（collapse 时忽略 tier>=2 的分支；service 仍保留为单边 segment，不会混入主干 collapse）

```bash
# 1) 拉取最新代码
cd ~/projects/Mobility_v3 && git pull

# 2) 设置环境变量
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export IN_DATA="$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0"
export PATHS_NPZ="$IN_DATA/paths_graph_combo.npz"
export ROAD_NPZ="$IN_DATA/road_graph_combo.npz"
export OUT_BASE="$RAW_ROOT/experiments/icml2026_routegen/CASD0_segdata_combo_seed0_term"
# export SEG_MODE=collapse   # 默认就是 collapse；edge 仅用于 debug
export SEG_COLLAPSE_DEGREE_MODE=undir
export SEG_COLLAPSE_TIER_MAX=1

# 3) 数据准备（Step 0）
bash run_casd_prep.sh

# 4) 训练 AE（Step A）
python -m src.training.train_casd_autoencoder \
  --segment_graph_npz "$OUT_BASE/S1_segment_graph/segment_graph.npz" \
  --routes_npz "$OUT_BASE/S2_segment_routes/segments_graph_routes.npz" \
  --out_dir "$OUT_BASE/S3_train_ae" \
  --batch_size 8 \
  --num_workers 8 \
  --n_epochs 20 \
  --d_model 256 \
  --n_latent 64 \
  --device cuda

# 5) 训练 Flow（Step B）
python -m src.training.train_casd_flow \
  --segment_graph_npz "$OUT_BASE/S1_segment_graph/segment_graph.npz" \
  --routes_npz "$OUT_BASE/S2_segment_routes/segments_graph_routes.npz" \
  --ae_ckpt "$OUT_BASE/S3_train_ae/ckpt_best.pt" \
  --out_dir "$OUT_BASE/S4_train_flow" \
  --batch_size 8 \
  --num_workers 8 \
  --n_epochs 20 \
  --d_model 256 \
  --n_latent 64 \
  --cfg_drop_prob 0.1 \
  --device cuda
```

调参建议（KISS）：
- `batch_size`：按 `nvidia-smi` 逐步翻倍（8→16→32）；若 dataloader 跟不上再加 `--num_workers`（建议 8-16）。
- `n_latent`：当前 combo 的 `seg_len_p90≈442`；若 AE 重建精度不够，优先把 `n_latent` 提到 96/128，再考虑其他改动。

### 6.4 Way-CASD（Way-token CASD）主线复现命令（W1-W4）

> 关键动机：直接使用 WorldTrace 的 `osm_way_id` 作为离散 token，避免 “GPS→node snap→bridging” 导致的千级序列长度爆炸。
> Detroit core 的审计结果：连续去重后的 `uniq_way_seq_len` 为 `p50≈35 / p90≈61`（已经对齐 GTG/Cardiff 的粒度）。

**(0) 产出含 `osm_way_id` 的 segments parquet（只需每城一次）**

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export INPUT_ZIP="$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip"
export OUT_SEG="$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet"

python -m src.data.worldtrace.build_detroit_segments \
  --trajectory_zip "$INPUT_ZIP" \
  --out_parquet "$OUT_SEG" \
  --require_way_id \
  --num_workers 24 --chunk_size 5000 --mp_start fork \
  |& tee "$EXP_ROOT/A_build_segments_wayid_detroit/run.log"

# Go/No-Go：way 序列长度分布（p50 是否 ~10–40）
python -m src.data.worldtrace.way_seq_stats_from_segments \
  --segments_parquet "$OUT_SEG" \
  --out_json "$EXP_ROOT/A_wayseq_detroit_seed0/report.json" \
  |& tee "$EXP_ROOT/A_wayseq_detroit_seed0/run.log"
```

**(0b) corridor-level 多模态 OD 扫描（推荐：用于数据筛选/PI sanity）**

> 口径：按 OD-bin 聚合同一 OD 的多条 GT，使用 **way-id 序列**做 signature，并用 **LCS distance** 判定是否存在多个走廊 mode。  
> 细节与术语区分（corridor vs corridor_type）见：`docs/WAY_CASD_METHOD.md`。

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export INPUT_ZIP="$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip"
export OUT_MM="$EXP_ROOT/A_mm_od_mioh_v2_bin02_sep50"

python -m src.data.worldtrace.scan_multimodal_od_region \
  --trajectory_zip "$INPUT_ZIP" \
  --out_json "$OUT_MM/report.json" \
  --bbox -90.4 38.4 -80.5 48.3 \
  --od_bin_deg 0.02 \
  --max_way_seq_len 128 \
  --min_routes_per_od 5 \
  --min_cluster_frac 0.2 \
  --cluster_sep_thr 0.50 \
  --merge_dist_thr 0.15 \
  --num_workers 48 --chunk_size 2000 --mp_start fork \
  |& tee "$OUT_MM/run.log"

# （推荐）先把代表轨迹抽出来做 viz cache，避免可视化时反复读 Trajectory.zip
python -m src.data.worldtrace.dump_multimodal_viz_cache \
  --scan_report_json "$OUT_MM/report.json" \
  --trajectory_zip "$INPUT_ZIP" \
  --out_npz "$OUT_MM/viz_cache_top200.npz" \
  --top_k 200 --clusters_keep 2 --max_files_per_cluster 2 \
  --prefer_matched --downsample_step 10 \
  --num_workers 48 --chunk_size 256 --mp_start fork \
  |& tee "$OUT_MM/viz_cache_top200.run.log"

# 可视化 top-K / random-K multimodal OD（输出两 panel：轨迹对比 + corridor footprint）
export OSM_MI="$RAW_ROOT/osm/michigan-latest.osm.pbf"
export OSM_OH="$RAW_ROOT/osm/ohio-latest.osm.pbf"
python -m src.evaluation.plot_worldtrace_multimodal_od_bins \
  --scan_report_json "$OUT_MM/report.json" \
  --viz_cache_npz "$OUT_MM/viz_cache_top200.npz" \
  --out_dir "$OUT_MM/viz_rand5_seed0" \
  --random_k 5 --seed 0 --max_files_per_cluster 2 \
  --prefer_matched --downsample_step 10 \
  --osm_pbf_michigan "$OSM_MI" --osm_pbf_ohio "$OSM_OH" \
  |& tee "$OUT_MM/viz_rand5_seed0/run.log"
```

**(0c) 从 multimodal scan 抽取 Way-CASD 训练 routes（可选：只训练“多走廊 OD”）**

> 说明：这是“数据筛选版”Way-CASD；会生成一个新的 `way_routes.npz`（并写出 `members.jsonl` 便于复现/追溯）。  
> 默认仅保留 `O&D` 都落在 MI/OH 的 OD（避免额外州的 OSM pbf 依赖）。

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export INPUT_ZIP="$RAW_ROOT/worldtrace/OpenTrace_WorldTrace/Trajectory.zip"
export IN_MM="$EXP_ROOT/A_mm_od_mioh_v2_bin02_sep50/report.json"
export OUT_BASE="$EXP_ROOT/WAYMM1_waydata_mioh_od0p02_seed0"

python -m src.data.way_graph.build_way_routes_from_multimodal_scan \
  --scan_report_json "$IN_MM" \
  --trajectory_zip "$INPUT_ZIP" \
  --out_npz "$OUT_BASE/W1_way_routes/way_routes.npz" \
  --od_filter mi_oh --prefer_matched \
  --min_seq_len 2 --coord_scale 1024 \
  --num_workers 48 --chunk_size 2000 --mp_start fork --seed 0

# 后续：用 W1_way_routes 继续构建 W3/W4/W5，再训练 AE/Flow
```

**(W1-W4) Way-CASD 数据准备（routes / graph / features / corridor label）**

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export SEGMENTS_PARQUET="$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet"
export SEMANTIC_DIR="$RAW_ROOT/worldtrace/detroit_core_v1"         # 提供 bbox/H/W（osm_road_prob_meta.json）
export OSM_PBF="$RAW_ROOT/osm/michigan-latest.osm.pbf"
export OUT_BASE="$EXP_ROOT/WAYCASD0_waydata_detroit_seed0"

bash run_way_casd_prep.sh
```

**(多城市) Rust Belt（Detroit+Columbus）合并数据准备**

> 目标：增大 routes 数量与分支覆盖，缓解 Detroit 单城 `N≈2k` 的欠拟合与 adjacency 过稀问题。

```bash
export RAW_ROOT=/home/jinlin/data/geoexplicit_data
export EXP_ROOT="$RAW_ROOT/experiments/icml2026_routegen"
export OUT_BASE="$EXP_ROOT/WAYCASD1_waydata_rustbelt_seed0"
# 可选：把 transition adjacency 做成无向（增加候选；KISS debug 用）
# export WAY_GRAPH_UNDIR=1

bash run_way_casd_prep_rustbelt.sh
```

**(推荐) Strict v1 + 语义 features：目录口径（避免路径写错）**

> 这个版本用于“数据质量优先 + 注入 `way_semantic`”。目录名里会出现 `*_strict_v1/`，并额外生成 `W4_way_features_sem/` 与 `W5_way_routes_strict/`。
>
> 常见踩坑：把 `way_routes_strict_masklen0.npz` 错写到 `way_features_sem/` 目录下。**正确口径是：routes/graph/features 分别在 W5/W3/W4。**

典型设置（Rust Belt strict v1）：

```bash
export OUT_BASE="$EXP_ROOT/WAYCASD1_waydata_rustbelt_seed0_strict_v1"
```

| stage | 目录 | 关键文件 |
|---|---|---|
| audit | `$OUT_BASE/W0_audit/` | `report_strict.json`, `strict_route_coverage_audit.json` |
| graph | `$OUT_BASE/W3_way_graph_strict/` | `way_graph.npz` |
| features（含语义） | `$OUT_BASE/W4_way_features_sem/` | `way_features.npz`（包含 `way_semantic`） |
| routes（strict） | `$OUT_BASE/W5_way_routes_strict/` | `way_routes_strict_masklen0.npz` |

建议训练/评估前显式设定 3 个路径变量（后续所有命令都用它们，避免手滑）：

```bash
export WAY_ROUTES_NPZ="$OUT_BASE/W5_way_routes_strict/way_routes_strict_masklen0.npz"
export WAY_GRAPH_NPZ="$OUT_BASE/W3_way_graph_strict/way_graph.npz"
export WAY_FEATS_NPZ="$OUT_BASE/W4_way_features_sem/way_features.npz"
```

**(可选 P0) 用 OSM 拓扑增强 way_graph（用于 Hierarchical / Louvain Region）**

> 背景：仅用 GT transition 构建的 `way_graph.npz` 可能严重碎片化（largest CC 很小），会直接阻塞 Louvain/Region→Way 的层级规划。
>
> 口径：我们把“OSM 物理相连”（共享至少一个 OSM node）与“行为相连”（GT transitions）合并成新图。
>
> ⚠️ 常见踩坑：`.osm.pbf` **不在** `$RAW_ROOT/worldtrace/`，而在 `$RAW_ROOT/osm/`。

```bash
# OSM pbf（Detroit=Michigan, Columbus=Ohio；文件名不一致就先 ls 看一下）
export OSM_MI="$RAW_ROOT/osm/michigan-latest.osm.pbf"
export OSM_OH="$RAW_ROOT/osm/ohio-latest.osm.pbf"
ls -lh "$OSM_MI" "$OSM_OH" || (echo ">>> Available pbfs:" && ls -lh "$RAW_ROOT/osm" | head)

# 依赖：pyosmium（只需要装一次；推荐 conda-forge）
python -c "import osmium; print('osmium ok')" \
  || (echo ">>> Installing pyosmium..." && conda install -c conda-forge pyosmium -y)

export OUT_WG="$OUT_BASE/W3b_way_graph_osm_topo"
PYTHONUNBUFFERED=1 python -u -m src.data.way_graph.build_way_graph_from_osm_pbf_topology \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --osm_pbf "$OSM_MI" \
  --osm_pbf "$OSM_OH" \
  --out_npz "$OUT_WG/way_graph_osm_topo.npz" \
  |& tee "$OUT_WG/run_build_way_graph_osm_topo.log" \
  && python -m src.data.way_graph.audit_way_graph_npz \
    --way_graph_npz "$OUT_WG/way_graph_osm_topo.npz"
```

> [!NOTE]
> 如果你跑的是 `bash run_way_casd_prep.sh`（单城市），目录命名是 `W1/W2/W3/W4`；
> 训练命令里把 `W5_way_routes_labeled/W3_way_graph/W4_way_features` 分别替换为
> `W4_way_routes_labeled/W2_way_graph/W3_way_features`，并将 `W6_train_ae/W7_train_flow` 相应替换为 `W5_train_ae/W6_train_flow`。

**(Step A) 训练 AE（48GB GPU 起步建议：`batch_size=512`；若 OOM 再降到 256；`num_workers=48`；建议 `n_epochs=60`）**

> 可选诊断：如果你要验证 “decoder 是否过度依赖 `dest_dist` 产生 shortcut”，在训练命令里加 `--no-decoder_use_dest_dist`（默认是启用的）。

```bash
python -m src.training.train_way_casd_autoencoder \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --out_dir "$OUT_BASE/W6_train_ae" \
  --batch_size 512 --num_workers 48 --n_epochs 60 \
  --d_model 256 --n_latent 64 --max_candidates 64 --max_way_len 128 \
  --device cuda
```

**(Step B) 训练 Flow（同上资源配置；若显存仍空闲可继续翻倍 batch；建议 `n_epochs=60`）**

```bash
python -m src.training.train_way_casd_flow \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --out_dir "$OUT_BASE/W7_train_flow" \
  --batch_size 512 --num_workers 48 --n_epochs 60 \
  --d_model 256 --n_latent 64 --solver_steps 20 \
  --device cuda
```

**(采样 + 可视化) Flow→latent→Way 序列（默认 Greedy；Beam 可选）**

```bash
python -m src.evaluation.way_casd_sample_viz \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --flow_ckpt "$OUT_BASE/W7_train_flow/ckpt_best.pt" \
  --out_dir "$OUT_BASE/W8_sample_viz" \
  --n_routes 12 --n_samples_per_route 4 \
  --decode greedy --max_decode_len 160 \
  --plot_all_ways \
  --device cuda
```

输出：
- `W8_sample_viz/report.json`：每条 route 的 success/valid/jaccard
- `W8_sample_viz/city0/case_route*.png`、`W8_sample_viz/city1/case_route*.png`：GT（黑）+ 多个 sample（彩色）叠图（按 city 分目录）

**(指标评估) Decision Stage：Way 序列生成**

> 目标：用统一口径输出“到达率 / hit-wall / Jaccard / 采样带来的 any-success”等核心指标（比只看 `val_acc` 更接近真实可用性）。

```bash
# 评估矩阵：gt vs flow；greedy vs beam
python -m src.evaluation.way_casd_decision_eval \
  --way_routes_npz "$WAY_ROUTES_NPZ" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --flow_ckpt "$OUT_BASE/W7_train_flow/ckpt_best.pt" \
  --out_json "$OUT_BASE/W8_diag/decision_eval_n200.json" \
  --latent_sources gt flow \
  --decode_methods greedy beam \
  --n_routes 200 --n_samples_per_route 10 \
  --beam_size 10 \
  --decode_max_candidates 0 --decode_candidate_policy first \
  --max_way_len 160 --max_decode_len 160 \
  --device cuda --seed 0 \
  |& tee "$OUT_BASE/W8_diag/run_decision_eval.log"
```

输出：
- `W8_diag/decision_eval_n200.json`：按城市汇总的 success/hit_wall/len_ratio/jaccard 等

**(可选 Step C) Execution Stage：GPS-level 条件扩散（依赖 segments_with_wayid.parquet）**

> 输入是 `segments_with_wayid.parquet`（包含 `y/x/t/osm_way_id`），用 Decision AE 产生 `skeleton_latent` 做条件扩散，输出固定长度 `traj_len` 的轨迹（相对起点）。

```bash
# Detroit + Columbus（Rust Belt）一起训练 execution
python -m src.training.train_way_casd_gps_diffusion \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
                    "$RAW_ROOT/worldtrace/columbus_core_v1/segments_with_wayid.parquet" \
  --route_city 0 1 \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --out_dir "$OUT_BASE/W9_train_exec" \
  --batch_size 128 --num_workers 48 --n_epochs 60 \
  --traj_len 256 --max_way_len 128 --prefer_matched \
  --d_model 256 --n_latent 64 --hidden_dim 128 --emb_dim 512 \
  --diffusion_steps 100 --prediction_type eps --skel_noise_sigma 0.1 \
  --device cuda

# 最小可视化（单城一次画几条）
python -m src.evaluation.way_casd_gps_sample_viz \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
  --route_city 0 \
  --semantic_dir "$RAW_ROOT/worldtrace/detroit_core_v1" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --exec_ckpt "$OUT_BASE/W9_train_exec/ckpt_best.pt" \
  --out_dir "$OUT_BASE/W10_exec_viz" \
  --n_routes 8 --n_samples_per_route 4 --traj_len 256 --prefer_matched \
  --device cuda
```

**(指标评估) Execution Stage：GPS 轨迹（micro + distribution + on-road）**

```bash
python -m src.evaluation.way_casd_exec_eval \
  --segments_parquet "$RAW_ROOT/worldtrace/detroit_core_v1/segments_with_wayid.parquet" \
                    "$RAW_ROOT/worldtrace/columbus_core_v1/segments_with_wayid.parquet" \
  --route_city 0 1 \
  --semantic_dir "$RAW_ROOT/worldtrace/detroit_core_v1" \
                 "$RAW_ROOT/worldtrace/columbus_core_v1" \
  --way_graph_npz "$WAY_GRAPH_NPZ" \
  --way_features_npz "$WAY_FEATS_NPZ" \
  --ae_ckpt "$OUT_BASE/W6_train_ae/ckpt_best.pt" \
  --exec_ckpt "$OUT_BASE/W9_train_exec/ckpt_best.pt" \
  --out_json "$OUT_BASE/W11_exec_eval/exec_eval_n512.json" \
  --n_routes 512 --n_samples_per_route 4 --traj_len 256 --prefer_matched \
  --batch_routes 16 --frechet_points 64 \
  --device cuda --seed 0 \
  |& tee "$OUT_BASE/W11_exec_eval/run_exec_eval.log"
```

**(2) segments→graph paths（map-match，T1）**：

```bash
# 推荐：先建立工作站别名目录（软链接），避免写长路径/文件名漂移
python tools/routegen_make_ws_aliases.py --raw_root "$RAW_ROOT"

python -m src.data.road_graph.dump_graph_paths_from_routes_npz \
  --routes_npz "$RAW_ROOT/experiments/icml2026_routegen/gt_segments/detroit_segments_route_F256_epoch_seed0.npz" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/G1n_roadgraph_detroit_native/road_graph.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/G3_argraph_detroit_seed0/T1_dump_paths" \
  --num_workers 24 --chunk_size 256 --mp_start fork \
  --progress auto --log_every 2000 \
  --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/G3_argraph_detroit_seed0/T1_dump_paths/run.log"

# 快速审计：node/edge 序列长度分布
python -m src.data.road_graph.audit_paths_graph_npz \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/G3_argraph_detroit_seed0/T1_dump_paths/paths_graph.npz"
```

**(3) dump waypoints（T4 Step-1）**：

```bash
python -m src.data.road_graph.dump_waypoints_from_paths_graph_npz \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints" \
  --num_waypoints 4 --mode rdp_turn --turn_alpha 1.0 \
  --progress json --log_every 200 --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/run.log"
```

如果要把 waypoint 从“几何转折点（RDP）”切换为“图上分叉点（degree$\ge$3）”，可用：

```bash
python -m src.data.road_graph.dump_waypoints_from_paths_graph_npz \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints_branch_thr3" \
  --num_waypoints 4 --mode branch --branch_degree_thr 3 \
  --progress json --log_every 200 --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints_branch_thr3/run.log"
```

**(4) 训练 waypoint AR（T4 Step-2）**：

```bash
python -m src.training.train_graph_ar_waypoint_bins \
  --waypoints_npz "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/waypoints_graph.npz" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T2_train_wp_ar_bin_seed0" \
  --wp_bin 32 --hidden_dim 256 --batch_size 512 --epochs 200 --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T2_train_wp_ar_bin_seed0/run.log"
```

**(5) 采样 + A* 连接（T4 Step-3，可并行 A*）**：

```bash
python -m src.training.sample_graph_ar_waypoints_astar \
  --checkpoint "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T2_train_wp_ar_bin_seed0/last.pt" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz" \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz" \
  --waypoints_npz "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/waypoints_graph.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T3_eval_wp_ar_astar_K20_seed0" \
  --K 20 --temperature 0.8 --num_routes 200 --viz_cases 10 \
  --viz_gt_od_bin 128 --viz_gt_max 50 \
  --pick_strategy tier_dir \
  --astar_workers -1 --progress json --log_every 10 --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T3_eval_wp_ar_astar_K20_seed0/run.log"
```

> 说明：该脚本会自动按 `route_city` 过滤 bin→node 候选（避免多城市 graph 混选导致 A* 大量失败），并默认使用 `--pick_strategy tier_dir`（主干道优先 + OD 方向一致）来实例化 bin 内节点。

**(6) Oracle 上界（T4 Step-3b，判定瓶颈在决策层还是执行层）**：

```bash
python -m src.training.sample_graph_ar_waypoints_astar \
  --checkpoint "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T2_train_wp_ar_bin_seed0/last.pt" \
  --road_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/road_graph_combo.npz" \
  --paths_graph_npz "$RAW_ROOT/experiments/icml2026_routegen/T3_combo_detroit_columbus_seed0/paths_graph_combo.npz" \
  --waypoints_npz "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T1_dump_waypoints/waypoints_graph.npz" \
  --out_dir "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T3_eval_wp_ar_astar_K20_seed0_oracle" \
  --oracle --K 20 --temperature 0.8 --num_routes 200 --viz_cases 10 \
  --viz_gt_od_bin 128 --viz_gt_max 50 \
  --pick_strategy tier_dir \
  --astar_workers -1 --progress json --log_every 10 --seed 0 \
  |& tee "$RAW_ROOT/experiments/icml2026_routegen/T4_wp_ar_astar_combo_seed0/T3_eval_wp_ar_astar_K20_seed0_oracle/run.log"
```

---

## 7) 代理与网络（Wayback / Census 常见问题）

### 7.1 Wayback（ArcGIS）SSL hostname mismatch

现象：不挂代理时访问 `https://wayback.maptiles.arcgis.com/...` 可能出现证书 hostname mismatch。  
实践口径：**在运行 Wayback 下载前显式设置代理**（以 Clash 为例）：

```bash
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
```

可用性探测（最小 smoke test：只扫一个小 bbox/少量 tile；确认能下载到 jpg）：

```bash
python -m src.data.wayback.download_wayback_tiles \
  --out_dir "$RAW_ROOT/wayback/_probe_z16_r4756" \
  --bbox -83.06 42.32 -83.03 42.34 \
  --zoom 16 --max_threads 4 --max_tiles 10 \
  --mode fixed_releases --release_ids 4756
find "$RAW_ROOT/wayback/_probe_z16_r4756" -name "*.jpg" | wc -l
```

> Wayback 的下载/落盘约定详见 `docs/WAYBACK.md`。

### 7.2 Census TIGER 403

部分环境/时段会遇到 `www2.census.gov` 的 `403 Forbidden`。建议：

- 保留脚本自动下载的 ACS 指标表（一般可用）
- TIGER 边界若自动下载失败，可手动下载 zip 并转换为 GeoParquet（手动流程只做一次，落到 `$RAW_ROOT/census/...` 即可）

---

## 8) 大文件与 git（避免 push 崩盘）

- `$RAW_ROOT/` 下的大文件（WorldTrace zip / OSM pbf / SafeGraph shards / Wayback tiles）**绝不进入 git**
- legacy 深圳原始包（rar/大目录）也不要进 git；若误加入会触发 GitHub 大文件限制
- 仓库只提交：代码、文档、配置、小体量图表/JSON

---

## 10) 深圳数据封存（legacy，当前不参与分析）

当前主线是 **WorldTrace ×（Detroit/参考城市）**。仓库中的深圳相关目录仅作为历史复现材料封存：

- `legacy/shenzhen/`：保留旧实验与清洗口径的可复现入口

约束（必须遵守）：

- 任何 Detroit story 的图/表/结论 **不得引用深圳实验产物**（避免叙事口径混用）
- 新的分析脚本默认只读 `$RAW_ROOT/worldtrace/...`，不要把路径写回 `legacy/shenzhen/`

---

## 9) 与现有文档的关系（避免口径打架）

- 数据版本/坐标/时间：`docs/DATA_CONTRACT.md`
- 数据目录与字段：`docs/DATA_STRUCTURE.md`
- 当前主线路线图：`docs/PHASE_D_ROADMAP_OSM_TOPO_SEMANTICS.md`
- Wayback 下载细节：`docs/WAYBACK.md`
