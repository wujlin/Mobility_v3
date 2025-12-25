Implementation Plan, Task List and Thought in Chinese：本文件记录 Phase B（CFG 目的地引导）阶段的“子刊级地理可视化”证据链与一键出图命令，避免重复踩坑。

# Phase B（CFG Destination Guidance）可视化与证据链

## 0) 结论先行（止损线：避免再在 CFG/采样上浪费一周）

**我们已经用“可证伪”的诊断把方向拍板了**：`prior + diffusion + CFG` 在当前设定下**无法生成“低频、平滑 detour”**，主要表现为 **Destination Gravity（直冲终点）+ 直线段抖动/折点**。因此不要再期待通过调 `cfg/temperature/diff_steps` 来“创造绕路”。

关键证据（已在本 repo，可直接打开）：
- 轨迹叠图：`essay/figures/stage_cfg_density10k/fig_geo_traj_overlay.png`（Prior 与 CFG2 都被终点“吸住”，GT 有明显绕路/先逆行再上高速）
- 物理统计（JSD+DCV）：`essay/figures/physical_stats/fig_physical_stats_density10k_validity.json`、`essay/figures/physical_stats/fig_physical_stats_cfg2_vs_cfg3_validity.json`

**数值快照（越小越好；DCV 为超限率）**：

| Setting | JSD_Turn | JSD_Speed | JSD_Accel | DCV_speed | DCV_accel |
|---|---:|---:|---:|---:|---:|
| Prior（N=10k, K=1） | 0.0956 | 0.1423 | 0.2537 | 0.128% | 0.002% |
| Ours(CFG2)（N=10k, K=1） | 0.0503 | 0.0727 | 0.0531 | 0.580% | 0.733% |
| CFG2（N=400, use_all_k, k≤10） | 0.0738 | 0.1429 | 0.0646 | 1.098% | 1.238% |
| CFG3（N=400, use_all_k, k≤10） | 0.0728 | 0.1448 | 0.0559 | 1.239% | 2.060% |

解释（务实版）：
- CFG2/CFG3 在 `JSD_TurnAngle` 上差异极小，但 **CFG3 的 DCV 更差** → “把 cfg 拉大”不是解。
- `JSD_TurnAngle` 也可能被 **高频 jitter** “刷好看”，不等价于 detour；因此需要 Oracle 类诊断来确认 support（见第 8 节）。

> 工作站上我们已跑完 `OracleWP / OracleSel`，结论为：`JSD_TurnAngle` 不降反升（且 OracleSel 的偏离往往伴随更高 DCV）→ detour 基本不在 support。  
> 建议把工作站产物 rsync 回来（`samples.npz` + `<stem>_validity.json/png`），把证据链落在仓库路径上。

同步模板（示例；在本 repo 根目录执行）：

```bash
# 工作站 -> 本机（按你的 ssh alias/路径替换 wsA:/...）
rsync -avP --relative wsA:/home/jinlin/projects/Mobility_v3/./data/experiments/phys_cfg_geo_viz_test_cfg2_oracleWP/ ./data/experiments/
rsync -avP --relative wsA:/home/jinlin/projects/Mobility_v3/./data/experiments/phys_cfg_geo_viz_test_cfg2_oracleSel/ ./data/experiments/
rsync -avP --relative wsA:/home/jinlin/projects/Mobility_v3/./essay/figures/physical_stats/./fig_physical_stats_cfg2_oracle* ./essay/figures/physical_stats/
```

## 1) 目标（写在图注/汇报里的三句话）

1. **Residual+Physics** 把宏观尺度锚定在合理区间（MSD/Rog/Speed 接近 GT），避免 shrinkage。
2. **CFG 是推理期可控旋钮**，但它放大的是“终点势能梯度”：会影响抖动/尾部/违例率，**不能凭空产生 detour**（Destination Gravity 仍然存在）。
3. 本文档的核心产出是 **证据链与止损线**：用 `地理叠图 + JSD(Speed/Accel/Turn) + DCV + Oracle 诊断`，快速判断“还能不能救/是否必须换范式”。

> PI 建议（caption 必写）：  
> “CFG mainly amplifies the destination gradient; we use JSD+DCV as the validity gate and run oracle diagnostics to decide whether detour exists in the model support.”

---

## 2) 前置：把 exp 产物从服务器 B 同步到工作站 A（模板）

你需要确保工作站 A 上存在要出图的目录（至少 `metrics.json` + `samples.npz` 或可重新生成的 checkpoint）。

在服务器 B 执行（示例）：

```bash
# 服务器B -> 工作站A（示例路径；按实际替换）
rsync -avP data/experiments/<EXP_DIR>/  jinlin@10.13.12.164:/home/jinlin/projects/Mobility_v3/data/experiments/<EXP_DIR>/
```

若需走 socks5（EasyConnect），请在 ssh config 中配置 ProxyCommand（你们已有实践），这里不重复。

---

## 3) 生成可视化所需的 `samples.npz`（建议 N=400）

> 地理可视化依赖 `evaluate.py` 的 `--save_samples`，会在 `data/experiments/<exp_name>/samples.npz` 写入子集样本。
> 说明（重要）：为了展示 **多模态**，建议对生成模型开启 `--save_all_k`，额外保存 `preds_k (N,K,F,2)`。
> 这样第 6 节的微观案例图可以画出“轨迹束（spaghetti）”，而不是单条随机采样。

### 3.1 密度图的特殊要求（重要）

密度图是**宏观统计图**，对样本量极度敏感。经验上：
- `N=400` 只够做 **case study/面条图**；
- 密度图建议至少 `N>=10000`（否则会显得稀疏、像“点云”而非“流场”）。

因此建议你分两套样本：
- **Case study**：`save_samples=400`（带 `preds_k`，展示多模态分叉）
- **Density**：`save_samples=10000~20000`（`K=1` 即可，重点是样本量而不是多模态）

> 重要更新（省时间 + 断点续算）：  
> `evaluate.py` 新增 `--samples_only/--resume_samples/--sample_offset`：  
> - `--samples_only`：只生成 `samples.npz`，达到 `--save_samples` 就提前停止，不再计算 DTW/Frechet 等重指标；  
> - `--resume_samples`：断点续算（同一 `exp_name` 下已有 `samples.npz` 就从已有 N 继续补齐）；  
> - `--sample_offset`：跳过前 N 个 condition（用于分片并行：两张卡各跑一半，避免重复算）。  
> 
> 这三个参数只影响“出图采样”，不改变你论文/主表的评估口径。

在工作站 A（GPU）执行：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
CKPT=data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0/last.pt

# ===== 推荐：仅用于出图的“样本生成口径”（避免跑满 max_batches=200）=====
# 关键点：
# - 我们只需要 N=400 的对齐子集写入 samples.npz；
# - evaluate.py 仍会对“跑过的 batch”计算指标，所以 max_batches 应该尽量小：
#   max_batches ~= ceil(save_samples / batch_size)

BS=32
MB=13   # 32*13=416 >= 400

# Prior（deterministic, K=1）——为对齐子集，也用相同的 BS/MB
python -m src.training.evaluate \
  --exp_name prior_geo_viz_test \
  --model_type baseline \
  --data_path $DATA --checkpoint $PRIOR \
  --split test --batch_size $BS --num_workers 8 \
  --max_batches $MB --save_samples 400 --seed 0

# Physics residual + CFG2（主表）
python -m src.training.evaluate \
  --exp_name phys_cfg_geo_viz_test_cfg2 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size $BS --num_workers 8 \
  --num_samples_per_condition 20 --diff_steps 100 \
  --cfg_scale 2 --save_samples 400 --save_all_k --max_batches $MB --seed 0

# Physics residual + CFG3（附图/讨论）
python -m src.training.evaluate \
  --exp_name phys_cfg_geo_viz_test_cfg3 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size $BS --num_workers 8 \
  --num_samples_per_condition 20 --diff_steps 100 \
  --cfg_scale 3 --save_samples 400 --save_all_k --max_batches $MB --seed 0
```

### 3.2（推荐）Density 专用：1000→10000 断点续算（避免重复计算）

目标：先 10 分钟内看到“有没有流场质感”，确认没问题后再补到 10k。

> 建议：density 只需要 `K=1`，并且用 `--samples_only` 跳过重指标。

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
CKPT=data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0/last.pt

# --- (A) 先生成 1000 条 ---
python -m src.training.evaluate \
  --exp_name prior_geo_density_test \
  --model_type baseline --data_path $DATA --checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --save_samples 1000 --samples_only --seed 0

python -m src.training.evaluate \
  --exp_name phys_cfg2_geo_density_test \
  --model_type physics --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 1 --diff_steps 100 --cfg_scale 2 \
  --save_samples 1000 --samples_only --seed 0

# --- (B) 确认效果 OK 后，补到 10000（断点续算，不会重算前 1000）---
python -m src.training.evaluate \
  --exp_name prior_geo_density_test \
  --model_type baseline --data_path $DATA --checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --save_samples 10000 --samples_only --resume_samples --seed 0

python -m src.training.evaluate \
  --exp_name phys_cfg2_geo_density_test \
  --model_type physics --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 1 --diff_steps 100 --cfg_scale 2 \
  --save_samples 10000 --samples_only --resume_samples --seed 0
```

### 3.3（可选）两卡并行分片：0–4999 / 5000–9999（最快）

如果你有两张 GPU（服务器 B），可以用 `--sample_offset` 分片并行生成，再合并/出图：

```bash
# GPU0: 前 5000
CUDA_VISIBLE_DEVICES=0 python -m src.training.evaluate \
  --exp_name phys_cfg2_geo_density_shard0 \
  --model_type physics --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 1 --diff_steps 100 --cfg_scale 2 \
  --save_samples 5000 --samples_only --sample_offset 0 --seed 0 &

# GPU1: 后 5000
CUDA_VISIBLE_DEVICES=1 python -m src.training.evaluate \
  --exp_name phys_cfg2_geo_density_shard1 \
  --model_type physics --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 1 --diff_steps 100 --cfg_scale 2 \
  --save_samples 5000 --samples_only --sample_offset 5000 --seed 0 &

wait
```

分片合并建议（KISS）：先把两份 `samples.npz` 拷到同一机器，再用一个小脚本 `np.concatenate` 拼接成 `N=10000` 的 `samples_merged.npz` 再画密度图。

---

## 4) 子刊级地图出图（叠加深圳 geojson 边界）

本项目已提供深圳区县边界：`geo_map/Shenzhen_county.geojson`（WGS84 经纬度）。

```bash
python -m src.visualization.plot_geo_phase_b \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --overlay_keep Prior --overlay_keep CFG2 --overlay_keep CFG3 \
  --density_keep Prior --density_keep CFG2 \
  --density_sigma 1.6 \
  --out_dir essay/figures/stage_cfg \
  --num_trajs 80 \
  --bins 220 \
  --seed 0 \
  --extent data \
  --pad_frac 0.08 \
  --axis_off --scalebar_km 2 --min_span_km 10 \
  --style paper
```

输出：
- 默认：`(png|pdf)` 两种格式都会输出；
- 如需只输出 PNG：追加 `--png_only`。

> 口径提示（避免“图里又出现 CFG3”）：  
> `plot_geo_phase_b` 支持分别筛选两类输出：  
> - 轨迹叠图：用 `--overlay_keep` 指定要画哪些模型；  
> - 密度图：用 `--density_keep` 指定要画哪些模型（GT 面板永远保留）。  
> 
> 示例：如果你只想在宏观密度层面展示 `GT vs CFG2`（不画 Prior/CFG3），可以这样：
> 
> ```bash
> python -m src.visualization.plot_geo_phase_b \
>   --stats_path data/processed_dt30/data_stats.json \
>   --basemap_geojson geo_map/Shenzhen_county.geojson \
>   --sample "CFG2:data/experiments/phys_cfg2_geo_density_test/samples.npz" \
>   --overlay_keep CFG2 \
>   --density_keep CFG2 \
>   --out_dir essay/figures/stage_cfg_density10k \
>   --num_trajs 80 \
>   --bins 320 --density_sigma 1.6 \
>   --gt_contour_quantiles 0.90 0.96 0.99 \
>   --extent data --axis_off --scalebar_km 2 --min_span_km 10 \
>   --png_only --style paper
> ```
>
> 说明（避免误读）：
> - 标题里的 `n_plot=80/10000` 表示“绘制用的子集数量/总样本量”，不是训练集大小。
> - 密度图的 GT contour 默认改为 **分位数等值线**（高分位、少量线），避免低密度区域的“满屏条纹/底纹”。如需更少线，可只用 `0.95 0.99`。
> 
> 注意：轨迹叠图标题里的 `n_plot` 表示绘制子集大小（例如 60 条），不是“测试集总量”。如果样本文件里保存了 10000 条，则标题会显示 `n_plot=60/10000`。

可选增强：
- 加区名标签：`--basemap_labels --basemap_label_size 8`
- 默认区名标签为英文（避免 CJK 字体缺失告警）；如需保留 GeoJSON 原始中文：`--basemap_label_lang raw`
- 若发现南北翻转：`--flip_y`

> 重要细节（避免“看起来更密”其实是 K 放大）：  
> 如果 `samples.npz` 里包含 `preds_k (N,K,F,2)`，密度图会使用 K 条样本来估计分布，但会对直方图按 `1/K` 做归一化，  
> 使得密度反映的是“每个条件的一条采样轨迹的期望密度”，而不是把 trips 数量乘以 K。

---

## 5) CFG Pareto 旋钮图（micro vs macro）

把 cfg sweep 的 `metrics.json` 画成一张“旋钮图”，用于解释 trade-off：

```bash
python -m src.visualization.plot_cfg_pareto \
  --glob "data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0_val_k10_mb200_cfg*/metrics.json" \
  --out_dir essay/figures/stage_cfg \
  --style paper
```

输出：
- 默认：`(png|pdf)` 两种格式都会输出；
- 如需只输出 PNG：追加 `--png_only`。

图中右轴自带 `y=1` 虚线（validity gate）。

---

## 6) 微观案例图（同一条件下对比多模型）

这张图强调 micro 行为差异：同一个 OD 条件下，GT vs Prior vs CFG2 vs CFG3 的局部轨迹形状差异。
（若 samples.npz 含 `preds_k`，则会自动画 spaghetti 轨迹束，用于展示多模态分叉。）

> 小技巧（避免抽到“无聊样本”）：  
> 你可以先用第 7 节的 `select_interesting_cases.py` 找到分叉最明显的 `case_idx`，再把这些 idx 直接喂给 `plot_geo_case_study.py` 的 `--case_idx`（可重复多次）。

```bash
python -m src.visualization.plot_geo_case_study \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg \
  --num_cases 9 --cols 3 --seed 0 --pad_frac 0.12 \
  --k_plot 12 \
  --stem fig_geo_case_study_cfg \
  --style paper
```

例如（选 6 个最有分叉的 case）：

```bash
python -m src.visualization.plot_geo_case_study \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg \
  --num_cases 6 --cols 3 --pad_frac 0.12 \
  --case_idx 12 --case_idx 99 --case_idx 122 --case_idx 163 --case_idx 35 --case_idx 3 \
  --k_plot 12 --stem fig_geo_case_study_cfg_top \
  --png_only --style paper
```

输出：
- `essay/figures/stage_cfg/fig_geo_case_study_cfg.(png|pdf)`

---

## 6.1（建议主图）Storytelling Grid（3 行 Case × 3 列 Model）

> 这张图是“讲故事最强”的版本：  
> **Prior（死板均值）→ CFG2（多模态炸开）→ CFG3（更守规矩/可控收束）**。  
> 子刊写作/答辩时建议用这张替代“单面条堆叠”的大叠图。

```bash
python -m src.visualization.plot_geo_story_grid \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg \
  --rows 3 --k_plot 12 --min_span_km 3 --scalebar_km 1 \
  --stem fig_geo_story_grid \
  --png_only --style paper
```

若要手动指定 case（推荐用 `top_cases.csv` 的 idx）：

```bash
python -m src.visualization.plot_geo_story_grid \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg \
  --rows 3 --case_idx 191 --case_idx 190 --case_idx 192 \
  --k_plot 12 --min_span_km 3 --scalebar_km 1 \
  --stem fig_geo_story_grid_top \
  --png_only --style paper
```

---

## 7) 动画（强烈建议：用“轨迹束随时间展开”展示多模态）

> 动画是最直观的“杀手级证据”：同一 OD 条件下，Prior 是单条均值轨迹；CFG2/CFG3 会生成轨迹束并在关键路口分叉。
> 建议先用 `--frames_only` 导出 PNG 帧（最稳，零依赖），再用 ffmpeg 合成 mp4/gif。

先确保 `samples.npz` 里包含 `preds_k`（也就是 eval 时用了 `--save_all_k`）。

### 7.1 先自动挑选“最有分叉”的案例（避免随机抽到无聊样本）

```bash
python -m src.visualization.select_interesting_cases \
  --samples data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz \
  --top_k 10
```

输出 `case_idx` 后，把它喂给动画脚本的 `--case_idx`（下一段）。

### 7.2 生成动画帧（最稳）

```bash
python -m src.visualization.animate_geo_case \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg/anim \
  --stem anim_cfg_bundle \
  --case_idx 0 \
  --k_plot 12 --fps 6 --dpi 150 \
  --frames_only --encode_mp4 \
  --axis_off --scalebar_km 1 --min_span_km 3 \
  --style talk \
  --seed 0
```

> 若你看到经纬度轴出现 `+1.14e2` 之类的 offset/scientific 记法（不够子刊风格），请先 `git pull` 更新脚本：新版本已默认关闭 offset 记法。

合成视频（在输出帧目录内执行；示例）：

```bash
# mp4
ffmpeg -r 6 -i frame_%03d.png -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" anim.mp4

# gif（可选，较大）
ffmpeg -r 6 -i frame_%03d.png -vf "scale=960:-1:flags=lanczos" -loop 0 anim.gif
```

如果你环境里有 Pillow，也可以不加 `--frames_only` 直接输出 `.gif`（脚本会调用 PillowWriter）。

### 7.3 无 ffmpeg 的应急方案：用 HTML 直接播放帧（零依赖）

若你的环境没有 ffmpeg / Pillow（例如本地 WSL），可以直接把帧目录变成一个可播放的网页：

```bash
python -m src.visualization.make_html_animation \
  --frames_dir essay/figures/stage_cfg/anim/anim_cfg_bundle_frames_case122 \
  --fps 6 --title "CFG bundle (case 122)"
```

输出：`<frames_dir>/anim.html`，用浏览器打开即可播放/暂停/拖动帧。

---

## 8) 物理统计图（主证据：用于“救论文叙事”）

当地图密度/叠图难以区分 `Prior` 与 `CFG2`（宏观结构过于相似）时，不要硬拗“地图上更好看”。
这在统计上是正常现象：密度图是强低通滤波，Prior 的 conditional mean 往往也落在高密度走廊上。

更强的证据链应该切到 **Physical Statistics**：
- **Speed distribution**：Prior 往往更“尖”（均值回归），CFG2 会恢复更宽的速度分布（慢堵/快行的长尾）。
- **Turn-angle distribution**：Prior 更平滑（转角集中在 0），CFG2 会恢复转角纹理（更接近真实路口行为）。
- **MSD curve**：展示宏观位移尺度（directional persistence）。

进一步（对齐学术界 Generative Trajectory Prediction 的主流 validity 叙事）：
- **Statistical Consistency**：用 `JSD`（Jensen–Shannon Divergence）量化 Pred/GT 在 `Speed/Accel/TurnAngle` 三个分布上的差异；
- **Dynamic Constraint Violation (DCV)**：用速度/加速度的超限率做 feasibility 门槛（本仓库默认用 GT 的高分位作数据校准阈值，map-free 设定下更稳健）。

脚本：`src/visualization/plot_physical_stats.py`（无 SciPy 依赖；可选输出 `JSD+DCV`）

推荐做法：对生成模型使用 `samples.npz` 里的 `preds_k`（需要 eval 时 `--save_all_k`），
并用 `--use_all_k --k_max 10` 展示多模态下的物理统计纹理。

```bash
python -m src.visualization.plot_physical_stats \
  --inputs "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
           "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --use_all_k --k_max 10 \
  --turn_min_speed 0.1 \
  --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5 \
  --save_metrics \
  --stride 5 \
  --output_dir essay/figures/stage_cfg \
  --stem fig_physical_stats_cfg2
```

输出：
- 图：`<out_dir>/<stem>.png`
- 指标：`<out_dir>/<stem>_validity.json`（含 `JSD_Speed/JSD_Accel/JSD_TurnAngle` + `Vio_*`）

### 8.1) Oracle Waypoint（诊断实验：看“给个点能不能弯”）

如果你怀疑模型是 “straight line + jitter / destination gravity”，最强的控制变量实验是：
用 **GT 的中间点** 做 `stage-1 destination`，跑两段推理 `start→wp→d` 拼接（不重新训练），
然后看 `JSD_TurnAngle` 是否显著下降（更接近 GT）。

示例（以 physics+CFG2 为例；生成一份 OracleWP 的 `samples.npz`）：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
CKPT=data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0/last.pt

python -m src.training.evaluate \
  --exp_name phys_cfg_geo_viz_test_cfg2_oracleWP \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 32 --num_workers 8 \
  --num_samples_per_condition 20 --diff_steps 100 --cfg_scale 2 \
  --oracle_waypoint --oracle_waypoint_frac 0.5 \
  --save_samples 400 --save_all_k --max_batches 13 --seed 0
```

然后把 OracleWP 也喂给 `plot_physical_stats`，看 `TurnAngle` 分布与 `JSD_TurnAngle`：

```bash
python -m src.visualization.plot_physical_stats \
  --inputs "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
           "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
           "OracleWP:data/experiments/phys_cfg_geo_viz_test_cfg2_oracleWP/samples.npz" \
  --use_all_k --k_max 10 \
  --turn_min_speed 0.1 --save_metrics \
  --output_dir essay/figures/stage_cfg \
  --stem fig_physical_stats_cfg2_oracleWP
```

#### 8.1.1) 已证伪结论（务必止损，避免再浪费一周）

我们在相同评估口径下得到的结论是：

- **OracleWP 并未改善 “能不能弯/能不能绕”**：`JSD_TurnAngle` 不降反升（更偏离 GT）。  
- OracleWP 的 **DCV_Bound**（拼接点加速度违规）确实上升，但即便排除拼接点，`JSD_Turn` 仍更差 → **不是边界污染导致失败**。

因此可以拍板：**当前生成器的 support 里基本没有“低频、平滑 detour”模态；主要是直线段 + 高频抖动/折点。**  
继续在 `prior+diffusion+CFG` 这条线上调参（温度/噪声/cfg）不会“创造 detour”，只会改变尾部与 DCV。

> 证据入口（本地/工作站产物，需按机器同步）：  
> - `essay/figures/physical_stats/fig_physical_stats_density10k_validity.json`（Prior vs Ours(CFG2), N=10k, K=1）  
> - `essay/figures/physical_stats/fig_physical_stats_cfg2_vs_cfg3_validity.json`（CFG2 vs CFG3, N=400, use_all_k）  
> - `essay/figures/physical_stats/fig_physical_stats_cfg2_oracleWP_validity.json`（CFG2 vs OracleWP, N=400, use_all_k）  

### 8.2) Oracle Selection（更干净的诊断：support 里有没有 detour）

OracleWP 会引入 “两段拼接” 的新分布（可能带 jerk），为了更干净地回答：

> “detour 模态是否已经存在于 `preds_k` 的 support，只是我们没抽到？”

我们提供一个 **post-hoc oracle selection**（不重新采样、不改条件，只在保存的 K 条里选最像 GT detour 的那条）：

- 脚本：`src/visualization/oracle_waypoint_select.py`
- 输出：`<out_dir>/<stem>.npz`（oracle 选出的 `preds`） + `<out_dir>/<stem>.json`（JSD+DCV 对比）

示例（从 CFG2 的 `preds_k` 里选 “最大直线偏离” 的样本）：

```bash
python -m src.visualization.oracle_waypoint_select \
  --input_npz data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz \
  --out_dir data/experiments/phys_cfg_geo_viz_test_cfg2_oracleSel \
  --stem cfg2_oracleSel_maxdev \
  --k_max 10 \
  --waypoint_mode max_dev \
  --turn_min_speed 0.1 \
  --dcv_speed_pctl 99.5 --dcv_accel_pctl 99.5
```

判据（止损线）：
- 若 `oracle_selected.JSD_TurnAngle` **显著下降**（更接近 GT），说明 detour 模态在 support 里 → 未来可以考虑 inference-time rerank；
- 若 `oracle_selected.JSD_TurnAngle` **不降反升**（我们已观测到该情形），说明 detour 模态基本不存在 → **立刻停止在采样/CFG 上耗时，必须换范式**。

### 8.3) 结论与下一步（避免“走错路一周”）

结论（拍板）：
- `CFG` / `temperature` / “更强 residual” **不能解决 detour（Destination Gravity）**，最多改变抖动与尾部（并恶化 DCV）。
- 若目标是学术界认可的 trajectory validity（statistical consistency + feasibility），当前路线应当止损并 pivot。

下一步（KISS，推荐优先级）：
0. **先跑 Waypoint Gate（Go/No-Go）**：用 `src/evaluation/waypoint_gate.py` 做 skeleton 碰撞率 + 可学性检验，避免“同义反复”的验证；细节见 `docs/SOTA_TRAJECTORY_GENERATION_2025_UPDATE.md#4`。
1. **Hierarchical / Coarse-to-Fine（waypoint predictor + segment generator）**：把 detour 变成“中间目标”问题，而不是让一步式生成器自己学全局规划。
2. 只有当层级路线被证伪，才考虑 **road graph / map constraints**（工程量级更大）。

如果你只想做宏观密度专用的 10k 采样（K=1），也可以直接替换输入为：
`prior_geo_density_test/samples.npz` 与 `phys_cfg2_geo_density_test/samples.npz`，
但要注意：这会弱化“多模态优势”（因为 K=1）。
