Implementation Plan, Task List and Thought in Chinese：本文件记录 Phase B（CFG 目的地引导）阶段的“子刊级地理可视化”证据链与一键出图命令，避免重复踩坑。

# Phase B（CFG Destination Guidance）可视化与证据链

## 1) 目标（写在图注/汇报里的三句话）

1. **Residual+Physics** 把宏观尺度锚定在合理区间（MSD/Rog/Speed 接近 GT），避免 shrinkage。
2. **CFG 是推理期可控旋钮**：`cfg=2` 偏 micro（best-of-K 更优），`cfg=3` 偏 macro-validity（MSD/Rog 更贴近或略过冲）。
3. 在深圳地理空间上，模型不仅“误差更小”，还能生成更符合真实城市走廊结构的多模态轨迹集合。

> PI 建议（caption 必写）：  
> “cfg=2 is micro-optimal within macro validity gate; cfg=3 demonstrates the model can be tuned towards macro-validity at the cost of micro-precision.”

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
  --out_dir essay/figures/stage_cfg \
  --num_trajs 80 \
  --bins 220 \
  --seed 0 \
  --extent data \
  --pad_frac 0.08 \
  --style paper
```

输出：
- 默认：`(png|pdf)` 两种格式都会输出；
- 如需只输出 PNG：追加 `--png_only`。

可选增强：
- 加区名标签：`--basemap_labels --basemap_label_size 8`
- 默认区名标签为英文（避免 CJK 字体缺失告警）；如需保留 GeoJSON 原始中文：`--basemap_label_lang raw`
- 若发现南北翻转：`--flip_y`

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
  --frames_only \
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
