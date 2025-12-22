Implementation Plan, Task List and Thought in Chinese：本文件说明如何把 v1 的 grid 轨迹映射回经纬度并出“地图式”可视化图件，用于 essay/paper（强调：v1 不做 map-matching，地理映射为 bbox 线性近似）。

# 地图可视化（Phase B, dt-fixed=30s）

## 1) 我们能做什么（v1 的边界）

- v1 的数据与模型输出在 **grid 坐标系**（`[y, x]`，单位为 grid cell / step）。
- `data/processed_dt30/data_stats.json` 里记录了深圳 bbox：
  - `min_lat/max_lat/min_lon/max_lon` 与 `H/W`
- 因此可以做一个 **线性投影**：把 `[y, x]` 映射回 `[lat, lon]`，用于地理空间展示。

> 注意：这不是 road-level map matching，只是为了把轨迹放回“城市地理空间”去解释空间异质性与复杂动力学。

---

## 2) 一键出图（基于 evaluate 保存的 samples.npz）

你需要先确保 quick eval 目录里有 `samples.npz`：
- `data/experiments/baseline_b_dt30_eval_quick/samples.npz`
- `data/experiments/diff_b_dt30_eval_quick/samples.npz`
- `data/experiments/physics_b_dt30_eval_quick/samples.npz`

然后运行（推荐：使用可重复的 `--sample "Label:Path"`，避免标签误导）：

```bash
python -m src.visualization.plot_geo_phase_b \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Baseline:data/experiments/baseline_b_dt30_eval_quick/samples.npz" \
  --sample "Diffusion:data/experiments/diff_b_dt30_eval_quick/samples.npz" \
  --sample "Physics:data/experiments/physics_b_dt30_eval_quick/samples.npz" \
  --out_dir data/experiments/phase_b_report/figures_geo_quick \
  --num_trajs 80 \
  --bins 220 \
  --seed 0
```

输出：
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_traj_overlay.(png|pdf)`
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_density.(png|pdf)`

> 兼容说明：脚本仍保留旧参数 `--baseline_samples/--diff_samples/--physics_samples`，但已标记为 deprecated。

## 2.3（v1.1 Residual）Prior + Residual 的地图证据

v1.1 的讲故事重点是：“prior 负责尺度与主走廊；residual 负责多模态偏离”。因此建议至少给两张图：

- 轨迹叠图（GT + 多条预测）
- 预测密度图（Pred heatmap + GT contour）

当前仓库已包含一份可直接用于 PPT/essay 的产物：

- `data/experiments/residual_report/figures_geo/fig_geo_traj_overlay.(png|pdf)`
- `data/experiments/residual_report/figures_geo/fig_geo_density.(png|pdf)`
- 同步拷贝到 `essay/figures/fig_geo_traj_overlay_v11.(png|pdf)`、`essay/figures/fig_geo_density_v11.(png|pdf)`（用于 `essay/slides.tex`）

若要重新生成（需要先在 eval 时保存 `samples.npz`）：

1) 在残差模型 eval 时加 `--save_samples 200`（示例）：

```bash
python -m src.training.evaluate \
  --exp_name diff_dt30_residual_priorB_eval_test_vis \
  --model_type diffusion \
  --data_path data/processed_dt30/trajectories/shenzhen_trajectories.h5 \
  --checkpoint data/experiments/diff_dt30_residual_priorB_h128_b2048_lr1e-3_e100_s0/last.pt \
  --prior_checkpoint data/experiments/baseline_b_dt30/last.pt \
  --split test \
  --obs_len 8 --pred_len 12 \
  --num_samples_per_condition 20 --diff_steps 100 \
  --save_samples 200 --seed 0
```

2) 然后用 `plot_geo_phase_b` 把 `samples.npz` 投影到经纬度（与 v1.0 完全同一套路）。

## 2.1（强烈建议加到 essay）OD 热点图（Origin/Destination）

这张图非常“城市科学友好”，直观展示空间异质性（哪里是上车热点/下车热点）：

```bash
python -m src.visualization.plot_od_hotspots \
  --data_path data/processed_dt30/trajectories/shenzhen_trajectories.h5 \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --out_dir data/experiments/phase_b_report/figures_geo_quick \
  --bins 240 --max_trajs 200000 --seed 0
```

输出：
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_od_hotspots.(png|pdf)`

## 2.2（解释 physics 很好用）Nav Field 地图

这张图用来解释 physics-conditioned diffusion 的先验：train-only mean-flow prior。

```bash
python -m src.visualization.plot_nav_field_geo \
  --nav_file data/processed_dt30/nav_field.npz \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --out_dir data/experiments/phase_b_report/figures_geo_quick \
  --stride 18
```

输出：
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_nav_field.(png|pdf)`

### 若发现南北翻转（lat 方向反了）

加上 `--flip_y`：

```bash
python -m src.visualization.plot_geo_phase_b ... --flip_y
```

---

## 2.4（CFG）micro–macro 旋钮图（Pareto）

这张图用来支撑核心叙事：CFG 是 **推理期可控旋钮**，而不是参数地狱。我们通常固定两点：
- `cfg=2`：micro-optimal（主表）
- `cfg=3`：macro-validity-optimal（附图/讨论）

运行：

```bash
python -m src.visualization.plot_cfg_pareto \
  --glob "data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0_val_k10_mb200_cfg*/metrics.json" \
  --out_dir essay/figures \
  --style paper
```

输出：
- `essay/figures/fig_cfg_pareto.(png|pdf)`

图中右轴会画 `y=1` 的虚线作为 validity gate（pred/GT=1）。

## 3) 论文/essay 的 caption 建议（务必诚实）

由于 `samples.npz` 通常只保存 `--save_samples` 的子集（例如 200 条 condition），图注建议写清楚：

- “We visualize a subset of saved evaluation samples (N=200 conditions) in geographic space via a linear bbox projection.”
- 或中文：“图中为评估阶段保存的样本子集（N=200 条 condition），通过 bbox 线性映射回经纬度用于展示。”

这样审阅者/老师不会误解为“全量密度”。
