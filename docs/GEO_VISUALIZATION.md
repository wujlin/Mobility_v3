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

然后运行：

```bash
python -m src.visualization.plot_geo_phase_b \
  --stats_path data/processed_dt30/data_stats.json \
  --baseline_samples data/experiments/baseline_b_dt30_eval_quick/samples.npz \
  --diff_samples data/experiments/diff_b_dt30_eval_quick/samples.npz \
  --physics_samples data/experiments/physics_b_dt30_eval_quick/samples.npz \
  --out_dir data/experiments/phase_b_report/figures_geo_quick \
  --num_trajs 80 \
  --bins 220 \
  --seed 0
```

输出：
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_traj_overlay.(png|pdf)`
- `data/experiments/phase_b_report/figures_geo_quick/fig_geo_density.(png|pdf)`

## 2.1（强烈建议加到 essay）OD 热点图（Origin/Destination）

这张图非常“城市科学友好”，直观展示空间异质性（哪里是上车热点/下车热点）：

```bash
python -m src.visualization.plot_od_hotspots \
  --data_path data/processed_dt30/trajectories/shenzhen_trajectories.h5 \
  --stats_path data/processed_dt30/data_stats.json \
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

## 3) 论文/essay 的 caption 建议（务必诚实）

由于 `samples.npz` 通常只保存 `--save_samples` 的子集（例如 200 条 condition），图注建议写清楚：

- “We visualize a subset of saved evaluation samples (N=200 conditions) in geographic space via a linear bbox projection.”
- 或中文：“图中为评估阶段保存的样本子集（N=200 条 condition），通过 bbox 线性映射回经纬度用于展示。”

这样审阅者/老师不会误解为“全量密度”。
