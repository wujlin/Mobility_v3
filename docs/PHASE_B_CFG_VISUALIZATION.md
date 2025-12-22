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
> 说明：当前 `samples.npz` 默认只保存 `k=0` 的一条采样（足够画城市级密度与 overlay）。

在工作站 A（GPU）执行：

```bash
DATA=data/processed_dt30/trajectories/shenzhen_trajectories.h5
NAV=data/processed_dt30/nav_field.npz
PRIOR=data/experiments/baseline_b_dt30/last.pt
CKPT=data/experiments/phys_residual_cfgp0.1_predeps_e20_mb200_s0/last.pt

# Prior（deterministic, K=1）
python -m src.training.evaluate \
  --exp_name prior_geo_viz_test \
  --model_type baseline \
  --data_path $DATA --checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --max_batches 200 --save_samples 400 --seed 0

# Physics residual + CFG2（主表）
python -m src.training.evaluate \
  --exp_name phys_cfg_geo_viz_test_cfg2 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 20 --diff_steps 100 \
  --cfg_scale 2 --save_samples 400 --max_batches 200 --seed 0

# Physics residual + CFG3（附图/讨论）
python -m src.training.evaluate \
  --exp_name phys_cfg_geo_viz_test_cfg3 \
  --model_type physics \
  --data_path $DATA --nav_file $NAV \
  --checkpoint $CKPT --prior_checkpoint $PRIOR \
  --split test --batch_size 256 --num_workers 8 \
  --num_samples_per_condition 20 --diff_steps 100 \
  --cfg_scale 3 --save_samples 400 --max_batches 200 --seed 0
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
- `essay/figures/stage_cfg/fig_geo_traj_overlay.(png|pdf)`
- `essay/figures/stage_cfg/fig_geo_density.(png|pdf)`

可选增强：
- 加区名标签：`--basemap_labels --basemap_label_size 8`
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
- `essay/figures/stage_cfg/fig_cfg_pareto.(png|pdf)`

图中右轴自带 `y=1` 虚线（validity gate）。

---

## 6) 微观案例图（同一条件下对比多模型）

这张图强调 micro 行为差异：同一个 OD 条件下，GT vs Prior vs CFG2 vs CFG3 的局部轨迹形状差异。
（注意：`samples.npz` 默认仅保存 `k=0` 一条采样，因此此图用于“定性对比”，不是多模态 fan-out。）

```bash
python -m src.visualization.plot_geo_case_study \
  --stats_path data/processed_dt30/data_stats.json \
  --basemap_geojson geo_map/Shenzhen_county.geojson \
  --sample "Prior:data/experiments/prior_geo_viz_test/samples.npz" \
  --sample "CFG2:data/experiments/phys_cfg_geo_viz_test_cfg2/samples.npz" \
  --sample "CFG3:data/experiments/phys_cfg_geo_viz_test_cfg3/samples.npz" \
  --out_dir essay/figures/stage_cfg \
  --num_cases 9 --cols 3 --seed 0 --pad_frac 0.12 \
  --stem fig_geo_case_study_cfg \
  --style paper
```

输出：
- `essay/figures/stage_cfg/fig_geo_case_study_cfg.(png|pdf)`
