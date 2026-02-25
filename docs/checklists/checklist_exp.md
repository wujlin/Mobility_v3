两个候选：

模型	SR (K16)	质量	备注
P1 stepemb e100	64.8%	succ_lenR_p50 = 1.887 (dest_efficient)	当前质量最优
RL dense sched e20	67.5%	未用 dest_efficient 评估	当前 SR 最优
RL dense sched SR 更高（+2.7pp），但它的质量指标从未用 dest_efficient 口径评估过，所以不确定谁是综合 SOTA。两个都应该跑 OD-level eval。

Prompt 给 Partner
任务：OD-level 多模态评估 + 补齐缺失 eval

背景
我们需要回答一个核心问题：Way-CASD 生成的 K=16 条路径，能否覆盖真实 OD 的多模态走廊分布？ 这决定了论文的 contribution 定位。工具 od_coverage_diversity_eval.py 已经实现，但从未在主线模型上运行过。

需要做的事（按优先级）
1. 补齐 RL dense sched 的 dest_efficient 评估（30分钟）
RL dense sched 从未用 dest_efficient 口径评估。请跑：
python -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz $WAY_ROUTES \
  --way_graph_npz $WAY_GRAPH \
  --way_features_npz $WAY_FEATURES \
  --ae_ckpt _sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/ckpt_best.pt \
  --flow_ckpt _sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/ckpt_best.pt \
  --way_regions_npz $WAY_REGIONS \
  --split_json $SPLIT_JSON \
  --split_part test \
  --n_routes 5000 --K 16 --seed 0 \
  --sample_select dest_efficient \
  --dump_way_seqs \
  --out_json _sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/eval/binned_rl_k16_dest_efficient_n5000.json

  注意 --dump_way_seqs——OD-level eval 需要 gt_way_ids 和 pred_way_ids。

2. 重跑 P1 e100 的 dest_efficient 评估（带 dump_way_seqs）（30分钟）
之前的 per_route JSON 可能没有 gt_way_ids/pred_way_ids。需要确认，如果没有则重跑：

python -m src.evaluation.way_casd_binned_eval \
  --way_routes_npz $WAY_ROUTES \
  --way_graph_npz $WAY_GRAPH \
  --way_features_npz $WAY_FEATURES \
  --ae_ckpt _sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/ckpt_best.pt \
  --flow_ckpt _sync/wsa/pi_verify/20260212_porto_flow_xattn_regionseq_dev10p_s0/ckpt_best.pt \
  --way_regions_npz $WAY_REGIONS \
  --split_json $SPLIT_JSON \
  --split_part test \
  --n_routes 5000 --K 16 --seed 0 \
  --sample_select dest_efficient \
  --dump_way_seqs \
  --out_json _sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/eval/binned_e100_k16_dest_efficient_n5000_wayseqs.json

  3. OD-level 多模态评估（核心）
用步骤 1、2 产出的带 way_seqs 的 per_route JSON，跑 OD-level eval：

python -m src.evaluation.od_coverage_diversity_eval \
  --method "P1_E100|greedy=_sync/wsa/pi_verify/20260214_porto_p1_stepemb_cont_e100_s0/eval/per_route_e100_k16_dest_efficient_n5000_wayseqs.json" \
  --method "RL_Dense|greedy=_sync/wsa/pi_verify/20260216_porto_rl_dense_sched09to03_e20_freshE100_from_e100_s0/eval/per_route_rl_k16_dest_efficient_n5000.json" \
  --out_json _sync/wsa/pi_verify/od_coverage_diversity_p1_vs_rl_k16.json \
  --k 16 \
  --min_routes_per_od 3 \
  --coverage_threshold 0.5 \
  --save_per_od_details

  如果 per_route JSON 的路径与上面不完全对应，请按 binned_eval 实际产出的 per_route_*.json 文件名调整。

4. Baseline OD-level eval（如果时间允许）
对 RNN-AR 和 Transformer-AR baselines 也跑同样的 K=16 dest_efficient + OD-level eval。需要确认 baseline 的 eval 脚本是否支持 --dump_way_seqs，如果不支持需要适配。

数据路径（Porto 全套）
EXP_ROOT="/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen"
DATA_BASE="${EXP_ROOT}/WAYCASD0_waydata_porto_seed0"
WAY_ROUTES="${DATA_BASE}/W5_way_routes_strict_gate/way_routes_strict_gate.npz"
WAY_GRAPH="${DATA_BASE}/W2_way_graph/way_graph.npz"
WAY_FEATURES="${DATA_BASE}/W3_way_features/way_features.npz"
WAY_REGIONS="${DATA_BASE}/region_sweep/way_regions_louvain_res5_seed0.npz"
SPLIT_JSON="${DATA_BASE}/od_split_min5_max160_seed0_dev10p.json"
关键输出指标
跑完后请汇报：

指标	P1 e100	RL dense	含义
SR (K16 dest_efficient)	64.8%	？	到达率
succ_lenR_p50	1.887	？	成功样本路径质量
GT Coverage@16	？	？	16 条生成路径覆盖了多少 GT 走廊
Self-Diversity@16	？	？	16 条生成路径之间的多样性
n_od_evaluated	？	？	有效 OD 组数
GT Coverage@16 和 Self-Diversity@16 是本轮最重要的数字。

成功判据
GT Coverage@16 > 0.5 → 方法有效地捕获多模态走廊
Self-Diversity@16 > 0.4 → 生成路径不是 mode collapse
如果两个都满足，论文的核心叙事可以围绕"多模态路径生成"展开> 