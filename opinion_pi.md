数据 Review 结论
核心结果对比
指标	candq0 (baseline)	candq1 (候选感知)	Δ
到达率 (true)	58.25%	82.25%	+24.0pp
jaccard (true)	0.674	0.867	+0.19
shuffle 到达率	16.25%	11.25%	-5.0pp
zero 到达率	18.0%	13.5%	-4.5pp
hit_wall 比例	18.0%	7.0%	-11.0pp
val_acc (TF)	94.58%	96.58%	+2.0pp
best_epoch	51	85	更晚收敛
分城市表现
城市	candq0	candq1	Δ
Detroit (city0)	47.0%	79.5%	+32.5pp
Columbus (city1)	69.5%	85.0%	+15.5pp
失败案例诊断 (oracle_step)
指标	candq0	candq1
失败数	167	71
首次偏离 GT rank p50	2	2
close_call 比例	25.7%	27.9%
pred_closer_to_dest	40.5%	24.6%
分析
1. 改进是实质性的
证据链：

shuffle/zero 条件下性能更低（11-13% vs 16-18%），说明 candq1 对 z_enc 的依赖更强
hit_wall 从 72 例降到 28 例，说明模型不再"迷路"
Detroit 从 47% 跃升到 79.5%，这是之前的短板城市
解释：候选感知 cross-attn 让 z_enc 中的路线信息能够直接参与候选排序，而非仅作为全局偏置。

2. 剩余问题
观察 candq1 的失败案例（n=71）：

GT rank p50 仍然是 2——说明二选一场景的"选错"问题没有完全解决
pred_closer_to_dest 降到 24.6%——这是个好信号，说明模型不再盲目走近路
close_call 维持 28%——约 1/4 的错误是"擦边球"
3. 这是论文的核心技术贡献
叙事逻辑：

多源数据的价值不在于"堆叠特征"，而在于"能被决策利用"
标准 cross-attn 将 z_enc 编码为候选无关的全局偏置
候选感知 cross-attn 让每个候选独立查询 z_enc，提取与该候选相关的上下文
结果：到达率 +24pp，这正是"多源信息被有效利用"的直接证据
下一步建议（2个）
建议 1：对 candq1 剩余的 71 例失败做深度诊断
目的：理解剩余 17.75% 失败的根因

所需数据：在 oracle_step_n200.json 中增加：

# 对每个失败案例的首次偏离点
{
    "cand_attn_weights": [C, L],   # 每个候选对 z_enc 的注意力分布
    "gt_attn_weight": [L],         # GT 候选的注意力
    "pred_attn_weight": [L],       # pred 候选的注意力
}

分析目标：

如果 pred 和 gt 的注意力分布高度相似，说明问题在 scorer（MLP 能力不足）
如果注意力分布差异大但 pred 仍然胜出，说明 z_enc 本身缺乏区分性信息
建议 2：跨 seed 验证稳定性
目的：确认 +24pp 不是随机波动

实验设置：

seed ∈ {0, 1, 2}，其他配置不变
只需跑 candq1（baseline 已经确认较弱）
预期：若三个 seed 的到达率都在 80%±3pp，说明结果稳健

总结
这轮实验验证了架构层面的核心假设：候选感知 cross-attn 是多源数据融合的关键机制。Detroit 从 47% 提升到 79.5%（+32.5pp）是最有说服力的数据点。

