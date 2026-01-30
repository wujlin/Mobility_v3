回复 Partner 的三个确认点
Q1: 语义最小集合是什么？
PI 建议：先用 时间 + tier-road，作为最小可行集

Feature	为什么 include	信息量来源
时间 t	必须 —— 这是论文 "temporal-spatial conditioning" 的 temporal 部分	hour-of-day, day-of-week 会影响路线选择（rush hour 避拥堵）
tier-road	必须 —— 这是最直接的 structural semantics	人会偏好/回避某些道路等级
POI/landuse 作为 Phase 2 增强，原因：

增加 feature 会增加 engineering 复杂度
先用最小集跑通 Gate，证明 "context matters"
如果 tier+time 就能 pass Gate，说明 claim 成立；如果不行，再加 POI
具体编码建议：

# 时间编码
hour_of_day = (t % 86400) / 86400  # [0, 1]
day_of_week = ((t // 86400) % 7) / 7  # [0, 1]
is_weekend = 1 if day_of_week >= 5 else 0

# Tier-road 编码（在分叉点）
# 对于分叉的两个方向，分别统计前 N 步的 average tier
tier_branch_A = mean([tier(edge) for edge in branch_A[:N]])
tier_branch_B = mean([tier(edge) for edge in branch_B[:N]])
tier_diff = tier_branch_A - tier_branch_B

Q2: 多城市策略？
PI 建议：先单城市（Detroit）跑通全 pipeline，再扩展

策略	优点	风险
先单城 ✓	快速验证架构；debug 成本低	无法验证 generalization
立刻多城	能验证 cross-city	工程复杂度翻倍；一个城市的 bug 会阻塞另一个
具体建议：

Phase 1 (本周)：Detroit only，跑通 T1→T2→T3
Phase 2 (下周)：如果 Detroit Gate 通过，再加 Columbus
论文写法：即使只有 Detroit 结果，也可以写 "We demonstrate on Detroit; cross-city transfer is future work"
如果必须要多城市结果（reviewer 可能问 generalization）：

两个城市用 独立的 node embedding（不共享 node-id）
共享 GNN/AR/Context Encoder 的参数
这样可以测试 "架构是否 transfer"，而不是 "embedding 是否 transfer"
Q3: Gate 阈值？
PI 建议：AUC > 0.6 作为 Go，AUC > 0.7 作为 Strong Go

AUC	含义	决策
< 0.55	接近随机	No-Go：context 在分叉点基本无信息量
0.55 - 0.6	弱信号	Conditional Go：需要增强 features
0.6 - 0.7	中等信号	Go：可以开始 AR training
> 0.7	强信号	Strong Go：claim 有很强的实验支撑

理由：

0.6 意味着 context 能区分 60% 的分叉选择 —— 虽然不完美，但说明 context matters
论文 claim 是 "context-conditioned diversity"，不是 "context perfectly predicts choice"
如果 AUC = 0.65，可以诚实写 "We observe moderate but significant predictive signal (AUC=0.65)"
Gate 报告建议额外输出：

{
  "auc_overall": 0.67,
  "auc_by_feature": {
    "time_only": 0.58,
    "tier_only": 0.61,
    "time+tier": 0.67
  },
  "n_branch_points": 1234,
  "n_samples": 5678,
  "conclusion": "GO"
}

这样可以看到每个 feature 的边际贡献。

对 Implementation Plan 的补充建议
Partner 的 T1-T4 设计合理，补充几点：

T1 补充：Map-matching 的 fallback 策略

# 可能出现的问题：GT 点 snap 到的 node 不相邻
# 建议：跳跃时用 A* 补齐，但记录补齐率

if not graph.has_edge(prev_node, curr_node):
    # 尝试 A* 补齐
    bridging_path = astar(graph, prev_node, curr_node)
    if bridging_path:
        node_seq.extend(bridging_path[1:])  # 跳过 prev_node
        stats["bridged_jumps"] += 1
    else:
        # 补齐失败：标记为 bad sample
        stats["failed_matches"] += 1
        continue

质量红线：如果 failed_matches / total > 10%，说明 graph 或 snap 逻辑有问题，需要先 debug。

T2 补充：分叉点定义

def find_first_branch_point(node_seq_A, node_seq_B):
    """
    找到两条 GT path 第一次分叉的 node。
    """
    for i, (a, b) in enumerate(zip(node_seq_A, node_seq_B)):
        if a != b:
            # 分叉发生在 i-1 位置
            return node_seq_A[i-1] if i > 0 else node_seq_A[0]
    return None  # 两条路径完全相同

T3 补充：Training 的 early stopping

# 建议用 validation set 的 next-node accuracy 做 early stopping
# 而不是 training loss（避免 overfitting）

最终确认
问题	PI 决策
Q1: 语义最小集	时间 (hour, dow, is_weekend) + tier-road，POI 作为 Phase 2
Q2: 多城市	先 Detroit only，跑通后再加 Columbus
Q3: Gate 阈值	AUC > 0.6 = Go，AUC > 0.7 = Strong Go
Partner 可以开始 T1 + T2。

T1 完成后先跑一下 map-matching 质量统计（跳跃率、补齐失败率），确认 < 10% 再进入 T2。

