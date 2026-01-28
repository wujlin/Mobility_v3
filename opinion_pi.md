深层架构分析
从代码可以确认打分流程：
ctx_t = _compute_context(cur_way, z_enc, ...)   # 候选无关
ctx_h = ctx_t.expand(T, C, -1)                   # 复制到每个候选

x = [ctx_h, cur_h, cand_h, dist]                # ctx_h、cur_h 对所有候选相同
logits = scorer(x)                              # 只有 cand_h、dist 能区分候选

核心发现：z_enc 的信息通过 cross-attention 被编码进 ctx_t，但 ctx_t 被 .expand() 复制到每个候选——z_enc 里的多源信息没有参与候选的排序。

这解释了所有现象：

z_enc 确实携带信息（true vs shuffle 有差异）
但在二选一分叉点选不对（因为区分候选只靠 cand_h 和 dist）
Direction Hint 帮助有限（它影响 ctx_t，但 ctx_t 仍然是候选无关的）
建议（2个）
建议 1：Candidate-Aware Cross-Attention
核心改动：让每个候选独立查询 z_enc，得到候选特定的上下文。

# 当前设计（候选无关）
query = cur_emb + step_emb + dest_proj + dir_hint + past_ctx  # (T, d)
ctx = cross_attn(query, z_enc)  # (T, d) → expand → (T, C, d)

# 改进设计（候选感知）
base_query = cur_emb + step_emb + dest_proj + past_ctx  # (T, d)
cand_query = base_query[:, None, :] + cand_proj(cand_emb)  # (T, C, d)
cand_ctx = cross_attn(cand_query.flatten(0,1), z_enc_expand)  # (T*C, d)
cand_ctx = cand_ctx.view(T, C, d)  # 每个候选有独立的上下文

为什么这是有效的：

z_enc 编码了完整路线的多源信息（语义、时空、OD）
当前架构让这些信息只作为"全局偏置"，无法区分候选
改进后，每个候选可以从 z_enc 提取"与自己相关"的信息
叙事契合：这正是"多源数据融合"的核心体现——不只是把多源信息塞进去，而是让模型能够利用这些信息做决策。

计算开销：cross-attention 从 T 次增加到 T×C 次（C 平均 2-4），可接受。

建议 2：诊断数据增强
在 oracle_step_diagnose.npz 中增加 scorer 输入特征，用于监控改进效果：

# 首次偏离点记录
{
    "ctx_norm": float,           # ||ctx_t||
    "cand_h_diff": float,        # ||cand_h_pred - cand_h_gt||
    "dist_diff": float,          # |dist_pred - dist_gt|
    "logit_gap": float,          # logit_pred - logit_gt
    "pred_way_id": int,
    "gt_way_id": int,
}

目的：

改进前：预期 cand_h_diff 和 dist_diff 是主导因素
改进后：预期 cand_ctx_diff 成为区分性特征
下一步
先实现建议 1：在 _compute_context 返回值改为 (T, C, d) 或增加 _compute_candidate_context 方法
A/B 对比：保持其他条件不变，只改动 cross-attention 的候选感知机制
关键指标：首次偏离点的 GT rank 分布变化
这个改动是架构层面的，如果有效，将是论文的核心贡献之一：多源数据不只是"有"，而是"能被用于决策"。