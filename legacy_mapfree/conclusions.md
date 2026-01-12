# legacy_mapfree：技术结论（供论文/复盘引用）

> 口径：这里记录“问题是什么、证据是什么、结论是什么”，避免陷入技术验证式的流水账。

## 1) 致命问题：Task Definition vs Data Mismatch

**问题**：叙事主线是 trip-level 的 **route generation / corridor choice**，但实际训练数据曾长期使用滑窗 `F=256`（window-level continuation），导致多数样本的 OD 直线距离较短、绕行比接近 1，根本不包含“路径选择”语义。

**证据**（示例指标，来自 D0/W0 审计）：
- full vs prefix 的 detour 相关性极低（`detour_corr` 接近 0），说明 `prefix(F=256)` 无法预测 full segment 的路径选择。
- segment-level “keep” 的数量足够（> 1000），说明转向 segment-level 可行。

**结论**：必须转向 **segment-level / trip-level** 的路线建模，否则 corridor 相关指标与语义条件化没有科学意义。

## 2) 语义条件化（POI/landuse）在 window-level 上“无效”的真实原因

**现象**：POI/entropy grid 在 decision stage 的干预实验出现：
- `none ~ zeros ~ shuffle`（不敏感）或 `shuffle > none`（负偏置）。

**真实原因**：window-level 片段没有路径选择点，语义信息即使存在也无法影响走廊选择；模型的“语义失败”并非融合机制必然失败，而是任务本身不需要语义。

**结论**：语义是否有用必须在 **trip-level corridor choice** 上重新验证。

## 3) tier-road 在 decision stage 的角色（阶段特异性）

**现象**：tier-road 在部分设置中能呈现“正确的语义敏感模式”（`none > shuffle > zeros`）。

**风险点**：在不包含路径选择的短片段上，tier-road 可能诱导模型“走主干道绕行”，造成 ADE/Jaccard 冲突；需要可视化与数据尺度审计。

**结论**：road topology 类信号对“决策”是否有效，取决于任务尺度是否真实涉及 corridor choice。

## 4) 时间特征（start_t）具备可用前提

**确认**：segments.parquet 的 `t` 为有效 Unix 秒；新生成的 `*_epoch.npz` 中 `start_t` 非 0；`temporal_mode=auto` 可以真实生效（通勤/非通勤分层具备前提）。

**结论**：时间特征可以作为 map-aware 走廊选择模型的条件输入之一。

## 5) 重要经验：可视化优先于指标

**教训**：纯看数值指标容易陷入 “metric gaming”。在 E25/E26 的 ADE vs Jaccard 冲突中，可视化揭示了 window artifact 与“主干道绕行”问题，从而定位到任务定义错误。

**结论**：未来所有 gate 必须配套最小可视化（case-level 三方对比/waypoint audit/error heatmap），否则不推进下一步。

