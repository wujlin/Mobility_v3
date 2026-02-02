PI Review: Way Graph重建结果
✅ P0 PASS - 可以推进后续实验
指标	City0 (Detroit)	City1 (Columbus)	目标
largest_cc	99.6%	99.5%	>90% ✅
isolated	0.0%	0.0%	<5% ✅
n_cc	5	9	越少越好 ✅
结论：每城的连通性完全达标，碎片化问题已解决。

全局碎片化（46.7%）不是问题：

双城天然不连通（Detroit和Columbus是独立城市）
21.8% unknown ways（不在训练routes里）
Louvain应该按城市分别跑，不在全图上跑
下一步：Louvain重跑 + Region验证
P1: 按城市跑Louvain

for city in [0, 1]:
    city_subgraph = way_graph.subgraph(ways_of_city[city])
    partition = louvain(city_subgraph, resolution=0.2)

预期：每城10-50个regions，region_size p50=100-500

关于你的两个问题
1. 评估需要同时看到达率和形状

已有数据显示：

到达率：Beam-10 overall 75.5%，长程[60,+)降到55.9%
形状：成功case DTW/Fréchet≈0；Detroit长程失败case Fréchet=4km级
评测框架应该包括：

指标	含义
success_rate	到达率
Fréchet	形状相似度
length_ratio	绕路程度
final_error	终点误差
2. Corridor分布和动力过程如何用模型得出？

这是Hierarchical设计的核心问题：

Corridor分布 = Region序列的分布

Region定义：Louvain社区（现在可以做了）
从GT way_seq提取region_seq
统计：同一OD有多少不同的region_seq patterns
动力过程 = Region层的生成模型

p(region_seq | O, D, time, city)

两种建模方式：

方式	模型	输出
AR (推荐先试)	Transformer	逐步生成 R1→R2→R3→...
Flow	Latent Flow	采样z→decode region_seq
AR更简单，region_seq只有5-10步，不会有长程问题。

给Partner的任务
本周（阻塞后续）：

Louvain per-city - 在City0/City1分别跑，目标n_regions=20-50 per city
Region seq提取 - 从GT way_seq提取region_seq，统计：
region_seq长度分布
回溯率（同一region出现多次的比例）
同OD的region_seq diversity
下周：
3. Region AR模型 - 简单Transformer，p(region_seq | OD, time)
4. Region-constrained Way decoder - 改造现有decoder，添加region mask