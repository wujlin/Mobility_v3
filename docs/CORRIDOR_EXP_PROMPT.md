# Corridor Diversity 实验 Prompt

> 给Partner C的任务说明
> 创建日期：2026-02-05
> PI审核：待审核

---

## 一、任务背景

### 你要做什么
我们正在写一篇关于route generation的论文（CascadeTraj）。论文的一个核心claim是：
> 现有方法生成的路线缺乏"corridor diversity"——同一OD生成多条路线时，它们都挤在同一个走廊里。

你的任务是**用数据验证这个claim**，具体包括：
1. 定义什么是corridor（基于way sequence聚类）
2. 分析GT数据中的corridor分布
3. 为后续的baseline比较实验建立ground truth

### 关键约束：避免循环论证
> ⚠️ **重要**：corridor的定义**必须独立于**我们的region_seq

**错误做法**：用region_seq定义corridor → 然后claim我们生成了多样的corridor（循环论证）

**正确做法**：用way sequence的LCS聚类定义corridor → 独立于任何方法 → 公平评估所有方法

---

## 二、代码框架说明

### 2.1 数据文件位置

```bash
# Way routes（GT路线数据）
/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W5_way_routes_strict/way_routes_strict_masklen0.npz

# Way graph（路网图）
/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W3_way_graph_strict/way_graph.npz

# Way features（道路特征）
/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W4_way_features_sem/way_features.npz
```

### 2.2 数据结构

**way_routes.npz 包含**：
```python
data = np.load(way_routes_npz)
# route_way: (N, max_len) int32, way_id序列, 0-padded
# route_len: (N,) int32, 每条路线的实际长度
# route_city: (N,) int32, 城市ID (0=Detroit, 1=Columbus)
# route_hour: (N,) int32, 出发小时
# route_dow: (N,) int32, 星期几
```

**way_graph.npz 包含**：
```python
data = np.load(way_graph_npz)
# adj_ptr: CSR格式的邻接表指针
# adj_idx: CSR格式的邻接表索引
# way_center: (n_ways, 2) float32, 每条way的中心点坐标
# way_city: (n_ways,) int32, 每条way所属城市
```

### 2.3 已有的相关代码

```
src/
  data/
    way_graph/
      way_sequence_dataset.py  # WayRouteDataset, load_way_routes_npz
  evaluation/
    way_casd_binned_eval.py    # 现有的binned evaluation
    shape_metrics.py           # DTW, Fréchet distance计算
```

### 2.4 现有的数据加载示例

```python
from src.data.way_graph.way_sequence_dataset import load_way_routes_npz

routes = load_way_routes_npz(way_routes_npz)
# routes["route_way"]: (N, max_len)
# routes["route_len"]: (N,)
# routes["route_city"]: (N,)
```

---

## 三、具体任务

### Task 1: 实现LCS相似度计算

**文件**：`src/evaluation/corridor_metrics.py`（新建）

```python
"""Corridor diversity metrics based on LCS similarity."""

import numpy as np
from typing import List, Tuple
from functools import lru_cache


def lcs_length(seq_a: np.ndarray, seq_b: np.ndarray) -> int:
    """
    计算两个way sequence的最长公共子序列长度。
    
    Args:
        seq_a: (L1,) int array, way_id序列（不含padding）
        seq_b: (L2,) int array, way_id序列（不含padding）
    
    Returns:
        LCS长度
    """
    # TODO: 实现标准LCS动态规划
    # 注意：输入已经去除padding（只包含有效way_id）
    pass


def lcs_similarity(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """
    计算两个way sequence的LCS相似度。
    
    similarity = LCS_length / min(len(seq_a), len(seq_b))
    
    Returns:
        float in [0, 1], 1表示完全相同
    """
    lcs_len = lcs_length(seq_a, seq_b)
    min_len = min(len(seq_a), len(seq_b))
    if min_len == 0:
        return 0.0
    return lcs_len / min_len


def compute_pairwise_similarity(routes: List[np.ndarray]) -> np.ndarray:
    """
    计算一组routes的两两LCS相似度矩阵。
    
    Args:
        routes: List of (Li,) arrays, 每个是一条route的way序列
    
    Returns:
        (N, N) float array, similarity matrix
    """
    n = len(routes)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            s = lcs_similarity(routes[i], routes[j])
            sim[i, j] = s
            sim[j, i] = s
    return sim
```

### Task 2: 实现Corridor聚类

**文件**：`src/evaluation/corridor_clustering.py`（新建）

```python
"""Corridor extraction via hierarchical clustering on LCS similarity."""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import List, Dict, Any

from src.evaluation.corridor_metrics import compute_pairwise_similarity


def extract_corridors(
    routes: List[np.ndarray],
    threshold: float = 0.5,
    method: str = "average"
) -> Dict[str, Any]:
    """
    从一组routes中提取corridors。
    
    Args:
        routes: List of way sequences (each is (Li,) array)
        threshold: LCS similarity threshold for clustering
        method: linkage method ("average", "complete", "single")
    
    Returns:
        {
            "n_corridors": int,
            "labels": (N,) int array, corridor label for each route
            "corridor_sizes": List[int], number of routes in each corridor
            "corridor_prototypes": List[int], index of prototype route for each corridor
        }
    """
    if len(routes) == 0:
        return {"n_corridors": 0, "labels": np.array([]), "corridor_sizes": [], "corridor_prototypes": []}
    
    if len(routes) == 1:
        return {"n_corridors": 1, "labels": np.array([0]), "corridor_sizes": [1], "corridor_prototypes": [0]}
    
    # Step 1: 计算相似度矩阵
    sim_matrix = compute_pairwise_similarity(routes)
    
    # Step 2: 转换为距离矩阵 (1 - similarity)
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)  # 确保对角线为0
    
    # Step 3: 层次聚类
    # squareform将方阵转为condensed form
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    
    # Step 4: 按threshold切割
    # threshold是similarity，所以distance threshold = 1 - threshold
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance")
    labels = labels - 1  # fcluster返回1-indexed，转为0-indexed
    
    # Step 5: 统计
    n_corridors = len(set(labels))
    corridor_sizes = [np.sum(labels == c) for c in range(n_corridors)]
    
    # Step 6: 找prototype（每个cluster内与其他成员平均相似度最高的route）
    corridor_prototypes = []
    for c in range(n_corridors):
        members = np.where(labels == c)[0]
        if len(members) == 1:
            corridor_prototypes.append(int(members[0]))
        else:
            # 计算每个member与其他member的平均相似度
            avg_sim = []
            for m in members:
                sim_to_others = [sim_matrix[m, other] for other in members if other != m]
                avg_sim.append(np.mean(sim_to_others))
            best_idx = members[np.argmax(avg_sim)]
            corridor_prototypes.append(int(best_idx))
    
    return {
        "n_corridors": n_corridors,
        "labels": labels,
        "corridor_sizes": corridor_sizes,
        "corridor_prototypes": corridor_prototypes
    }
```

### Task 3: 实现GT Corridor分析脚本

**文件**：`src/evaluation/analyze_gt_corridors.py`（新建）

```python
"""Analyze corridor distribution in GT route data."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz
from src.evaluation.corridor_clustering import extract_corridors

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def get_od_key(route_way: np.ndarray, route_len: int) -> str:
    """获取route的OD key (start_way, end_way)"""
    start = int(route_way[0])
    end = int(route_way[route_len - 1])
    return f"{start}_{end}"


def analyze_corridors(
    way_routes_npz: str,
    min_routes_per_od: int = 5,
    lcs_threshold: float = 0.5,
    output_path: str = None
) -> dict:
    """
    分析GT数据中的corridor分布。
    
    Args:
        way_routes_npz: 路径到way_routes.npz
        min_routes_per_od: OD至少有多少条route才分析
        lcs_threshold: LCS相似度阈值
        output_path: 输出JSON路径
    
    Returns:
        分析结果字典
    """
    # 加载数据
    log.info(f"Loading routes from {way_routes_npz}")
    routes = load_way_routes_npz(way_routes_npz)
    route_way = routes["route_way"]  # (N, max_len)
    route_len = routes["route_len"]  # (N,)
    route_city = routes["route_city"]  # (N,)
    
    n_routes = len(route_len)
    log.info(f"Loaded {n_routes} routes")
    
    # 按OD分组
    od_groups = defaultdict(list)
    for i in range(n_routes):
        seq = route_way[i, :route_len[i]]  # 去除padding
        od_key = get_od_key(route_way[i], route_len[i])
        city = int(route_city[i])
        od_groups[(city, od_key)].append((i, seq))
    
    log.info(f"Found {len(od_groups)} unique (city, OD) pairs")
    
    # 筛选有足够routes的OD
    valid_ods = {k: v for k, v in od_groups.items() if len(v) >= min_routes_per_od}
    log.info(f"Found {len(valid_ods)} ODs with >= {min_routes_per_od} routes")
    
    # 对每个OD做corridor分析
    results = {
        "config": {
            "way_routes_npz": way_routes_npz,
            "min_routes_per_od": min_routes_per_od,
            "lcs_threshold": lcs_threshold
        },
        "summary": {},
        "per_od": []
    }
    
    all_n_corridors = []
    all_effective_k = []
    
    for (city, od_key), route_list in valid_ods.items():
        indices = [r[0] for r in route_list]
        seqs = [r[1] for r in route_list]
        
        # 提取corridors
        corridor_result = extract_corridors(seqs, threshold=lcs_threshold)
        
        n_corridors = corridor_result["n_corridors"]
        corridor_sizes = corridor_result["corridor_sizes"]
        
        # 计算有效corridor数（基于熵）
        if n_corridors > 0:
            probs = np.array(corridor_sizes) / sum(corridor_sizes)
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            effective_k = np.exp(entropy)
        else:
            effective_k = 0.0
        
        all_n_corridors.append(n_corridors)
        all_effective_k.append(effective_k)
        
        # 记录
        od_result = {
            "city": city,
            "od_key": od_key,
            "n_routes": len(seqs),
            "n_corridors": n_corridors,
            "corridor_sizes": corridor_sizes,
            "effective_k": float(effective_k),
            "route_indices": indices,
            "corridor_labels": corridor_result["labels"].tolist(),
            "prototype_indices": [indices[p] for p in corridor_result["corridor_prototypes"]]
        }
        results["per_od"].append(od_result)
    
    # 汇总统计
    results["summary"] = {
        "n_valid_ods": len(valid_ods),
        "avg_n_corridors": float(np.mean(all_n_corridors)) if all_n_corridors else 0.0,
        "median_n_corridors": float(np.median(all_n_corridors)) if all_n_corridors else 0.0,
        "avg_effective_k": float(np.mean(all_effective_k)) if all_effective_k else 0.0,
        "n_corridors_distribution": {
            "1": sum(1 for n in all_n_corridors if n == 1),
            "2": sum(1 for n in all_n_corridors if n == 2),
            "3": sum(1 for n in all_n_corridors if n == 3),
            "4+": sum(1 for n in all_n_corridors if n >= 4)
        }
    }
    
    log.info(f"Summary: avg_n_corridors={results['summary']['avg_n_corridors']:.2f}, "
             f"avg_effective_k={results['summary']['avg_effective_k']:.2f}")
    
    # 保存
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze GT corridor distribution")
    parser.add_argument("--way_routes_npz", type=str, required=True)
    parser.add_argument("--min_routes_per_od", type=int, default=5)
    parser.add_argument("--lcs_threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    analyze_corridors(
        way_routes_npz=args.way_routes_npz,
        min_routes_per_od=args.min_routes_per_od,
        lcs_threshold=args.lcs_threshold,
        output_path=args.output
    )
```

### Task 4: 实现Shortest Path Match分析

**文件**：`src/evaluation/analyze_shortest_path_match.py`（新建）

```python
"""Analyze how many GT routes match shortest path."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import networkx as nx

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def load_way_graph(way_graph_npz: str) -> nx.DiGraph:
    """加载way graph为NetworkX图"""
    data = np.load(way_graph_npz)
    adj_ptr = data["adj_ptr"]
    adj_idx = data["adj_idx"]
    
    G = nx.DiGraph()
    n_ways = len(adj_ptr) - 1
    G.add_nodes_from(range(n_ways))
    
    for i in range(n_ways):
        start, end = adj_ptr[i], adj_ptr[i + 1]
        for j in adj_idx[start:end]:
            G.add_edge(i, int(j))
    
    return G


def is_shortest_path_match(gt_seq: np.ndarray, G: nx.DiGraph, tolerance: float = 0.1) -> bool:
    """
    检查GT路线是否与shortest path匹配。
    
    Args:
        gt_seq: GT way sequence
        G: NetworkX图
        tolerance: 允许的长度偏差比例
    
    Returns:
        True if GT matches shortest path within tolerance
    """
    start = int(gt_seq[0])
    end = int(gt_seq[-1])
    
    try:
        sp = nx.shortest_path(G, start, end)
    except nx.NetworkXNoPath:
        return False
    
    # 检查长度是否接近
    sp_len = len(sp)
    gt_len = len(gt_seq)
    
    if abs(gt_len - sp_len) / sp_len > tolerance:
        return False
    
    # 检查路径是否相同（考虑到可能有多条shortest path）
    if gt_len == sp_len:
        # 完全相同长度，检查是否是相同路径或另一条shortest path
        gt_set = set(gt_seq.tolist())
        sp_set = set(sp)
        overlap = len(gt_set & sp_set) / len(gt_set | sp_set)
        return overlap > 0.9  # 90%重叠认为是match
    
    return False


def analyze_shortest_match(
    way_routes_npz: str,
    way_graph_npz: str,
    tolerance: float = 0.1,
    output_path: str = None
) -> dict:
    """分析GT routes中shortest path match的比例"""
    
    log.info("Loading graph...")
    G = load_way_graph(way_graph_npz)
    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    log.info("Loading routes...")
    routes = load_way_routes_npz(way_routes_npz)
    route_way = routes["route_way"]
    route_len = routes["route_len"]
    route_city = routes["route_city"]
    
    n_routes = len(route_len)
    log.info(f"Analyzing {n_routes} routes...")
    
    # 统计
    match_count = 0
    per_city = {0: {"total": 0, "match": 0}, 1: {"total": 0, "match": 0}}
    
    for i in range(n_routes):
        seq = route_way[i, :route_len[i]]
        city = int(route_city[i])
        
        per_city[city]["total"] += 1
        
        if is_shortest_path_match(seq, G, tolerance):
            match_count += 1
            per_city[city]["match"] += 1
        
        if (i + 1) % 1000 == 0:
            log.info(f"Processed {i + 1}/{n_routes}")
    
    results = {
        "config": {
            "way_routes_npz": way_routes_npz,
            "way_graph_npz": way_graph_npz,
            "tolerance": tolerance
        },
        "overall": {
            "n_routes": n_routes,
            "n_match": match_count,
            "match_rate": match_count / n_routes if n_routes > 0 else 0.0
        },
        "per_city": {
            str(city): {
                "n_routes": stats["total"],
                "n_match": stats["match"],
                "match_rate": stats["match"] / stats["total"] if stats["total"] > 0 else 0.0
            }
            for city, stats in per_city.items()
        }
    }
    
    log.info(f"Overall match rate: {results['overall']['match_rate']:.1%}")
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"Saved to {output_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--way_routes_npz", type=str, required=True)
    parser.add_argument("--way_graph_npz", type=str, required=True)
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    
    analyze_shortest_match(
        way_routes_npz=args.way_routes_npz,
        way_graph_npz=args.way_graph_npz,
        tolerance=args.tolerance,
        output_path=args.output
    )
```

---

## 四、执行步骤

### Step 1: 创建文件
```bash
# 在workstation上执行
cd /home/jinlin/projects/Mobility_v3

# 创建文件
touch src/evaluation/corridor_metrics.py
touch src/evaluation/corridor_clustering.py
touch src/evaluation/analyze_gt_corridors.py
touch src/evaluation/analyze_shortest_path_match.py
```

### Step 2: 实现并测试LCS
```bash
# 先写corridor_metrics.py中的lcs_length
# 测试：
python -c "
from src.evaluation.corridor_metrics import lcs_length
import numpy as np
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 3, 5])
print(lcs_length(a, b))  # 应该是3
"
```

### Step 3: 运行Shortest Path分析
```bash
python src/evaluation/analyze_shortest_path_match.py \
  --way_routes_npz /home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W5_way_routes_strict/way_routes_strict_masklen0.npz \
  --way_graph_npz /home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W3_way_graph_strict/way_graph.npz \
  --output _sync/wsa/corridor_analysis/shortest_path_match.json
```

### Step 4: 运行GT Corridor分析
```bash
python src/evaluation/analyze_gt_corridors.py \
  --way_routes_npz /home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1/W5_way_routes_strict/way_routes_strict_masklen0.npz \
  --min_routes_per_od 5 \
  --lcs_threshold 0.5 \
  --output _sync/wsa/corridor_analysis/gt_corridors.json
```

---

## 五、输出规范

### 输出目录结构
```
_sync/wsa/corridor_analysis/
  shortest_path_match.json   # Task 4输出
  gt_corridors.json          # Task 3输出
  run_shortest_match.log     # 运行日志
  run_gt_corridors.log       # 运行日志
```

### JSON输出格式（参考）

**shortest_path_match.json**:
```json
{
  "config": {...},
  "overall": {
    "n_routes": 12345,
    "n_match": 2345,
    "match_rate": 0.19
  },
  "per_city": {
    "0": {"n_routes": 5000, "n_match": 800, "match_rate": 0.16},
    "1": {"n_routes": 7345, "n_match": 1545, "match_rate": 0.21}
  }
}
```

**gt_corridors.json**:
```json
{
  "config": {...},
  "summary": {
    "n_valid_ods": 234,
    "avg_n_corridors": 2.3,
    "avg_effective_k": 1.8,
    "n_corridors_distribution": {"1": 50, "2": 100, "3": 60, "4+": 24}
  },
  "per_od": [...]
}
```

---

## 六、注意事项

### 6.1 性能考虑
- LCS是O(n*m)复杂度，对长序列可能慢
- 先在小数据集上测试（用`--max_routes 1000`之类的参数）
- 如果太慢，考虑用numba加速

### 6.2 调试建议
- 先用print调试，确保数据加载正确
- 检查route_way的padding是否正确处理
- 检查OD分组是否正确

### 6.3 常见错误
- **忘记去除padding**：route_way是0-padded的，必须用route_len截取
- **索引越界**：way_id从0开始，注意边界
- **内存爆炸**：不要一次性计算所有routes的两两相似度

---

## 七、完成标准

完成后请回复以下信息：

1. **Shortest Path Match Rate**：overall match rate是多少？
2. **GT Corridor统计**：
   - 有多少个OD有>=5条routes？
   - 平均每个OD有多少个corridors？
   - n_corridors的分布（1/2/3/4+各占多少）？
3. **代码PR**：提交代码，包含上述4个文件
4. **任何问题或发现**：数据中有什么意外情况？

---

## 八、联系方式

如有问题，先查阅：
- `docs/corridor_analysis.md`：叙事逻辑
- `docs/CORRIDOR_DIVERSITY_LITERATURE-2.md`：文献背景
- `src/evaluation/way_casd_binned_eval.py`：现有评估代码参考

实在不清楚再问PI。
