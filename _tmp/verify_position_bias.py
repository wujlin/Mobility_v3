"""
验证位置偏差假设：训练数据中 target_idx 的分布

如果训练数据中 target（GT）经常在位置 0，模型会学到"位置0更可能正确"。
但在推理时，CSR 顺序可能让 GT 不在位置 0，导致系统性选错。
"""
import numpy as np
import sys
sys.path.insert(0, '/Users/jinlin/Desktop/project/v3')

from src.data.way_graph.way_sequence_dataset import load_way_routes_npz, WayRouteDataset, make_way_casd_collate_fn

# 加载数据
routes = load_way_routes_npz('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1_sem5_train/way_routes_strict_masklen0.npz')

# 加载图
wg = np.load('/Users/jinlin/Desktop/project/v3/_sync/wsa/icml2026_routegen/WAYCASD1_waydata_rustbelt_seed0_strict_v1_sem5_train/way_graph.npz', allow_pickle=True)
way_adj_ptr = wg['way_adj_ptr']
way_adj_idx = wg['way_adj_idx']

print(f"Routes: {len(routes.way_seq_len)}")
print(f"Way graph: {len(way_adj_ptr)-1} nodes, {len(way_adj_idx)} edges")

# 创建 collate_fn
collate_fn = make_way_casd_collate_fn(
    way_adj_ptr=way_adj_ptr,
    way_adj_idx=way_adj_idx,
    max_candidates=32,
    tz_offset_hours=-5.0,
    past_k=8,
)

# 创建数据集
dataset = WayRouteDataset(routes, max_routes=None, max_way_len=160)

# 收集一个批次的 target_idx 分布
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

target_idx_all = []
outdeg_all = []

for i, batch in enumerate(loader):
    target_idx = batch['trans']['target_idx'].numpy()
    cand_mask = batch['trans']['cand_mask'].numpy()
    
    # 计算每个转移的实际候选数（outdeg）
    outdeg = cand_mask.sum(axis=1)
    
    target_idx_all.extend(target_idx.tolist())
    outdeg_all.extend(outdeg.tolist())
    
    if i >= 20:  # 只看前几个batch
        break

target_idx_all = np.array(target_idx_all)
outdeg_all = np.array(outdeg_all)

print(f"\n收集了 {len(target_idx_all)} 个转移样本")

# 整体 target_idx 分布
print("\n=== 整体 target_idx 分布 ===")
for idx in range(5):
    frac = (target_idx_all == idx).mean()
    print(f"  位置 {idx}: {frac:.1%}")

# 只看 outdeg=2 的情况
mask_od2 = (outdeg_all == 2)
print(f"\n=== outdeg=2 的情况 (n={mask_od2.sum()}) ===")
target_od2 = target_idx_all[mask_od2]
for idx in range(2):
    frac = (target_od2 == idx).mean()
    print(f"  位置 {idx}: {frac:.1%}")

print("\n" + "=" * 60)
print("分析结论")
print("=" * 60)
if (target_od2 == 0).mean() > 0.7:
    print("""
关键发现：在 outdeg=2 时，GT 大多数在位置 0！

这解释了"GT rank=2 恒定"的现象：
1. 训练时：GT 经常在位置 0 → 模型学到"位置 0 更可能正确"
2. 但这是因为 CSR 存储顺序 + _ensure_target 机制
3. 评估时：如果 GT 恰好在 CSR 的位置 1，模型仍然偏好位置 0
4. 结果：GT rank=2（GT 在位置 1，但模型选了位置 0）

解决方案：训练时随机打乱候选顺序！
""")
elif (target_od2 == 1).mean() > 0.7:
    print("""
关键发现：在 outdeg=2 时，GT 大多数在位置 1！

这可能是 _ensure_target 把 GT 强制插入末位导致的。
""")
else:
    print(f"""
GT 在 outdeg=2 时的位置分布相对均匀。
位置偏差假设可能不成立，需要进一步分析。
""")
