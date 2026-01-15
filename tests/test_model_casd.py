import torch

from src.models.casd.casd import CASDAECfg, CASDAutoEncoder
from src.models.casd.conditions import ConditionEncoderCfg
from src.models.casd.latent_flow import LatentFlowCfg, LatentFlowMatching
from src.models.casd.segment_encoder import make_segment_feature_tensors


def _toy_graph():
    # 5 segments, 4 nodes.
    seg_u = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)
    seg_v = torch.tensor([1, 2, 3, 3, 2], dtype=torch.long)
    S = int(seg_u.numel())

    seg_center_y = torch.linspace(0, 1, S)
    seg_center_x = torch.linspace(0, 1, S)
    seg_dir_y = torch.ones(S) * 0.0
    seg_dir_x = torch.ones(S) * 1.0
    seg_len_m = torch.ones(S) * 10.0
    seg_tier = torch.zeros(S, dtype=torch.long)
    seg_city = torch.zeros(S, dtype=torch.long)

    # node -> outgoing segments
    # node0: [seg0, seg4], node1: [seg1, seg3], node2: [seg2], node3: []
    node_seg_ptr = torch.tensor([0, 2, 4, 5, 5], dtype=torch.long)
    node_seg_idx = torch.tensor([0, 4, 1, 3, 2], dtype=torch.long)

    # seg -> successor segments via seg_v node
    # seg0(v=1)->[1,3], seg1(v=2)->[2], seg2(v=3)->[], seg3(v=3)->[], seg4(v=2)->[2]
    seg_succ_ptr = torch.tensor([0, 2, 3, 3, 3, 4], dtype=torch.long)
    seg_succ_idx = torch.tensor([1, 3, 2, 2], dtype=torch.long)

    features = make_segment_feature_tensors(
        seg_center_y=seg_center_y,
        seg_center_x=seg_center_x,
        seg_dir_y=seg_dir_y,
        seg_dir_x=seg_dir_x,
        seg_len_m=seg_len_m,
        seg_tier=seg_tier,
        seg_city=seg_city,
    )

    return {
        "features": features,
        "seg_u": seg_u,
        "seg_v": seg_v,
        "node_seg_ptr": node_seg_ptr,
        "node_seg_idx": node_seg_idx,
        "seg_succ_ptr": seg_succ_ptr,
        "seg_succ_idx": seg_succ_idx,
    }


def test_casd_autoencoder_smoke():
    g = _toy_graph()
    cfg = CASDAECfg(d_model=32, n_latent=4, n_heads=4, dropout=0.0, max_candidates=4, max_len=16, coord_scale=1.0)
    model = CASDAutoEncoder(
        cfg=cfg,
        seg_features=g["features"],
        seg_v=g["seg_v"],
        seg_succ_ptr=g["seg_succ_ptr"],
        seg_succ_idx=g["seg_succ_idx"],
        node_seg_ptr=g["node_seg_ptr"],
        node_seg_idx=g["node_seg_idx"],
    )

    seg_seq_pad = torch.tensor([[0, 1, 2], [0, 3, -1]], dtype=torch.long)
    route_cond = {
        "start_pos": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        "dest_pos": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        "hour": torch.tensor([8, 20], dtype=torch.long),
        "route_city": torch.tensor([0, 0], dtype=torch.long),
        "corridor_type": torch.tensor([0, 0], dtype=torch.long),
        "start_node": torch.tensor([0, 0], dtype=torch.long),
        "dest_node": torch.tensor([3, 3], dtype=torch.long),
    }

    # Packed transitions (T=5, C=4)
    trans = {
        "route_idx": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "cur_seg": torch.tensor([-1, 0, 1, -1, 0], dtype=torch.long),
        "cand_seg": torch.tensor(
            [
                [0, 4, -1, -1],  # start->seg0
                [1, 3, -1, -1],  # seg0->seg1
                [2, -1, -1, -1],  # seg1->seg2
                [0, 4, -1, -1],  # start->seg0
                [1, 3, -1, -1],  # seg0->seg3
            ],
            dtype=torch.long,
        ),
        "cand_mask": torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [True, True, False, False],
            ],
            dtype=torch.bool,
        ),
        "target_idx": torch.tensor([0, 0, 0, 0, 1], dtype=torch.long),
    }

    loss, stats = model.compute_loss({"seg_seq_pad": seg_seq_pad, "route_cond": route_cond, "trans": trans})
    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert "acc" in stats


def test_casd_latent_flow_smoke():
    cfg = LatentFlowCfg(d_model=32, n_latent=4, n_layers=2, n_heads=4, dropout=0.0, solver_steps=5, cfg_drop_prob=0.1)
    flow = LatentFlowMatching(cfg=cfg, cond_cfg=ConditionEncoderCfg(d_model=32, coord_scale=1.0))
    z1 = torch.randn((2, 4, 32), dtype=torch.float32)
    route_cond = {
        "start_pos": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
        "dest_pos": torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
        "hour": torch.tensor([8, 20], dtype=torch.long),
        "route_city": torch.tensor([0, 0], dtype=torch.long),
        "corridor_type": torch.tensor([0, 3], dtype=torch.long),
    }
    loss, _ = flow.compute_loss(z1=z1, route_cond=route_cond)
    assert loss.shape == ()
    assert not torch.isnan(loss)

