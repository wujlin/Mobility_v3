# ICML 2026 RouteGen: 本地同步结果索引（_sync）

- root: `_sync/wsA/icml2026_routegen`
- num_entries: `29`

## Experiments

- `E0_gt_baseline_detroit_F256_n200k_seed0` — `report` — `E0_gt_baseline_detroit_F256_n200k_seed0`
- `E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2` — `report` — `E0_gt_baseline_detroit_F256_n200k_seed0_od128_n10_sep2`
- `E10_rand4_control_detroit_F256_od128_seed0` (E10 (rand4 control)) — `report` — `E10_rand4_control_detroit_F256_od128_seed0`
- `E11_gridcnn_road_detroit_F256_od128_seed0` (E11 (gridcnn + road_prob)) — `report` — `E11_gridcnn_road_detroit_F256_od128_seed0`
- `E12_exec_road_prior_detroit_F256_K20_seed0` (E12 (Execution-stage road prior)) — `exec_prior` — `E12_exec_road_prior_detroit_F256_K20_seed0`
- `E13_fullscale_columbus_F256_K20_seed0` (E15 Full-scale (Columbus)) — `report` — `E13_fullscale_columbus_F256_K20_seed0`
- `E13_fullscale_detroit_F256_K20_seed0` (E13 (Full-scale Detroit)) — `fullscale` — `E13_fullscale_detroit_F256_K20_seed0`
- `E14_wp_audit_detroit_case04_seed0` (E14_wp_supervision_audit) — `report` — `E14_wp_audit_detroit_case04_seed0`
- `E16_realism_detroit_seed0` (E16 (Figure4 realism validation)) — `report` — `E16_realism_detroit_seed0`
- `E17_fig1_hero_detroit_case01_seed0` (E17 (Figure1 hero: corridor multi-modality)) — `report` — `E17_fig1_hero_detroit_case01_seed0`
- `E18_sem_intervention_gridcnn_road_case04_seed0` — `audit` — `E18_sem_intervention_gridcnn_road_case04_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E11_gridcnn_road_detroit_F256_od128_seed0/gridcnn_road_ckpt/last.pt`
  - best(mixture_jsd): shuffle jsd=0.012990752408024601 l1=0.30352941176471565 cov=1.0
- `E19a_audit_gridpos_case04_seed0` — `audit` — `E19a_audit_gridpos_case04_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E19a_gridpos_poi_entropy_detroit_F256_od128_seed0/last.pt`
  - best(mixture_jsd): none jsd=0.07012312996831116 l1=0.6329411764705977 cov=1.0
- `E19b_audit_gridattn_case04_seed0` — `audit` — `E19b_audit_gridattn_case04_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E19b_gridattn_poi_entropy_detroit_F256_od128_seed0/last.pt`
  - best(mixture_jsd): none jsd=0.049826506309147686 l1=0.5529411764705977 cov=1.0
- `E19c_attn_viz_gridattn_case04_idx0_seed0` — `report` — `E19c_attn_viz_gridattn_case04_idx0_seed0`
- `E1_end2end_diffusion_npz_detroit_F256_case01_seed0` — `other` — `E1_end2end_diffusion_npz_detroit_F256_case01_seed0`
- `E1b_end2end_l2_npz_detroit_F256_case01_seed0` — `other` — `E1b_end2end_l2_npz_detroit_F256_case01_seed0`
- `E20a_audit_gridpos_sc_case04_seed0` — `audit` — `E20a_audit_gridpos_sc_case04_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E22_gridpos_sc_tierroad_detroit_F256_od128_seed0/last.pt`
  - best(mixture_jsd): none jsd=0.028173435458265984 l1=0.4329411764705978 cov=1.0
- `E22_gridpos_sc_tierroad_detroit_F256_od128_seed0` — `train` — `E22_gridpos_sc_tierroad_detroit_F256_od128_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E22_gridpos_sc_tierroad_detroit_F256_od128_seed0/last.pt`
- `E22a_audit_gridpos_sc_tierroad_case04_seed0` — `audit` — `E22a_audit_gridpos_sc_tierroad_case04_seed0`
  - symlink -> `E20a_audit_gridpos_sc_case04_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E22_gridpos_sc_tierroad_detroit_F256_od128_seed0/last.pt`
  - best(mixture_jsd): none jsd=0.028173435458265984 l1=0.4329411764705978 cov=1.0
- `E23_fullscale_detroit_E22tierroad_K20_res0p1_seed0` — `other` — `E23_fullscale_detroit_E22tierroad_K20_res0p1_seed0`
- `E2_exec_diffusion_wp_residual_npz_detroit_F256_case01_seed0` — `other` — `E2_exec_diffusion_wp_residual_npz_detroit_F256_case01_seed0`
- `E2s_skeleton_only_detroit_F256_case01_seed0` — `other` — `E2s_skeleton_only_detroit_F256_case01_seed0`
- `E3_wp_diffusion_rel_detroit_F256_od128_seed0` — `train` — `E3_wp_diffusion_rel_detroit_F256_od128_seed0`
  - checkpoint: `/home/jinlin/data/geoexplicit_data/experiments/icml2026_routegen/E3_wp_diffusion_rel_detroit_F256_od128_seed0/last.pt`
- `E4_cascade_wpdec_exec_npz_detroit_F256_case01_seed0` — `other` — `E4_cascade_wpdec_exec_npz_detroit_F256_case01_seed0`
- `E5_cascade_wpdec_exec_res0p1_detroit_F256_case01_seed0` — `other` — `E5_cascade_wpdec_exec_res0p1_detroit_F256_case01_seed0`
- `E6_semantic_decision_ablation_detroit_F256_od128_seed0` — `other` — `E6_semantic_decision_ablation_detroit_F256_od128_seed0`
- `E7_corridor_semantic_audit_detroit_seed0` (E7 (corridor semantic audit, GT)) — `report` — `E7_corridor_semantic_audit_detroit_seed0`
- `E8_semantic_strength_mixture_detroit_F256_od128_seed0` (E8 (Semantic Mixture Match)) — `report` — `E8_semantic_strength_mixture_detroit_F256_od128_seed0`
- `E9_gridsem_poolquad_detroit_F256_od128_seed0` (E9 (grid semantics pooling)) — `report` — `E9_gridsem_poolquad_detroit_F256_od128_seed0`
