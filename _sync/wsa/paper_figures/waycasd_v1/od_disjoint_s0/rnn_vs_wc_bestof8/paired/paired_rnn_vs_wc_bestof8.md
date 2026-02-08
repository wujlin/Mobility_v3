# Paired Compare (RNN_beam10 vs WC_bestof8)

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/viz_rnn_vs_wc_bestof8/rnn_ar_beam10_dump.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/viz_rnn_vs_wc_bestof8/per_route_wc_bestof8_dest_dump.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|65.4%|75.0%|+9.6pp|54|22|0.0003|
|[5,10)|50|72.0%|80.0%|+8.0pp|8|4|0.3877|
|[10,20)|63|66.7%|74.6%|+7.9pp|10|5|0.3018|
|[20,30)|71|74.6%|83.1%|+8.5pp|9|3|0.1460|
|[30,40)|54|59.3%|74.1%|+14.8pp|10|2|0.0386|
|[40,60)|75|54.7%|64.0%|+9.3pp|14|7|0.1892|
|[60,+)|19|68.4%|78.9%|+10.5pp|3|1|0.6250|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=195
  - ↓ Fréchet(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.08, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.10, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=88.4, median(B)=88.4, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.000, median(Δ=B-A)=+0.000, frac(B better)=0.12, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=34
  - ↓ Fréchet(m): median(A)=0.0, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.09, CI95%(median Δ)=[+0.0,+22.8]
  - ↓ DTW(m): median(A)=0.0, median(B)=31.0, median(Δ=B-A)=+0.0, frac(B better)=0.12, CI95%(median Δ)=[+0.0,+33.9]
  - ↓ FinalErr(m): median(A)=83.1, median(B)=83.1, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.18, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=12
  - ↓ Fréchet(m): median(A)=111.0, median(B)=129.8, median(Δ=B-A)=+0.0, frac(B better)=0.08, CI95%(median Δ)=[+0.0,+405.0]
  - ↓ DTW(m): median(A)=149.8, median(B)=175.8, median(Δ=B-A)=+12.7, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+9667.1]
  - ↓ FinalErr(m): median(A)=67.3, median(B)=67.3, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.001, median(|B-1|)=0.003, median(Δ=B-A)=+0.000, frac(B better)=0.25, CI95%(median Δ)=[-0.0,+0.0]
