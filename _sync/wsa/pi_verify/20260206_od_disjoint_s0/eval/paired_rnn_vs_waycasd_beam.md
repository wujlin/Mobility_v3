# Paired Compare (RNN-AR vs Way-CASD)

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/binned_rnn_ar_test_n200pc.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/per_route_waycasd_flow_test_n200pc.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|65.4%|62.3%|-3.0pp|31|41|0.2888|
|[5,10)|50|72.0%|68.0%|-4.0pp|5|7|0.7744|
|[10,20)|63|66.7%|65.1%|-1.6pp|5|6|1.0000|
|[20,30)|71|74.6%|70.4%|-4.2pp|4|7|0.5488|
|[30,40)|54|59.3%|61.1%|+1.9pp|7|6|1.0000|
|[40,60)|75|54.7%|49.3%|-5.3pp|8|12|0.5034|
|[60,+)|19|68.4%|63.2%|-5.3pp|2|3|1.0000|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=176
  - ↓ Fréchet(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.10, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.12, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=93.4, median(B)=93.4, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.000, median(Δ=B-A)=+0.000, frac(B better)=0.13, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=29
  - ↓ Fréchet(m): median(A)=0.0, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.10, CI95%(median Δ)=[+0.0,+22.8]
  - ↓ DTW(m): median(A)=0.0, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+28.6]
  - ↓ FinalErr(m): median(A)=93.6, median(B)=93.6, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=10
  - ↓ Fréchet(m): median(A)=30.8, median(B)=60.8, median(Δ=B-A)=+0.0, frac(B better)=0.10, CI95%(median Δ)=[+0.0,+31.1]
  - ↓ DTW(m): median(A)=69.6, median(B)=112.0, median(Δ=B-A)=+0.0, frac(B better)=0.20, CI95%(median Δ)=[-12.7,+45749.6]
  - ↓ FinalErr(m): median(A)=67.3, median(B)=67.3, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.001, median(Δ=B-A)=+0.000, frac(B better)=0.40, CI95%(median Δ)=[-0.0,+0.0]
