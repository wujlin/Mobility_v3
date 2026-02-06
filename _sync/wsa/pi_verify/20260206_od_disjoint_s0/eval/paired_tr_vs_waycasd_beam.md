# Paired Compare (Tr-AR vs Way-CASD)

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/binned_transformer_ar_test_n200pc.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/per_route_waycasd_flow_test_n200pc.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|50.9%|62.3%|+11.4pp|67|29|0.0001|
|[5,10)|50|52.0%|68.0%|+16.0pp|14|6|0.1153|
|[10,20)|63|46.0%|65.1%|+19.0pp|14|2|0.0042|
|[20,30)|71|57.7%|70.4%|+12.7pp|12|3|0.0352|
|[30,40)|54|55.6%|61.1%|+5.6pp|9|6|0.6072|
|[40,60)|75|46.7%|49.3%|+2.7pp|12|10|0.8318|
|[60,+)|19|42.1%|63.2%|+21.1pp|6|2|0.2891|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=140
  - ↓ Fréchet(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.19, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.22, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=120.9, median(B)=120.9, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.000, median(Δ=B-A)=+0.000, frac(B better)=0.21, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=25
  - ↓ Fréchet(m): median(A)=0.0, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.08, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.16, CI95%(median Δ)=[+0.0,+60.9]
  - ↓ FinalErr(m): median(A)=127.4, median(B)=127.4, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.16, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=6
  - ↓ Fréchet(m): median(A)=7.2, median(B)=100.0, median(Δ=B-A)=+96.5, frac(B better)=0.17
  - ↓ DTW(m): median(A)=7.2, median(B)=112.0, median(Δ=B-A)=+108.4, frac(B better)=0.17
  - ↓ FinalErr(m): median(A)=79.7, median(B)=79.7, median(Δ=B-A)=+0.0, frac(B better)=0.00
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.001, median(Δ=B-A)=+0.000, frac(B better)=0.17
