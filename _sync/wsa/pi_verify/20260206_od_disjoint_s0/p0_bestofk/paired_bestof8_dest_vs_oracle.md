# Paired Compare (Flow(best-of-8,dest) vs Oracle(gt latent))

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/p0_bestofk/per_route_flow_bestof8_dest.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/p0_bestofk/per_route_oracle_gtlatent.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|75.0%|67.8%|-7.2pp|7|31|0.0001|
|[5,10)|50|80.0%|78.0%|-2.0pp|2|3|1.0000|
|[10,20)|63|74.6%|74.6%|+0.0pp|3|3|1.0000|
|[20,30)|71|83.1%|73.2%|-9.9pp|1|8|0.0391|
|[30,40)|54|74.1%|64.8%|-9.3pp|0|5|0.0625|
|[40,60)|75|64.0%|54.7%|-9.3pp|1|8|0.0391|
|[60,+)|19|78.9%|57.9%|-21.1pp|0|4|0.1250|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=218
  - ↓ Fréchet(m): median(A)=7.2, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.14, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=7.2, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=80.3, median(B)=80.3, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.000, median(Δ=B-A)=+0.000, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=40
  - ↓ Fréchet(m): median(A)=22.8, median(B)=19.0, median(Δ=B-A)=+0.0, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=24.1, median(B)=19.0, median(Δ=B-A)=+0.0, frac(B better)=0.23, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=90.5, median(B)=90.5, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.002, median(|B-1|)=0.001, median(Δ=B-A)=+0.000, frac(B better)=0.20, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=11
  - ↓ Fréchet(m): median(A)=90.6, median(B)=38.2, median(Δ=B-A)=+0.0, frac(B better)=0.27, CI95%(median Δ)=[-7.1,+0.0]
  - ↓ DTW(m): median(A)=169.1, median(B)=38.2, median(Δ=B-A)=+0.0, frac(B better)=0.45, CI95%(median Δ)=[-577.0,+0.0]
  - ↓ FinalErr(m): median(A)=66.2, median(B)=66.2, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.002, median(|B-1|)=0.001, median(Δ=B-A)=+0.000, frac(B better)=0.45, CI95%(median Δ)=[-0.0,+0.0]
