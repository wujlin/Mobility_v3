# Paired Compare (Flow(K=1) vs Flow(best-of-8,dest))

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/eval/per_route_waycasd_flow_test_n200pc.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/p0_bestofk/per_route_flow_bestof8_dest.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|62.3%|75.0%|+12.7pp|43|1|0.0000|
|[5,10)|50|68.0%|80.0%|+12.0pp|6|0|0.0312|
|[10,20)|63|65.1%|74.6%|+9.5pp|6|0|0.0312|
|[20,30)|71|70.4%|83.1%|+12.7pp|9|0|0.0039|
|[30,40)|54|61.1%|74.1%|+13.0pp|8|1|0.0391|
|[40,60)|75|49.3%|64.0%|+14.7pp|11|0|0.0010|
|[60,+)|19|63.2%|78.9%|+15.8pp|3|0|0.2500|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=206
  - ↓ Fréchet(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.03, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=0.0, median(Δ=B-A)=+0.0, frac(B better)=0.05, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=90.5, median(B)=90.5, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.000, median(Δ=B-A)=+0.000, frac(B better)=0.07, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=37
  - ↓ Fréchet(m): median(A)=22.8, median(B)=22.8, median(Δ=B-A)=+0.0, frac(B better)=0.08, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=22.8, median(B)=25.4, median(Δ=B-A)=+0.0, frac(B better)=0.11, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=95.8, median(B)=95.8, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.002, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.11, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=12
  - ↓ Fréchet(m): median(A)=34.6, median(B)=64.4, median(Δ=B-A)=+0.0, frac(B better)=0.08, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=46.5, median(B)=112.0, median(Δ=B-A)=+0.0, frac(B better)=0.25, CI95%(median Δ)=[-30.3,+12.7]
  - ↓ FinalErr(m): median(A)=64.6, median(B)=64.6, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.001, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.17, CI95%(median Δ)=[+0.0,+0.1]
