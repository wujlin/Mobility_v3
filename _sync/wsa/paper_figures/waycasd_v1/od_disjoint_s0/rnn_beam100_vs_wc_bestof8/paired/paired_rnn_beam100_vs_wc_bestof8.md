# Paired Compare (RNN_beam100 vs WC_bestof8)

- a_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/viz_rnn_vs_wc_bestof8/rnn_ar_beam100_dump.json`
- b_json: `_sync/wsa/pi_verify/20260206_od_disjoint_s0/viz_rnn_vs_wc_bestof8/per_route_wc_bestof8_dest_dump.json`
- key: `beam`
- n_routes (intersection): `332`

|Bin|n|succ(A)|succ(B)|Δsucc(B-A)|n01(A0,B1)|n10(A1,B0)|p(McNemar)|
|---|---:|---:|---:|---:|---:|---:|---:|
|overall|332|87.7%|75.0%|-12.7pp|10|52|0.0000|
|[5,10)|50|94.0%|80.0%|-14.0pp|0|7|0.0156|
|[10,20)|63|92.1%|74.6%|-17.5pp|2|13|0.0074|
|[20,30)|71|87.3%|83.1%|-4.2pp|2|5|0.4531|
|[30,40)|54|85.2%|74.1%|-11.1pp|2|8|0.1094|
|[40,60)|75|82.7%|64.0%|-18.7pp|3|17|0.0026|
|[60,+)|19|84.2%|78.9%|-5.3pp|1|2|1.0000|

## Shape（仅在 A 与 B 同时成功的 route 上做配对）

- overall: n_pair_success=239
  - ↓ Fréchet(m): median(A)=0.0, median(B)=13.3, median(Δ=B-A)=+0.0, frac(B better)=0.09, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ DTW(m): median(A)=0.0, median(B)=13.3, median(Δ=B-A)=+0.0, frac(B better)=0.10, CI95%(median Δ)=[+0.0,+0.0]
  - ↓ FinalErr(m): median(A)=78.5, median(B)=78.5, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.001, median(Δ=B-A)=+0.000, frac(B better)=0.13, CI95%(median Δ)=[+0.0,+0.0]
- [40,60): n_pair_success=45
  - ↓ Fréchet(m): median(A)=0.0, median(B)=82.8, median(Δ=B-A)=+0.0, frac(B better)=0.11, CI95%(median Δ)=[+0.0,+60.9]
  - ↓ DTW(m): median(A)=0.0, median(B)=95.1, median(Δ=B-A)=+22.8, frac(B better)=0.11, CI95%(median Δ)=[+0.0,+95.1]
  - ↓ FinalErr(m): median(A)=85.8, median(B)=85.8, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.000, median(|B-1|)=0.002, median(Δ=B-A)=+0.000, frac(B better)=0.16, CI95%(median Δ)=[+0.0,+0.0]
- [60,+): n_pair_success=14
  - ↓ Fréchet(m): median(A)=45.5, median(B)=129.8, median(Δ=B-A)=+0.0, frac(B better)=0.07, CI95%(median Δ)=[+0.0,+396.0]
  - ↓ DTW(m): median(A)=84.4, median(B)=175.8, median(Δ=B-A)=+12.7, frac(B better)=0.14, CI95%(median Δ)=[+0.0,+7848.8]
  - ↓ FinalErr(m): median(A)=64.6, median(B)=64.6, median(Δ=B-A)=+0.0, frac(B better)=0.00, CI95%(median Δ)=[+0.0,+0.0]
  - → LenRatio(=1): median(|A-1|)=0.001, median(|B-1|)=0.003, median(Δ=B-A)=+0.000, frac(B better)=0.21, CI95%(median Δ)=[+0.0,+0.0]
