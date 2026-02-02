# Coarse OD-bin corridor diversity (od_bin_deg=0.02)

| metric | res=0.2 | res=1.0 |
|---|---:|---:|
| n_routes_in_region_seq_npz | 5022 | 5022 |
| od_bins.n_bins | 1925 | 1925 |
| od_bins.n_bins_ge_min_routes | 171 | 171 |
| multimodal.n_bins_multi | 149 | 122 |
| multimodal.frac_multi_over_ge_min | 0.8713450292397661 | 0.7134502923976608 |
| multimodal.n_bins_sep_lcs | 84 | 85 |
| multimodal.frac_sep_over_multi | 0.5637583892617449 | 0.6967213114754098 |
| uniq_patterns_per_bin_ge_min.p95 | 7.0 | 5.0 |
| uniq_patterns_per_bin_ge_min.max | 12 | 8 |

## Per-city

| city | metric | res=0.2 | res=1.0 |
|---:|---|---:|---:|
| 0 | n_bins_ge_min_routes | 34 | 34 |
| 0 | multi.frac_multi_over_ge_min | 0.8529411764705882 | 0.7647058823529411 |
| 0 | uniq_p95 | 6.349999999999998 | 5.0 |
| 0 | uniq_max | 7 | 5 |
| 0 | routes_per_bin.max | 21 | 21 |
| 1 | n_bins_ge_min_routes | 137 | 137 |
| 1 | multi.frac_multi_over_ge_min | 0.8759124087591241 | 0.7007299270072993 |
| 1 | uniq_p95 | 7.0 | 5.0 |
| 1 | uniq_max | 12 | 8 |
| 1 | routes_per_bin.max | 89 | 89 |

## Top multimodal bins (res=1.0)

Shown for quick manual inspection (rep routes).

- #0 city=1 n_routes=18 n_patterns=8 lcs_sep=0.800 od_bin=(-4152,1999,-4151,1999)
  - count=7 len=2 rep_routes=[2491, 2814, 3793]
  - count=3 len=4 rep_routes=[2528, 6615, 6977]
  - count=3 len=3 rep_routes=[2632, 5789, 7099]
- #1 city=1 n_routes=9 n_patterns=6 lcs_sep=0.750 od_bin=(-4151,1999,-4151,2000)
  - count=4 len=4 rep_routes=[4744, 6291, 6871]
  - count=1 len=4 rep_routes=[2903]
  - count=1 len=3 rep_routes=[3619]
- #2 city=1 n_routes=6 n_patterns=6 lcs_sep=1.000 od_bin=(-4152,2000,-4151,1999)
  - count=1 len=7 rep_routes=[2435]
  - count=1 len=3 rep_routes=[2984]
  - count=1 len=3 rep_routes=[3681]
- #3 city=1 n_routes=51 n_patterns=5 lcs_sep=0.500 od_bin=(-4152,2000,-4152,2003)
  - count=33 len=3 rep_routes=[2535, 2548, 2654]
  - count=12 len=4 rep_routes=[3259, 3393, 3475]
  - count=3 len=2 rep_routes=[3982, 4551, 5708]
- #4 city=1 n_routes=37 n_patterns=5 lcs_sep=1.000 od_bin=(-4146,2002,-4146,2003)
  - count=21 len=2 rep_routes=[2593, 2660, 2683]
  - count=6 len=2 rep_routes=[3450, 4692, 4941]
  - count=6 len=1 rep_routes=[3460, 3652, 3714]
- #5 city=1 n_routes=16 n_patterns=5 lcs_sep=0.600 od_bin=(-4153,2003,-4152,2000)
  - count=10 len=3 rep_routes=[3153, 3188, 5414]
  - count=3 len=4 rep_routes=[3119, 3540, 6515]
  - count=1 len=5 rep_routes=[4081]
- #6 city=1 n_routes=12 n_patterns=5 lcs_sep=0.750 od_bin=(-4151,2000,-4151,1999)
  - count=5 len=3 rep_routes=[5026, 5883, 6279]
  - count=3 len=1 rep_routes=[3368, 4037, 6907]
  - count=2 len=2 rep_routes=[2734, 3961]
- #7 city=0 n_routes=10 n_patterns=5 lcs_sep=0.500 od_bin=(-4155,2120,-4154,2116)
  - count=5 len=5 rep_routes=[225, 317, 811]
  - count=2 len=6 rep_routes=[1167, 1265]
  - count=1 len=6 rep_routes=[535]
- #8 city=1 n_routes=9 n_patterns=5 lcs_sep=0.667 od_bin=(-4150,2000,-4151,2000)
  - count=4 len=2 rep_routes=[4169, 6777, 6848]
  - count=2 len=2 rep_routes=[3309, 5043]
  - count=1 len=2 rep_routes=[2377]
- #9 city=1 n_routes=8 n_patterns=5 lcs_sep=1.000 od_bin=(-4150,1998,-4150,1999)
  - count=4 len=3 rep_routes=[2438, 2586, 2646]
  - count=1 len=2 rep_routes=[2797]
  - count=1 len=2 rep_routes=[3207]
