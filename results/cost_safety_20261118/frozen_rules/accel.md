# Wall-clock cost of the verification rule (frozen harmful runs)

tok/s is `n_tokens / seconds` per record, summed within a cell. Every cell uses the same fine-tuned malicious draft, max_new=160, gamma=6, T=1.

| Rule | Gemma_3_4B | MiniCPM4_8B | Qwen3_4B | vs rank-50 |
|---|---:|---:|---:|---:|
| Bare target | 27.7 | 25.9 | 34.3 | – |
| Draft alone | 36.4 | 49.0 | 43.6 | – |
| Rank-50 (K=50) | 19.7 (75%) | 22.6 (92%) | 27.8 (86%) | 1.00× |
| Head-exact (N=10) | 13.6 (68%) | 20.9 (66%) | 20.4 (56%) | 0.69–0.92× |
| Mass floor 1e-3 | 10.0 (46%) | 19.2 (65%) | 13.5 (34%) | 0.48–0.85× |
| Exact | 6.3 (22%) | 11.2 (25%) | 8.9 (17%) | 0.32–0.50× |
| Rank-20 | 12.9 (64%) | 22.6 (83%) | 23.9 (71%) | 0.66–1.00× |
| Rank-100 | 15.8 (83%) | 24.9 (96%) | 28.8 (92%) | 0.80–1.10× |
| Typical A | 8.3 (34%) | 15.0 (48%) | 11.0 (25%) | 0.40–0.67× |
| Typical B | 7.4 (29%) | 13.0 (39%) | 10.1 (21%) | 0.36–0.58× |
| Floor 1e-2 | 8.8 (38%) | 16.1 (52%) | 11.7 (28%) | 0.42–0.72× |
| Floor 1e-3 | 10.1 (46%) | 19.1 (65%) | 13.4 (34%) | 0.48–0.85× |

Numbers in parentheses are the mean draft acceptance rate. `vs rank-50` is tokens-per-second divided by the rank-50 cell of the same model.

