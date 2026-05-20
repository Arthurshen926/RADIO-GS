# Efficiency Profile Summary

| Profile | Workload | Wall Time | Peak GPU Mem (MiB) | Peak GPU Util (%) | Mean GPU Util (%) | Notes |
|---|---|---:|---:|---:|---:|---|
| `output/radio_gs/profiles/figurines_fdh_ws240_sweep_fixed` | Figurines FDH best-checkpoint LERF sweep (T25-T50) | 101.636 s | 1568 | 12 | 0.43 | shell time fallback |
| `output/radio_gs/profiles/waldo_kitchen_fdh_ws240_240ep_lerf_autoeval` | Waldo Kitchen FDH full auto-eval (best+latest, T10-T25) | 215.057 s | 2076 | 38 | 1.39 | shell time fallback |
| `output/radio_gs/profiles/ramen_nofdh_240ep_lerf_autoeval` | Ramen noFDH full auto-eval (best+latest, T20-T40) | 173.292 s | 1762 | 16 | 1.09 | shell time fallback |
| `output/radio_gs/profiles/room0_pure_frozen_depth_only_autoeval` | Room0 pure_frozen_depth_only full auto-eval (latest, eval_seed 42) | 570.981 s | 4071 | 90 | 37.63 | exact profile |
