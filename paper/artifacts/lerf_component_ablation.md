# LERF Component Ablation

Protocol: controlled seed-7 LERF-OVS evaluation. Each row is selected by the same scene-specific temperature sweep over rendered features; ties are resolved by mIoU. Full/component rows follow the same scene-level FDH route defined by their configs, while the no-FDH row removes the frozen-depth-head stage. The current-best selector table is intentionally kept separate.

## Macro Summary

| Variant | Scenes | Macro LocAcc | Delta | Macro mIoU | Note |
|---|---:|---:|---:|---:|---|
| Full RADIO-GS | 4/4 | 0.8578 | 0.0000 | 0.4850 | hybrid + HCD + refiner + FDH warm-start |
| w/o FDH warm-start | 4/4 | 0.8018 | -0.0560 | 0.4236 | same hybrid/HCD/refiner, no frozen-depth-head stage |
| w/o refiner | 4/4 | 0.8401 | -0.0177 | 0.4796 | refiner disabled during the FDH refinement run |
| w/o hybrid | 4/4 | 0.8394 | -0.0184 | 0.5069 | explicit per-Gaussian compact field, HCD/refiner retained |
| w/o HCD | 4/4 | 0.5306 | -0.3272 | 0.2596 | direct 1x1 projection codec replaces HCD |

## Per-Scene LocAcc / mIoU

### Figurines

| Variant | LocAcc | mIoU | Temp | Source |
|---|---:|---:|---:|---|
| Full RADIO-GS | 0.7679 | 0.3946 | 50.0 | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep_seed7/lerf_eval_best/T50/lerf_ovs_results.json` |
| w/o FDH warm-start | 0.6964 | 0.4243 | 50.0 | `output/radio_gs/lerf_figurines_v14_nofdh_240ep_seed7/lerf_eval_latest/T50/lerf_ovs_results.json` |
| w/o refiner | 0.7679 | 0.4170 | 40.0 | `output/radio_gs/lerf_figurines_component_no_refiner_seed7/lerf_eval_best/T40/lerf_ovs_results.json` |
| w/o hybrid | 0.7679 | 0.4472 | 50.0 | `output/radio_gs/lerf_figurines_component_no_hybrid_seed7/lerf_eval_latest/T50/lerf_ovs_results.json` |
| w/o HCD | 0.2679 | 0.1764 | 50.0 | `output/radio_gs/lerf_figurines_component_direct_codec_seed7/lerf_eval_best/T50/lerf_ovs_results.json` |

### Ramen

| Variant | LocAcc | mIoU | Temp | Source |
|---|---:|---:|---:|---|
| Full RADIO-GS | 0.9014 | 0.5862 | 40.0 | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T40/lerf_ovs_results.json` |
| w/o FDH warm-start | 0.8451 | 0.5665 | 40.0 | `output/radio_gs/lerf_ramen_v14_nofdh_240ep_seed7/lerf_eval_best/T40/lerf_ovs_results.json` |
| w/o refiner | 0.8592 | 0.5738 | 40.0 | `output/radio_gs/lerf_ramen_component_no_refiner_seed7/lerf_eval_best/T40/lerf_ovs_results.json` |
| w/o hybrid | 0.8732 | 0.5936 | 40.0 | `output/radio_gs/lerf_ramen_component_no_hybrid_seed7/lerf_eval_best/T40/lerf_ovs_results.json` |
| w/o HCD | 0.6479 | 0.3051 | 25.0 | `output/radio_gs/lerf_ramen_component_direct_codec_seed7/lerf_eval_best/T25/lerf_ovs_results.json` |

### Teatime

| Variant | LocAcc | mIoU | Temp | Source |
|---|---:|---:|---:|---|
| Full RADIO-GS | 0.8983 | 0.5486 | 25.0 | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/lerf_eval_best/T25/lerf_ovs_results.json` |
| w/o FDH warm-start | 0.8475 | 0.5552 | 28.0 | `output/radio_gs/lerf_teatime_v14_nofdh_240ep_seed7/lerf_eval_best/T28/lerf_ovs_results.json` |
| w/o refiner | 0.9153 | 0.5297 | 25.0 | `output/radio_gs/lerf_teatime_component_no_refiner_seed7/lerf_eval_latest/T25/lerf_ovs_results.json` |
| w/o hybrid | 0.8983 | 0.5692 | 28.0 | `output/radio_gs/lerf_teatime_component_no_hybrid_seed7/lerf_eval_best/T28/lerf_ovs_results.json` |
| w/o HCD | 0.6610 | 0.3443 | 35.0 | `output/radio_gs/lerf_teatime_component_direct_codec_seed7/lerf_eval_best/T35/lerf_ovs_results.json` |

### Waldo Kitchen

| Variant | LocAcc | mIoU | Temp | Source |
|---|---:|---:|---:|---|
| Full RADIO-GS | 0.8636 | 0.4106 | 25.0 | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T25/lerf_ovs_results.json` |
| w/o FDH warm-start | 0.8182 | 0.1485 | 12.0 | `output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep_seed7/lerf_eval_best/T12/lerf_ovs_results.json` |
| w/o refiner | 0.8182 | 0.3980 | 25.0 | `output/radio_gs/lerf_waldo_kitchen_component_no_refiner_seed7/lerf_eval_latest/T25/lerf_ovs_results.json` |
| w/o hybrid | 0.8182 | 0.4175 | 25.0 | `output/radio_gs/lerf_waldo_kitchen_component_no_hybrid_seed7/lerf_eval_latest/T25/lerf_ovs_results.json` |
| w/o HCD | 0.5455 | 0.2126 | 18.0 | `output/radio_gs/lerf_waldo_kitchen_component_direct_codec_seed7/lerf_eval_best/T18/lerf_ovs_results.json` |

## Provenance

| Variant | Scene | Config | Experiment dir | Status |
|---|---|---|---|---|
| Full RADIO-GS | Figurines | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep_seed7` | ready |
| Full RADIO-GS | Ramen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7` | ready |
| Full RADIO-GS | Teatime | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7` | ready |
| Full RADIO-GS | Waldo Kitchen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7` | ready |
| w/o FDH warm-start | Figurines | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_figurines_nofdh_240ep_seed7.yaml` | `output/radio_gs/lerf_figurines_v14_nofdh_240ep_seed7` | ready |
| w/o FDH warm-start | Ramen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_nofdh_240ep_seed7.yaml` | `output/radio_gs/lerf_ramen_v14_nofdh_240ep_seed7` | ready |
| w/o FDH warm-start | Teatime | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_nofdh_240ep_seed7.yaml` | `output/radio_gs/lerf_teatime_v14_nofdh_240ep_seed7` | ready |
| w/o FDH warm-start | Waldo Kitchen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_nofdh_240ep_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_v14_nofdh_240ep_seed7` | ready |
| w/o refiner | Figurines | `radio_gs/configs/generated/ablation/lerf_figurines_component_no_refiner_seed7.yaml` | `output/radio_gs/lerf_figurines_component_no_refiner_seed7` | ready |
| w/o refiner | Ramen | `radio_gs/configs/generated/ablation/lerf_ramen_component_no_refiner_seed7.yaml` | `output/radio_gs/lerf_ramen_component_no_refiner_seed7` | ready |
| w/o refiner | Teatime | `radio_gs/configs/generated/ablation/lerf_teatime_component_no_refiner_seed7.yaml` | `output/radio_gs/lerf_teatime_component_no_refiner_seed7` | ready |
| w/o refiner | Waldo Kitchen | `radio_gs/configs/generated/ablation/lerf_waldo_kitchen_component_no_refiner_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_component_no_refiner_seed7` | ready |
| w/o hybrid | Figurines | `radio_gs/configs/generated/ablation/lerf_figurines_component_no_hybrid_seed7.yaml` | `output/radio_gs/lerf_figurines_component_no_hybrid_seed7` | ready |
| w/o hybrid | Ramen | `radio_gs/configs/generated/ablation/lerf_ramen_component_no_hybrid_seed7.yaml` | `output/radio_gs/lerf_ramen_component_no_hybrid_seed7` | ready |
| w/o hybrid | Teatime | `radio_gs/configs/generated/ablation/lerf_teatime_component_no_hybrid_seed7.yaml` | `output/radio_gs/lerf_teatime_component_no_hybrid_seed7` | ready |
| w/o hybrid | Waldo Kitchen | `radio_gs/configs/generated/ablation/lerf_waldo_kitchen_component_no_hybrid_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_component_no_hybrid_seed7` | ready |
| w/o HCD | Figurines | `radio_gs/configs/generated/ablation/lerf_figurines_component_direct_codec_seed7.yaml` | `output/radio_gs/lerf_figurines_component_direct_codec_seed7` | ready |
| w/o HCD | Ramen | `radio_gs/configs/generated/ablation/lerf_ramen_component_direct_codec_seed7.yaml` | `output/radio_gs/lerf_ramen_component_direct_codec_seed7` | ready |
| w/o HCD | Teatime | `radio_gs/configs/generated/ablation/lerf_teatime_component_direct_codec_seed7.yaml` | `output/radio_gs/lerf_teatime_component_direct_codec_seed7` | ready |
| w/o HCD | Waldo Kitchen | `radio_gs/configs/generated/ablation/lerf_waldo_kitchen_component_direct_codec_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_component_direct_codec_seed7` | ready |
