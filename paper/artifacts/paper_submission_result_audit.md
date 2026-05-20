# RADIO-GS Paper Submission Result Audit

This audit lists the exact rendered-feature JSON provenance for every RADIO-GS score used in the submission table.

- Eval search root: output/radio_gs
- Selection rule: frozen mainline scene rows from `output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.json`; component ablations and adaptor candidates are audited as alternatives, not auto-promoted into the main row.

| Scene | Reported | Source JSON | Verified |
|---|---:|---|---|
| Figurines | 0.821 | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/lerf_eval_latest/T50/lerf_ovs_results.json` | YES |
| Ramen | 0.901 | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T40/lerf_ovs_results.json` | YES |
| Teatime | 0.898 | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/lerf_eval_best/T25/lerf_ovs_results.json` | YES |
| Waldo Kitchen | 0.864 | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T25/lerf_ovs_results.json` | YES |

## Source details

### Figurines

- Reported score: 0.821
- Selected rendered JSON: 0.821 / mIoU=0.431 @ output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/lerf_eval_latest/T50/lerf_ovs_results.json (T=50.0, hm=4)
- Selected config: radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml
- Selected checkpoint: output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth
- Best rendered alternative in search root: 0.821 / mIoU=0.434 @ output/radio_gs/figurines_v14_dino_cv001_spatial001_b2_ft20/lerf_eval_best/T50/lerf_ovs_results.json (T=50.0, hm=4)
- Best GT-only alternative in search root: 0.732 / mIoU=0.431 @ output/radio_gs/paper_figures/lerf_teacher_overlay_cpu_20260501/lerf_ovs_results.json (T=50.0, hm=4)
- Action: this scene already has direct rendered JSON backing.

### Ramen

- Reported score: 0.901
- Selected rendered JSON: 0.901 / mIoU=0.586 @ output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T40/lerf_ovs_results.json (T=40.0, hm=4)
- Selected config: radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml
- Selected checkpoint: /root/RADIO-GS/output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth
- Best rendered alternative in search root: 0.901 / mIoU=0.587 @ output/radio_gs/ramen_v14_dino_rel_sam_region_sig005/lerf_eval_latest/T40/lerf_ovs_results.json (T=40.0, hm=4)
- Best GT-only alternative in search root: 0.887 / mIoU=0.589 @ output/radio_gs/lerf_ramen_v14_nofdh_240ep/lerf_eval_best/T40/lerf_ovs_results.json (T=40.0, hm=4)
- Action: this scene already has direct rendered JSON backing.

### Teatime

- Reported score: 0.898
- Selected rendered JSON: 0.898 / mIoU=0.549 @ output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/lerf_eval_best/T25/lerf_ovs_results.json (T=25.0, hm=4)
- Selected config: radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml
- Selected checkpoint: /root/RADIO-GS/output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth
- Best rendered alternative in search root: 0.915 / mIoU=0.530 @ output/radio_gs/lerf_teatime_component_no_refiner_seed7/lerf_eval_latest/T25/lerf_ovs_results.json (T=25.0, hm=4)
- Best GT-only alternative in search root: 0.898 / mIoU=0.522 @ output/radio_gs/lerf_teatime_v14_nofdh_240ep/lerf_eval_best/T30/lerf_ovs_results.json (T=30.0, hm=4)
- Action: this scene already has direct rendered JSON backing.

### Waldo Kitchen

- Reported score: 0.864
- Selected rendered JSON: 0.864 / mIoU=0.411 @ output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T25/lerf_ovs_results.json (T=25.0, hm=4)
- Selected config: radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml
- Selected checkpoint: output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth
- Best rendered alternative in search root: 0.864 / mIoU=0.411 @ output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/lerf_eval_latest/T25/lerf_ovs_results.json (T=25.0, hm=4)
- Best GT-only alternative in search root: 0.727 / mIoU=0.457 @ output/radio_gs/waldo_kitchen_v14_dino_cv001_textsp004_b2_ft30/lerf_eval_best/T40/lerf_ovs_results.json (T=40.0, hm=4)
- Action: this scene already has direct rendered JSON backing.

