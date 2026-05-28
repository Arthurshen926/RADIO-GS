# Direct-3D Initial-IoU Bucket Diagnostic

| Source | Scene | Selection | Bucket | n | Initial mIoU | Final mIoU | Delta | Initial BF | Final BF | Delta BF | SAM accept |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| output/radio_gs/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528_direct_prompt_sam3_heatguard_eval/ramen/lerf_direct_3d_selection_results.json | ramen | thr0p15 | lt_025 | 54 | 0.0519 | 0.0478 | -0.0041 | 0.1246 | 0.1204 | -0.0042 | 0.093 |
| output/radio_gs/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528_direct_prompt_sam3_heatguard_eval/ramen/lerf_direct_3d_selection_results.json | ramen | thr0p15 | 025_050 | 14 | 0.3544 | 0.3184 | -0.0360 | 0.4871 | 0.4583 | -0.0287 | 0.214 |
| output/radio_gs/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528_direct_prompt_sam3_heatguard_eval/ramen/lerf_direct_3d_selection_results.json | ramen | thr0p15 | 050_075 | 3 | 0.5708 | 0.5708 | +0.0000 | 0.6156 | 0.6156 | +0.0000 | 0.000 |
| output/radio_gs/lerf_ramen_adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16_20260528_direct_prompt_sam3_heatguard_eval/ramen/lerf_direct_3d_selection_results.json | ramen | thr0p15 | gte_075 | 0 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | 0.0000 | +0.0000 | 0.000 |
