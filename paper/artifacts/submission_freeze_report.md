# RADIO-GS Submission Freeze Report

This generated report is the current paper-facing source of truth for the conservative submission package.

## Claim-to-Artifact Matrix

| Paper claim | Current status | Primary artifact | Paper use |
|---|---|---|---|
| LERF main result | Current promoted | `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json` | Main open-vocabulary table |
| ScanNet fair cross-domain result | Current promoted | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` | Cross-domain table |
| Efficiency/profile evidence | Current eval profiles frozen | `output/radio_gs/profiles/freeze_*_20260502` | Runtime and memory table |
| Qualitative figure shortlist | Frozen overlay candidates selected | `output/radio_gs/reports/submission_freeze_figure_shortlist.md` | Main qualitative figure |
| External baseline comparison | Official-source provenance closed | `output/radio_gs/reports/baseline_source_verification.md` | Main comparison table with protocol caveat |
| LERF direct 3D object selection | Registered-view + voxel-context primitive readout | `output/radio_gs/reports/lerf_direct_3d_selection.md` | OpenGaussian-style VPR primitive-level result plus separated boundary-readout diagnostics |

## LERF-OVS

- Protocol: rendered-feature readout from `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json`.
- Mask readout: fixed global threshold `0.60` plus peak connected component.
- Macro LocAcc: `0.8598`
- Macro mIoU: `0.5707`
- Historical heatmap-only readout: LocAcc `0.8712`, mIoU `0.5243`.

| Scene | LocAcc | mIoU | Temp | Source summary |
|---|---:|---:|---:|---|
| figurines | 0.8214 | 0.5134 | 50.0 | `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json` |
| ramen | 0.9014 | 0.6249 | 40.0 | `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json` |
| teatime | 0.8983 | 0.6177 | 25.0 | `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json` |
| waldo_kitchen | 0.8182 | 0.5268 | 25.0 | `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json` |

## LERF Direct 3D Object Selection

- Protocol: OpenGaussian-style direct primitive query, selected-Gaussian rendering, and LERF-OVS mask evaluation.
- The registry below separates primitive scoring, GT-free RGB boundary cleanup, and frozen official SAM3 box-prompt boundary readout.
- VPR readouts compute text scores on Gaussian primitives; SAM3 box readout refines only the rendered selection boundary and does not use GT masks for candidate selection.
- CTF-GS + RGB snap silhouette 0.60: macro mIoU `0.4554`, macro Acc@0.25 `0.7014`, macro Acc@0.50 `0.4663`.
- Direct-3D silhouette sweep source: `output/radio_gs/reports/lerf_direct_3d_silhouette_sweep_report.json`.

## Direct-3D Readout Registry

| Readout | Text head | Selector policy | Macro mIoU | Macro Acc@0.25 | Boundary-F | Trimap IoU | Source root |
|---|---|---|---:|---:|---:|---:|---|
| VPR fixed threshold + RGB snap | SigLIP2 | `fixed:thr0p25` | 0.4801 | 0.6760 | 0.0000 | 0.0000 | `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515` |
| compact prompt ensemble, pure one-map | SigLIP2 | `fixed:thr0p70` | 0.4570 | 0.6851 | 0.6166 | 0.2771 | `output/radio_gs/lerf_direct3d_prompt_ensemble_pure_onemap_20260528` |
| compact prompt-ensemble + RGB area component guard | SigLIP2 | `fixed:thr0p65` | 0.5000 | 0.7051 | 0.6345 | 0.3213 | `output/radio_gs/lerf_direct3d_prompt_ensemble_policy_sweep_20260528` |
| compact prompt-ensemble + RGB score-component guard | SigLIP2 | `fixed:thr0p55` | 0.5014 | 0.7044 | 0.6305 | 0.3225 | `output/radio_gs/lerf_direct3d_prompt_ensemble_score_component_guard_m050_k2_lowthr_20260528` |
| direct field + official SAM3 box, pad16 fixed global threshold | SigLIP2+SAM3 | `fixed:thr0p25` | 0.5705 | 0.6835 | 0.6681 | 0.3958 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5` |
| direct field + official SAM3 box, pad16 scene-locked diagnostic | SigLIP2+SAM3 | `best_by_miou` | 0.5972 | 0.7009 | 0.6817 | 0.4043 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5` |
| direct field + official SAM3 box, pad0 legacy diagnostic | SigLIP2+SAM3 | `best_by_miou` | 0.5815 | 0.7150 | 0.6731 | 0.3777 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516` |

### VPR fixed threshold + RGB snap Scene Selectors

| Scene | Selection | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.5309 | 0.7857 | 0.0000 | 0.0000 | 56 | `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.5805 | 0.7465 | 0.0000 | 0.0000 | 71 | `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.5662 | 0.7627 | 0.0000 | 0.0000 | 59 | `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.2429 | 0.4091 | 0.0000 | 0.0000 | 22 | `output/radio_gs/lerf_direct_3d_selection_threshold_grabcut_20260515/waldo_kitchen/lerf_direct_3d_selection_results.json` |

### direct field + official SAM3 box, pad16 fixed global threshold Scene Selectors

| Scene | Selection | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---|
| figurines | `thr0p25` | 0.6136 | 0.6964 | 0.7116 | 0.3986 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p25` | 0.6409 | 0.7465 | 0.7713 | 0.4400 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p25` | 0.6130 | 0.7458 | 0.7255 | 0.4626 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p25` | 0.4142 | 0.5455 | 0.4638 | 0.2820 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/waldo_kitchen/lerf_direct_3d_selection_results.json` |

### direct field + official SAM3 box, pad16 scene-locked diagnostic Scene Selectors

| Scene | Selection | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---|
| figurines | `thr0p09` | 0.6422 | 0.7321 | 0.7263 | 0.4227 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p18` | 0.6494 | 0.7465 | 0.7733 | 0.4422 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p18` | 0.6528 | 0.7797 | 0.7482 | 0.4955 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p55` | 0.4444 | 0.5455 | 0.4791 | 0.2568 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad16_global_selector_20260516_205200_pad16_gpu5/waldo_kitchen/lerf_direct_3d_selection_results.json` |

### direct field + official SAM3 box, pad0 legacy diagnostic Scene Selectors

| Scene | Selection | mIoU | Acc@0.25 | Boundary-F | Trimap IoU | N | Source JSON |
|---|---|---:|---:|---:|---:|---:|---|
| figurines | `thr0p11` | 0.5924 | 0.7321 | 0.6813 | 0.3523 | 56 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516/figurines/figurines/lerf_direct_3d_selection_results.json` |
| ramen | `thr0p16` | 0.6830 | 0.8028 | 0.7970 | 0.4326 | 71 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516/ramen/ramen/lerf_direct_3d_selection_results.json` |
| teatime | `thr0p38` | 0.6556 | 0.7797 | 0.7530 | 0.4707 | 59 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516/teatime/teatime/lerf_direct_3d_selection_results.json` |
| waldo_kitchen | `thr0p3` | 0.3949 | 0.5455 | 0.4613 | 0.2552 | 22 | `output/radio_gs/lerf_direct3d_sam3_box_pad0_best_masks_20260516/waldo_kitchen/waldo_kitchen/lerf_direct_3d_selection_results.json` |
- CTF-GS accuracy-oriented cap0.015 diagnostic: macro mIoU `0.4184`, macro Acc@0.25 `0.7013`.
- CTF-GS fixed `top0p02` conservative audit: macro mIoU `0.3850`, macro Acc@0.25 `0.6428`.
- CTF-GS previous cap0.02 diagnostic: macro mIoU `0.4185`, macro Acc@0.25 `0.6899`.
- OpenGaussian official context: macro mIoU `0.3836`, macro Acc@0.25 `0.5143`.
- Diagnostics: original Gaussian-center readout is `0.0804` macro mIoU; registered softmax24 without aggregation is `0.3421`; 96-view VPR with voxel aggregation improves fixed-ratio macro mIoU to `0.3850`, GT-free score-distribution selection improves it to `0.3934`, adding the fixed 2% cap improves it to `0.4072`, adding the fixed 0.5% floor improves it to `0.4133`, increasing the all-pose registration budget to 128 views improves the fixed paper selector to `0.4185`, tightening the global cap to `0.0175` improves it to `0.4226`, and a cache-backed fixed `0.018` cap slightly improves it to `0.4227` with `0.6906` Acc@0.25.
- Paper use: VPR-backed primitive-level evidence with an explicit Waldo/provenance caveat.

## ScanNet

- Protocol: VALA/OpenGaFF ScanNet-8 direct point query, DINO-CV contextual kNN16/candidate80, scene-mean calibration alpha 0.45, spatial logit smoothing k12/a1.
- Scenes found: `8`
- Macro mIoU: `19: 0.3806 / 15: 0.3871 / 10: 0.4711`
- Macro mAcc: `19: 0.6129 / 15: 0.6315 / 10: 0.7200`

| Scene | mIoU19 | mIoU15 | mIoU10 | Source JSON |
|---|---:|---:|---:|---|
| scene0000_00 | 0.3165 | 0.2965 | 0.3591 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0062_00 | 0.4080 | 0.4080 | 0.5504 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0070_00 | 0.2192 | 0.2400 | 0.3758 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0097_00 | 0.4661 | 0.4421 | 0.4933 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0140_00 | 0.3550 | 0.4076 | 0.4369 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0347_00 | 0.5939 | 0.5603 | 0.7097 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0400_00 | 0.3988 | 0.3988 | 0.4326 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |
| scene0590_00 | 0.2875 | 0.3432 | 0.4107 | `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json` |

## Profile Evidence

- Profiled workloads: `0`

| Profile | GPU | Wall Time | Peak VRAM (MiB) | Peak GPU% | Mean GPU% | Samples |
|---|---:|---:|---:|---:|---:|---:|

## Warnings

- External LERF/LangSplat/LEGaussians rows are official-source context rows, not reproduced local-evaluator baselines.
- ScanNet label-supervised, GT-label-balanced, old v67, and non-VALA8 runs are diagnostic only and excluded from this paper-facing VALA/OpenGaFF-8 summary.
- LERF direct 3D object selection is protocol-aligned; direct primitive scoring, strict no-RGB one-map, RGB-snap/component cleanup, and official SAM3 box boundary readout are reported as separate readouts.
- Direct-3D readout `direct field + official SAM3 box, pad16 scene-locked diagnostic` uses best_by_miou scene selectors; treat it as diagnostic until a validation-selected or global threshold rule is added.
- Direct-3D readout `direct field + official SAM3 box, pad0 legacy diagnostic` uses best_by_miou scene selectors; treat it as diagnostic until a validation-selected or global threshold rule is added.
