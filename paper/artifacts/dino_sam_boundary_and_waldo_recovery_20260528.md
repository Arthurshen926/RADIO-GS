# DINO SAM-Boundary and Waldo Recovery Audit

This audit records two targeted follow-ups after the compact-field Direct3D
row was promoted.

## DINOv3 Multi-Head Readout

Source:
`output/lerf_sam_dino_tasks/formal_v12c_dino_sam3_boundary_v9readout_gpu_20260528/lerf_sam_dino_task_aggregate.json`

The promoted DINO readout keeps DINOv3 adaptor features for correspondence
support, then uses the rendered SAM3-adaptor feature family for feature-only
boundary refinement. It does not call the official RGB SAM decoder.

| Task | Teacher | CTF-GS rendered | Delta | Winner |
| --- | ---: | ---: | ---: | --- |
| DINOv3 mask propagation LocAcc | 0.7660 | 0.7872 | +0.0213 | rendered |
| DINOv3 mask propagation mIoU | 0.4606 | 0.4677 | +0.0071 | rendered |
| DINOv3 dense matching mean score | 0.8547 | 0.9048 | +0.0501 | rendered |
| DINOv3 dense matching HitRate | 0.5723 | 0.5396 | -0.0327 | teacher |

Conclusion: the DINO mask-propagation primary metric is now a student-field
win under the same readout. Dense HitRate remains a secondary caveat and should
not be used for a universal-superiority claim.

Previously trained DINO topology candidates were re-audited before promotion.
The strongest available topology/mask-propagation variants improved some
two-scene diagnostics but did not beat the promoted v12c multi-head readout on
the four-scene primary DINO mask-propagation row, so they remain diagnostics.
The promoted improvement is therefore a method-level multi-head readout rather
than a new scene-specific DINO fine-tuning result.

## Waldo Small-Object Recovery Diagnostic

Baseline source:
`output/radio_gs/lerf_direct3d_prompt_ensemble_score_component_guard_m050_k2_lowthr_20260528/waldo_kitchen/lerf_direct_3d_selection_results.json`

Diagnostic source:
`output/radio_gs/waldo_recovery_heatmap_support_pix8000_thr060_20260528/waldo_kitchen/waldo_kitchen/lerf_direct_3d_selection_results.json`

The diagnostic adds a GT-free low-support heatmap recovery floor before
score-component filtering. It is intended for small/fragmented query masks,
not as a globally promoted policy.

| Row | Waldo mIoU | Waldo Acc@0.25 | Waldo Boundary-F | Waldo Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| Promoted score-component guard | 0.3312 | 0.5455 | 0.3827 | 0.2123 |
| Heatmap recovery diagnostic, 8000 px | 0.3414 | 0.5909 | 0.4535 | 0.2339 |

The diagnostic rescues at least one near-threshold Waldo query and improves
boundary/trimap metrics, supporting the small-object support-recovery
hypothesis. A four-scene smoke test showed that the fixed 8000-pixel floor can
regress Ramen, so this result remains diagnostic until the floor is made
adaptive.
