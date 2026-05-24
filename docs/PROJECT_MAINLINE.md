# RADIO-GS Project Mainline

Status: 2026-05-14 conservative submission mainline with calibrated rendered grounding, VPR direct-selection upgrade, and adaptor diagnostics.

This document is the navigation layer for the cleaned project. It separates the
current strongest paper route from historical validation branches.

## One-Sentence Thesis

The paper-facing method, **CTF-GS**, compactly distills frozen RADIO teacher
features into 3D Gaussian scenes so that novel views can render reusable
teacher-compatible feature maps for open-vocabulary grounding and cross-domain
scene understanding, while View-to-Primitive Registration (VPR) exposes a
registered Gaussian-level readout for direct 3D querying. `RADIO-GS` remains the repository/project
implementation name.

## Current Strongest Mainline

The active paper route is:

1. Frozen 3DGS geometry.
2. **HGCF**: Hybrid Gaussian Code Field with per-Gaussian latent storage and a
   coarse spatial branch.
3. **CTR**: Compact-to-Teacher Reconstruction from compact features back to
   1280d RADIO space (implemented as the HCD codec).
4. **VFA**: View-Space Feature Alignment (implemented as screen-space feature
   refinement).
5. **FGC**: Frozen Geometry-Head Consistency warm-start training as the main
   geometry-aware regularizer (formerly FDH warm-start).
6. LERF-OVS as the primary benchmark.
7. ScanNet v67 direct point query as cross-domain feature-usability evidence.
8. LERF-OVS direct 3D object selection via **VPR** rendered-feature-to-primitive
   registration under an OpenGaussian-style query-select-render protocol.
9. Formal profile runs for evaluation runtime and peak VRAM.
10. Optional RADIO adaptor/cross-view consistency: DINOv3 relation + SAM3
   soft-region alignment for Ramen/Teatime, and DINOv3 cross-view + spatial
   text-heatmap preservation for Figurines.

## Source-of-Truth Files

Use these files first:

| Purpose | Path |
|---|---|
| Submission status | `docs/submission_status.md` |
| Current prose draft | `docs/paper_draft_current.md` |
| LaTeX draft | `paper/radio_gs_draft.tex` |
| Freeze report | `output/radio_gs/reports/submission_freeze_report.md` |
| Machine-readable manifest | `output/radio_gs/reports/submission_freeze_manifest.json` |
| Baseline provenance risk | `output/radio_gs/reports/baseline_source_verification.md` |
| Profile summary | `output/radio_gs/reports/submission_freeze_profile_summary.md` |
| Figure shortlist | `output/radio_gs/reports/submission_freeze_figure_shortlist.md` |
| Storage footprint | `output/radio_gs/reports/storage_footprint_report.md` |
| Compression/downstream correlation | `output/radio_gs/reports/compression_downstream_correlation.md` |
| Feature error vs text relevance | `output/radio_gs/reports/feature_error_text_relevance_report.md` |
| Boundary-error readout | `output/radio_gs/reports/boundary_error_readout_report.md` |
| Alpha/depth boundary alignment coverage | `output/radio_gs/reports/alpha_depth_boundary_alignment_report.md` |
| Alpha/depth boundary case figure | `paper/figures/alpha_depth_boundary_cases.png` |
| SAM3-box geometry-map sweep | `output/radio_gs/reports/lerf_sam3_box_global_threshold_sweep_20260517_geometry.md` |
| Training-entry auditability report | `output/radio_gs/reports/train_feature_field_audit.md` |
| LERF failure analysis | `output/radio_gs/reports/lerf_failure_analysis.md` |
| Waldo failure stratification | `output/radio_gs/reports/waldo_failure_stratification.md` |
| Direct3D confidence/coverage analysis | `output/radio_gs/reports/lerf_direct3d_confidence_coverage_analysis.md` |
| LERF direct 3D selection | `output/radio_gs/reports/lerf_direct_3d_selection.md` |
| LERF direct 3D debug audit | `output/radio_gs/reports/lerf_direct_3d_debug_audit.md` |
| LERF rendered internal SAM3 boundary readout | `paper/artifacts/sam3_prompt_mask_head_lerf2d_support_readout_20260523.md` |
| LERF category-macro stability audit | `output/radio_gs/reports/lerf2d_activeC_category_macro_stability_20260524.md` |
| Direct field joint 2D/3D optimization audit | `docs/experiments/2026-05-24-direct-field-joint2d3d-optimization.md` |
| Latest expert follow-up audit | `docs/experiments/2026-05-16-expert-latest-followup-audit.md` |
| Controlled evidence table | `output/radio_gs/reports/controlled_evidence_table.md` |
| Nearest-view cache baseline | `output/radio_gs/reports/lerf_nearest_view_cache_baseline.md` |
| Per-Gaussian 1280-D explicit baseline | `output/radio_gs/reports/lerf_per_gaussian_1280d_baseline.md` |
| Controlled baseline gap audit | `output/radio_gs/reports/controlled_baseline_gap_audit.md` |

## Active Quantitative Claims

### LERF-OVS

Use `output/radio_gs/reports/lerf_rendered_grounding_paper_ckpt_threshold_sweep.json`
at the GT-free threshold-0.60 readout. The threshold-0.50 CSV remains the
historical default readout, not the paper-facing mask metric.

| Scene | LocAcc | mIoU | Temperature |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.4244 | 50 |
| Ramen | 0.9014 | 0.6201 | 40 |
| Teatime | 0.8983 | 0.5760 | 25 |
| Waldo Kitchen | 0.8636 | 0.4769 | 25 |
| Macro | 0.8712 | 0.5243 | - |

The same calibrated readout gives a stronger same-evaluator teacher-vs-rendered
comparison: frame-wise RADIO RGB is 0.7985 LocAcc / 0.4634 mIoU, while CTF-GS
rendered features are 0.8712 / 0.5243.

A feature-only prompt-conditioned SAM3 boundary readout has been implemented as
a rendered-view ablation, trained from official SAM3 pseudo masks on unlabelled
training views and evaluated without RGB SAM3 calls. The stable active-run
support/area gate improves weighted sample mIoU from 0.5493 to 0.5666 while
leaving LocAcc unchanged, and removes the previous Ramen negative delta. It is
still positioned as a boundary-readout ablation rather than an unconditional
replacement for the main category-macro LERF table. The stability audit reports
0.8598/0.5511 scene-macro LocAcc/mIoU, 0.8702/0.5666 sample-weighted
LocAcc/mIoU, and 0.8271/0.5141 scene-category-macro LocAcc/mIoU, with a
category-minus-sample mIoU gap of -0.0525. See
`paper/artifacts/sam3_prompt_mask_head_lerf2d_support_readout_20260523.md` and
`output/radio_gs/reports/lerf2d_activeC_category_macro_stability_20260524.md`.

The controlled evidence table also includes a measured nearest-view RADIO cache
baseline. For each target annotation frame, it uses the closest cached RADIO
feature frame by camera-center distance, excludes the target frame itself, does
not warp the feature map, and then runs the same LERF text scorer and mask
evaluator. This cache-only control reaches 0.2722 macro LocAcc / 0.1545 macro
mIoU, far below the rendered CTF-GS row, so it should be framed as evidence
that a simple 2D feature cache is not a substitute for the reconstructed 3D
feature field.

The same controlled table now includes the full per-Gaussian 1280-D explicit
RADIO-memory baseline. Cached teacher features are registered to visible
Gaussian centers, stored as fp16 1280-D vectors, rendered to the annotated LERF
views, and scored with the same frozen SigLIP2 evaluator. It reaches 0.5642
macro LocAcc / 0.3182 macro mIoU with 0.2020 mean registered-Gaussian fraction
and 1039.7 MiB mean fp16 feature storage. This closes the raw-feature
scene-memory control, while the compact CTF-GS row remains stronger at 0.8712 /
0.5243 and uses 3.03x less mean checkpoint storage.

A GT-free adaptive mean+std threshold readout was tested as a boundary-refinement
diagnostic:
`output/radio_gs/reports/lerf_rendered_grounding_adaptive_threshold_diagnostic.md`.
With `k=1.0` and a [0.50, 0.70] clamp it keeps LocAcc at 0.8712 but lowers
macro mIoU to 0.4939, so the fixed global threshold-0.60 readout remains the
paper-facing result.

### LERF-OVS Adaptor-Enhanced Candidate

This candidate uses the DINO cross-view + spatial text-heatmap checkpoint for
Figurines, DINO relation + SAM3 region checkpoints for Ramen and Teatime, and
keeps the baseline checkpoint for Waldo Kitchen. It should be reported as an
adaptor/cross-view ablation candidate unless the paper explicitly adopts it as
the main selector.

| Scene | Selected branch | LocAcc | mIoU | Temperature |
|---|---|---:|---:|---:|
| Figurines | DINO cv + spatial text heatmap | 0.8214 | 0.4343 | 50 |
| Ramen | DINO relation + SAM3 region | 0.9014 | 0.5873 | 40 |
| Teatime | DINO relation + SAM3 region | 0.8983 | 0.5592 | 28 |
| Waldo Kitchen | Baseline | 0.8636 | 0.4106 | 25 |
| Macro | - | 0.8712 | 0.4979 | - |

The failed adaptor branches are still informative: Figurines gains mIoU with
strong relation/region and query-distribution heatmap weights but loses
localization, while the conservative spatial heatmap branch is the only
Figurines branch promoted by the preserve-LocAcc rule. Waldo Kitchen does not
benefit from adaptor/cross-view supervision yet.

### LERF-OVS Direct 3D Object Selection

Use `output/radio_gs/reports/lerf_direct_3d_selection.md`.
Use `output/radio_gs/reports/vpr_protocol_card.md` for the protocol card and
`output/radio_gs/reports/vpr_contribution_weighting_ablation.md` for the
Dr. Splat-inspired registration-weighting ablation.
Use `output/radio_gs/reports/lerf_direct_3d_published_context.md` for the
published-context table against newer primitive-/instance-aware methods.

This experiment follows an OpenGaussian-style query-select-render protocol:
text similarity is computed on 3D Gaussian primitives, selected primitives are
rendered into binary masks, and evaluation uses LERF-OVS object masks only at
the final metric stage. The current strict primitive-mask row is the VPR/RGB
snap result with a fixed global score threshold. The compact direct-field rows
with official SAM3 box-prompt readout are boundary-readout probes: SAM3
candidates are selected by overlap with the rendered prediction, not by ground
truth. The latest global-threshold SAM3-box sweep adds a strict fixed `thr0p25`
pad16 row; scene-locked and best-fixed variants remain diagnostics.

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | VPR + fixed score threshold 0.25 + RGB snap mIoU | 0.5309 | 0.5805 | 0.5662 | 0.2429 | 0.4801 |
| CTF-GS | SigLIP2+SAM3 | compact direct field + official SAM3 box readout mIoU, pad16 fixed `thr0p25` | 0.6136 | 0.6409 | 0.6130 | 0.4142 | 0.5705 |
| CTF-GS | SigLIP2+SAM3 | compact direct field + official SAM3 box readout mIoU, pad0 diagnostic | 0.5924 | 0.6830 | 0.6556 | 0.3949 | 0.5815 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene mIoU | 0.5327 | 0.5822 | 0.5662 | 0.2463 | 0.4819 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 mIoU | 0.4055 | 0.4491 | 0.4862 | 0.1991 | 0.3850 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | VPR + fixed score threshold 0.25 + RGB snap Acc@0.25 | 0.7857 | 0.7465 | 0.7627 | 0.4091 | 0.6760 |
| CTF-GS | SigLIP2+SAM3 | compact direct field + official SAM3 box readout Acc@0.25, pad16 fixed `thr0p25` | 0.6964 | 0.7465 | 0.7458 | 0.5455 | 0.6835 |
| CTF-GS | SigLIP2+SAM3 | compact direct field + official SAM3 box readout Acc@0.25, pad0 diagnostic | 0.7321 | 0.8028 | 0.7797 | 0.5455 | 0.7150 |
| CTF-GS | SigLIP2 | diagnostic best-by-scene Acc@0.25 | 0.7679 | 0.7465 | 0.7627 | 0.4091 | 0.6715 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 Acc@0.25 | 0.6786 | 0.7324 | 0.7966 | 0.3636 | 0.6428 |

Published-context rows from newer primitive-/instance-aware methods are kept in
a separate context table rather than the strict local table and are now aligned
to the OpenGaFF comparison route: Dr. Splat 43.29/64.30, InstanceGaussian
45.30/58.44, and OpenGaFF 54.36/80.84 mIoU/Acc@0.25. CAGS is not used as a
main comparison target. The strict direct-field + SAM3-box row is above
OpenGaFF on mIoU but below it on Acc@0.25, so the paper should claim
mIoU-competitive/stronger direct-3D evidence rather than universal dominance.

The latest follow-up diagnostics show that rendered-feature registration is a
strong primitive scoring path, while official SAM3 box-prompt readout fixes the
main direct-field boundary/fragmentation gap without changing the selected 3D
primitive scores. Under the new strict fixed-global SAM3-box readout, pad16 at
`thr0p25` reaches 0.5705 macro mIoU / 0.6835 Acc@0.25 / 0.6081 Acc@0.50, above
the VPR/RGB-snap row and with the strongest tested fixed-threshold mIoU. The
post-hoc best fixed pad16 threshold (`thr0p18`) reaches 0.5863 mIoU, and the
scene-locked pad16 upper bound reaches 0.5972 mIoU / 0.7009 Acc@0.25 with
0.6817 boundary-F and 0.4043 trimap IoU. These latter rows should remain
appendix-only unless selected on held-out validation scenes.

The earlier Gaussian-center direct readout was 0.0804 macro
mIoU / 0.0932 macro Acc@0.25 under its top10%
selector; under the same top2% selector used by the VPR rows it is 0.012 /
0.009. Registered softmax24 reached 0.3421 / 0.5547, registered+voxel with the
fixed top2% selector reached 0.3850 / 0.6428. Under the same 128-view VPR
readout, the current reproducible mean+std selector with fixed 0.5% floor,
1.8% cap, and RGB snap reaches 0.4461 / 0.6611; replacing it with a single
global softmax-score threshold of 0.25 improves the paper-facing row to
0.4801 / 0.6760. This exceeds the OpenGaussian official macro reference. Under
the OpenGaFF published-context route, the stronger fixed SAM3-box row also
exceeds OpenGaFF's reported direct-3D mIoU while trailing its Acc@0.25; Waldo
Kitchen remains the limiting scene.

The Dr. Splat-inspired registration audit now includes both center-sampled
weights and rasterizer-level Gaussian-pixel hit assignment. Under the same
96-view VPR + voxel-max + fixed top-2% protocol, center-sampled alpha weighting
drops macro mIoU/Acc@0.25 to 0.2978/0.5389 and alpha-depth weighting drops to
0.2967/0.5345, compared with the uniform top2% VPR baseline at 0.3850/0.6428
and the promoted 128-view threshold-0.25 + RGB snap VPR selector at
0.4801/0.6760. The true rasterization paths are also negative on Figurines:
all-footprint uniform raster hits reach 0.0002 mIoU, per-pixel dominant
alpha-depth hits reach 0.0178 mIoU under the 128-view budget, and per-Gaussian
top-footprint alpha hits reach 0.0004 mIoU under official views. Proposal/OPR
component selection on the cached strong VPR scores drops Figurines to 0.0430
mIoU. The paper should keep center-based uniform VPR plus global score-threshold
selection as the main primitive readout and report raster/proposal branches as implemented
negative diagnostics.

Additional component/voxel diagnostics from the Expert (5) pass are also
negative for the main Waldo bottleneck. Top-score component filtering with one
or two retained components reduces Waldo mIoU to 0.1757/0.1827, `voxel_mean`
reduces it to 0.0931 under the earlier paper selector, and `voxel_max_dilate`
reduces it to 0.1286. The current 128-view VPR + threshold-0.25 + RGB snap
protocol is the strongest fixed, GT-free direct-3D readout.

VPR-to-field consistency was upgraded with GT-free registration-confidence
weights. The training loss weights registered primitives by the normalized
`log1p(view_counts)` support from the VPR feature cache, so Gaussians seen
consistently across more registration views exert more influence while no LERF
mask is used. Under the same threshold-sweep diagnostic protocol, the direct
field improves from the previous VPR-to-field row at 0.4119 mIoU / 0.5876
Acc@0.25 to 0.4363 / 0.6191. The gain is strongest on Ramen
(`0.4381 -> 0.5975`) and Waldo (`0.2115 -> 0.2245`), mildly positive on
Teatime (`0.5103 -> 0.5196`), and negative on Figurines
(`0.4877 -> 0.4037`). This is promoted as a field-quality improvement and
reported separately from the still-stronger streamed registered VPR readout
at 0.4799 / 0.6760. Two follow-up sweeps were negative: lowering the
adapter blend to 0.3 reduced Ramen/Figurines, and a `log` weight floor of 2.0
did not recover Figurines while reducing Ramen.
The compression/downstream mechanism audit in
`output/radio_gs/reports/compression_downstream_correlation.md` explicitly
separates compactness from downstream robustness. Checkpoint saving ratio has
only weak positive correlation with rendered mIoU (Pearson r=0.3606) and a
negative correlation with Direct3D mIoU (r=-0.6158), mainly because Waldo has
the highest storage saving but the weakest direct-query result. The paper
should therefore present compact storage and direct-3D robustness as separate
claims rather than implying compression causes better querying.
The feature-error/text-relevance audit in
`output/radio_gs/reports/feature_error_text_relevance_report.md` connects
global reconstruction quality to rendered grounding errors: `1 - best validation
cos_decoded` correlates with rendered mIoU error at Pearson r=0.9568 and with
LocAcc error at r=0.8713 across the four frozen LERF scenes. This supports the
feature reconstruction thesis, while remaining a scene-level mechanism audit
rather than a per-query causal proof.
The RGB-snap query audit in
`output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md`
confirms that Waldo Kitchen remains the failure scene: 0.2429 mIoU, 0.4091
Acc@0.25, 0.1818 zero-prediction rate, and a 95% bootstrap mIoU interval of
[0.1487, 0.3403].
The confidence/coverage mechanism report in
`output/radio_gs/reports/lerf_direct3d_confidence_coverage_analysis.md` adds a
GT-free explanation layer: scene-level mean valid VPR views correlate with
strict Direct3D mIoU at Pearson r=0.7588, and query instances in the highest
teacher-score bucket reach 0.6358 mean IoU / 0.8551 Acc@0.25 versus 0.4345 /
0.6143 in the lowest bucket. Text-margin stratification adds the ambiguity
view: distinct-margin queries reach 0.6202 / 0.8261 versus 0.4329 / 0.6286 for
ambiguous queries. This supports discussing Waldo failures as a
coverage/confidence/ambiguity bottleneck without using LERF masks for
selection.
The boundary-error readout in
`output/radio_gs/reports/boundary_error_readout_report.md` adds a measured
boundary layer for the strict pad16 SAM3-box row. Across 208 query instances,
IoU correlates with boundary-F at Pearson r=0.9148 and with trimap IoU at
r=0.8412. Balanced-size predictions have low mean boundary error (0.0637),
whereas under-selection and over-selection have much higher errors (0.7042 and
0.6919). The alpha/depth boundary-alignment instrumentation is now populated
for the strict pad16 Direct3D rerun via `--save_geometry_maps`, and
`output/radio_gs/reports/alpha_depth_boundary_alignment_report.md` records
208/208 query-level geometry-map overlays. Boundary error has weak positive
correlation with alpha/depth discontinuity statistics, and the high
discontinuity bucket has lower mean IoU and higher boundary error than the low
bucket. This supports boundary-mechanism discussion, not a causal
occlusion/discontinuity proof.
The compact qualitative subset for this diagnostic is now generated at
`paper/figures/alpha_depth_boundary_cases.png`, with selected case provenance in
`output/radio_gs/reports/alpha_depth_boundary_case_figure_manifest.md`.

### Training-Code Auditability

Use `output/radio_gs/reports/train_feature_field_audit.md`. The current
training entry point has the important paper-freeze guards in place: run
manifest, git metadata, artifact paths, metrics history, train/val split
resolution, trusted checkpoint loading, and an output training lock. The audit
now marks the release state as `pass`: `radio_gs/scripts/train_feature_field.py`
is reduced to 3735 lines, the extracted support modules live under
`radio_gs/training/`, and feature/text/cache tensor loads go through
`load_training_tensor_cache`.

### DINOv3/SAM3 Downstream Adaptor Probes

Use `output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`.
These results compare original RADIO RGB features (`Teacher`) against
RADIO-GS rendered features (`Rendered`) after frozen DINOv3/SAM3 adaptor
projection. They are diagnostic and should not replace the LERF text-grounding
main result.

| Adaptor | Probe | Teacher LocAcc | Teacher mIoU | Rendered LocAcc | Rendered mIoU |
|---|---|---:|---:|---:|---:|
| DINOv3 | prototype segmentation | 0.6543 | 0.0945 | 0.6277 | 0.0937 |
| DINOv3 | source-target matching | 0.5957 | 0.1032 | 0.5035 | 0.1019 |
| SAM3 | prototype segmentation | 0.8404 | 0.0757 | 0.6649 | 0.0564 |
| SAM3 | source-target matching | 0.7872 | 0.0953 | 0.7092 | 0.0687 |

Qualitative examples are in
`paper/figures/lerf_adaptor_downstream_qualitative.png`. The aggregate takeaway
for these preliminary prototype probes is mixed: rendered DINOv3 is close in
mIoU and has positive Waldo Kitchen matching cases, while unconstrained SAM3
prototype scoring still exposes a teacher-rendered gap.

The formal promptable task sweep upgrades the SAM3/DINOv3 evidence from
prototype probes to downstream-style tasks. Use
`output/lerf_sam_dino_tasks/formal_v9_dino_topk_area200_bg110_peak_20260514/lerf_sam_dino_task_report.md`.
The readout sweep is recorded at
`output/lerf_sam_dino_tasks/formal_v9_dino_readout_sweep_20260514.md`.

| Task | Teacher LocAcc/Hit | Teacher mIoU/Score | Rendered LocAcc/Hit | Rendered mIoU/Score |
|---|---:|---:|---:|---:|
| SAM3 point prompt | 1.0000 | 0.3700 | 1.0000 | 0.4169 |
| SAM3 box prompt | 0.8702 | 0.6560 | 0.8221 | 0.6638 |
| SAM3 mask propagation | 0.7872 | 0.3583 | 0.6667 | 0.3756 |
| DINOv3 dense matching | 0.5723 | 0.8547 | 0.5393 | 0.9048 |
| DINOv3 mask propagation + bg-suppressed readout | 0.7660 | 0.5119 | 0.7943 | 0.4805 |

Rendered CTF-GS features now exceed the frame-wise teacher on SAM3-adaptor
mask mIoU for all three prompt modes. The robust DINO readout combines
source-background contrast, foreground top-k pooling, 2.0x area scaling, and
peak-component cleanup. The v9 background-suppressed variant raises rendered
DINO mask-propagation from the v6 robust row of 0.7730/0.4456 to
0.7943/0.4805 LocAcc/mIoU. This now essentially matches the previous v6 teacher
mIoU reference (0.4806) and improves rendered LocAcc, but the same v9 readout
also improves teacher to 0.7660/0.5119. The paper claim should therefore be
"improves SigLIP2 grounding and SAM3-adaptor region mIoU, gives a DINO rendered
LocAcc advantage, and substantially narrows the DINO propagation mIoU gap."

The mutual matching + homography RANSAC diagnostic is recorded at
`output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_report.md`.
It improves the visual reliability of DINO matches by reducing outliers and
raising rendered dense-match similarity to 0.9277, but the rendered DINO mask
propagation row remains below the teacher under the same readout
(v9: 0.4805 vs. 0.5119 mIoU). Use it as qualitative/diagnostic evidence rather
than as a main superiority claim.

### ScanNet VALA/OpenGaFF-8 Direct Point Query

Use `paper/artifacts/scannet_pointcloud_radio_gs_vala8_direct_point_query_results.json`,
`paper/artifacts/scannet_pointcloud_radio_gs_vala8_contextual_knn_scene_mean_a05_results.json`,
and `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn_scene_mean_a05_results.json`.

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3583 | 0.6006 |
| 15 classes | 0.3618 | 0.6152 |
| 10 classes | 0.4367 | 0.6998 |

The stronger direct point-readout support row now uses the DINO-CV compact
field with the same contextual kNN readout:
`query_mode=knn`, `k=8`, `candidate_k=32`, `logit_calibration=scene_mean`,
`alpha=0.5`, prompt template `{query}`.

| Split | contextual kNN mIoU | contextual kNN mAcc |
|---|---:|---:|
| 19 classes | 0.3704 | 0.6017 |
| 15 classes | 0.3771 | 0.6198 |
| 10 classes | 0.4585 | 0.7032 |

This is the current strongest balanced ScanNet evidence. A more aggressive
DINO-CV `alpha=0.75` setting raises split10 mIoU to 0.4612 but lowers split19/15
mIoU and mAcc, so it stays diagnostic.

Label-free ScanNet prompt/calibration ablations:

| Variant | split19 | split15 | split10 | Use |
|---|---:|---:|---:|---|
| v67 baseline | 0.3583 / 0.6006 | 0.3618 / 0.6152 | 0.4367 / 0.6998 | Main conservative row on VALA/OpenGaFF-8 |
| scene-mean calibration, alpha=0.5 | 0.3575 / 0.6101 | 0.3604 / 0.6227 | 0.4353 / 0.7074 | Positive supporting ablation |
| v67 kNN contextual readout + scene-mean alpha=0.5 | 0.3677 / 0.5997 | 0.3748 / 0.6181 | 0.4562 / 0.7008 | Previous strongest support row |
| DINO-CV kNN contextual readout + scene-mean alpha=0.5 | 0.3704 / 0.6017 | 0.3771 / 0.6198 | 0.4585 / 0.7032 | Promoted balanced support row on VALA/OpenGaFF-8 |
| DINO-CV kNN contextual readout + scene-mean alpha=0.75 | 0.3683 / 0.5957 | 0.3746 / 0.6136 | 0.4612 / 0.7036 | Higher split10 mIoU, weaker balance |
| ScanNet aliases | 0.3592 / 0.6191 | 0.3561 / 0.6192 | 0.4234 / 0.7002 | Mixed; not promoted |
| aliases + scene-mean alpha=0.5 | 0.3617 / 0.6180 | 0.3554 / 0.6174 | 0.4295 / 0.7026 | Mixed; not promoted |
| scene-mean calibration, alpha=1.0 | 0.3528 / 0.5834 | 0.3541 / 0.5935 | 0.4386 / 0.7048 | Hurts 19/15 and mAcc |

The DINOv3 cross-view branch is now also summarized on VALA/OpenGaFF-8:
0.3704/0.6159, 0.3718/0.6268, and 0.4390/0.7020 for the 19/15/10 splits.
It improves the conservative Gaussian-index row. With contextual kNN readout and
single-template text prompts, it also replaces the previous v67 contextual row
as the strongest balanced direct-field support evidence.

As of 2026-05-24, the training code also supports a stricter direct-field
variant that jointly constrains rendered 2D compact features and direct
3D primitive/point compact features, with visibility-weighted pair contrast from
registration view counts and cached-visible direct-point sampling. Generated
VALA/OpenGaFF-8 configs live under
`radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_{scene}.yaml`.
The full eight-scene v70 run reaches 0.3571/0.5997, 0.3644/0.6178, and
0.4419/0.7091 for the 19/15/10 splits. It improves split10 mAcc but hurts
split19/15 mIoU, so it is kept as an ablation rather than promoted.

Earlier targeted diagnostics:

| Scene | Branch | split19 | split15 | split10 |
|---|---|---:|---:|---:|
| scene0070_00 | v67 baseline | 0.2297 | 0.2405 | 0.3238 |
| scene0070_00 | DINO cv weight 0.001 | 0.2437 | 0.2466 | 0.3284 |
| scene0645_00 | v67 baseline | 0.2381 | 0.2458 | 0.2875 |
| scene0645_00 | DINO cv weight 0.003 | 0.2427 | 0.2500 | 0.2833 |

### Profile Evidence

Use `output/radio_gs/reports/submission_freeze_profile_summary.md`.

| Workload | Wall Time | Peak VRAM |
|---|---:|---:|
| LERF Figurines overlay | 26.198 s | 1568 MiB |
| LERF Ramen overlay | 40.474 s | 1762 MiB |
| LERF Teatime overlay | 36.997 s | 1850 MiB |
| LERF Waldo Kitchen overlay | 21.101 s | 2076 MiB |
| ScanNet v67 10-scene eval | 150.903 s | 1666 MiB |

### Storage Footprint Evidence

Use `output/radio_gs/reports/storage_footprint_report.md`.

| Scene | Direct 1280-D fp16 | Compact ckpt | Saving | Optional VPR cache | Saving w/ cache |
|---|---:|---:|---:|---:|---:|
| Figurines | 412.1 MiB | 237.0 MiB | 1.74x | 501.3 MiB | 0.56x |
| Ramen | 934.3 MiB | 311.2 MiB | 3.00x | 1131.4 MiB | 0.65x |
| Teatime | 1123.4 MiB | 338.1 MiB | 3.32x | 1360.4 MiB | 0.66x |
| Waldo Kitchen | 1688.8 MiB | 418.5 MiB | 4.04x | 2050.3 MiB | 0.68x |

The compactness claim applies to persistent scene/checkpoint storage. VPR is a
streamed inference-time readout; if the 1536-D registered SigLIP2 primitive
embeddings and per-query voxel scores are persisted as a cache, that optional
cache is larger than direct 1280-D feature storage and must be reported
separately.

## Active Configs

Do not treat every YAML under `radio_gs/configs/` as a current paper route. The
active LERF mainline is:

- `radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_ramen_fdh_ws240_240ep_seed7.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml`
- `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml`

The active ScanNet protocol is the generated v67 fair direct point-query config
family under the ignored path:

- `radio_gs/configs/generated/scannet_og/scannet_og_hybrid_v67fair_teacherbalanced_gidx_labelpoint_dp080_pce10_tdist05_s32768_b4_long20_fromv63_{scene}.yaml`

The active next direct-field optimization candidate is:

- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v70_cachedvisible_vc005_rc005_b2_s32768_ft20_{scene}.yaml`

Replica and room0 configs are supporting evidence only, not the main benchmark
route.

## Historical Branches

| Branch | What It Validated | Current Status |
|---|---|---|
| noFGC/noFDH seed runs | Robustness and ablation companion for geometry-head consistency | Supporting table only |
| pure frozen depth-only | Whether frozen depth supervision alone carries useful geometry | Archived support branch |
| room0 Replica depth/segmentation | Downstream utility beyond grounding | Supporting qualitative/auxiliary evidence |
| FGC/FDH weight sweeps | Sensitivity of geometry regularization | Archived design evidence |
| Figurines 2x | Small-object feature-resolution hypothesis | Archived unless re-run under frozen protocol |
| ScanNet v43/v62 diagnostics | Early point-query and label-informed debugging | Archived; not fair main table |
| ScanNet v67 | Current fair cross-domain protocol | Active mainline support |
| prompt ensemble / overlay sweeps | Visualization and prompt-sensitivity checks | Supporting only |
| LERF component ablations | Validated FGC/FDH, VFA/refiner, HGCF/hybrid, and CTR/HCD under controlled seed-7 LERF-OVS | Active paper ablation; see `output/radio_gs/reports/lerf_component_ablation.md` |
| RADIO DINOv3/SAM3 adaptor ablations | Tested adaptor-space consistency, DINO relation, and SAM3 soft-region supervision | Active ablation; only Ramen/Teatime promoted |
| ProFuse-style DINO cross-view branch | Tested cross-view DINO affinity plus text-heatmap peak protection | LERF diagnostic only; improves some overlap scores but does not preserve LocAcc |
| ScanNet DINO cross-view branch | Tested DINO affinity as cross-view context for direct point queries | Completed 10-scene conservative-weight ablation; positive but modest supporting evidence |
| LERF direct 3D selection | Upgraded from Gaussian-center text scores to rendered-feature-to-primitive registration with softmax-scene scoring | Active paper result; near OpenGaussian macro mIoU, above official Acc@0.25 macro, but Waldo remains weak |

## Archive Policy

Archived files are retained, not deleted. They can explain decisions, but they
are not paper-number sources. A result becomes paper-eligible only when it is
listed in `submission_freeze_report.md` or explicitly linked from the current
LaTeX draft.

The main archive locations are:

- `docs/archive/2026-05-02_legacy/`
- `output/radio_gs/reports/archive_legacy_20260502/`
