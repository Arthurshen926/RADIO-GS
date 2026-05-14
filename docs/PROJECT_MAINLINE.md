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
| LERF failure analysis | `output/radio_gs/reports/lerf_failure_analysis.md` |
| LERF direct 3D selection | `output/radio_gs/reports/lerf_direct_3d_selection.md` |
| LERF direct 3D debug audit | `output/radio_gs/reports/lerf_direct_3d_debug_audit.md` |

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
rendered SigLIP2-aligned features are registered back to visible 3D Gaussian
primitives with depth/alpha checks, text similarity is computed on those
registered primitives, selected primitives are rendered into binary masks, and
evaluation uses LERF-OVS object masks only at the final metric stage.

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed meanstd2p5 + floor0.005 + cap0.018 mIoU | 0.4829 | 0.4665 | 0.5043 | 0.2373 | 0.4227 |
| CTF-GS | SigLIP2 | + RGB snap, silhouette 0.60 mIoU | 0.5484 | 0.4706 | 0.5621 | 0.2406 | 0.4554 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 mIoU | 0.4055 | 0.4491 | 0.4862 | 0.1991 | 0.3850 |
| CTF-GS | SigLIP2 | VPR + voxel context cap0.015 diagnostic mIoU | 0.4829 | 0.4615 | 0.4965 | 0.2328 | 0.4184 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed meanstd2p5 + floor0.005 + cap0.018 Acc@0.25 | 0.8214 | 0.7183 | 0.8136 | 0.4091 | 0.6906 |
| CTF-GS | SigLIP2 | + RGB snap, silhouette 0.60 Acc@0.25 | 0.7857 | 0.7465 | 0.8644 | 0.4091 | 0.7014 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 Acc@0.25 | 0.6786 | 0.7324 | 0.7966 | 0.3636 | 0.6428 |
| CTF-GS | SigLIP2 | VPR + voxel context cap0.015 diagnostic Acc@0.25 | 0.8214 | 0.7042 | 0.7797 | 0.5000 | 0.7013 |

Published-context rows from newer primitive-/instance-aware methods are kept in
a separate context table rather than the strict local table: Dr. Splat
43.29/64.30, CAGS 50.79/69.62, InstanceGaussian 45.30/58.44, and OpenGaFF
54.36/80.84 mIoU/Acc@0.25. These numbers show that CTF-GS + VPR is competitive
with the OpenGaussian anchor but should not be claimed as global direct-3D SOTA.

GPU4/GPU5 follow-up diagnostics show that rendered-feature registration, not
score thresholding, is the main improvement. GT-free voxel-max context
aggregation further reduces primitive fragmentation. The earlier Gaussian-center
direct readout was 0.0804 macro mIoU / 0.0932 macro Acc@0.25 under its top10%
selector; under the same top2% selector used by the VPR rows it is 0.012 /
0.009. Registered softmax24 reached 0.3421 / 0.5547, registered+voxel with the
fixed top2% selector reached 0.3850 / 0.6428. With the same GT-free voxel-max
context and a larger 128-view all-pose VPR budget, the mean+2.5std
score-distribution selector with fixed 0.5% floor and 1.8% cap is now the
strongest primitive-score protocol at 0.4227 / 0.6906; applying the optional
GT-free RGB snap only to the rendered evaluation mask improves the paper-facing
refined row to 0.4554 / 0.7014. A 1.5% cap is a useful accuracy-oriented
diagnostic at 0.4184 / 0.7013. This exceeds the OpenGaussian official macro reference, but
Waldo Kitchen remains below OpenGaussian and the table is still official-source
context rather than a locally rerun same-evaluator SOTA comparison.

The Dr. Splat-inspired contribution-weighting path has been implemented in the
evaluator as `--registration_weight_mode {alpha,alpha_depth}`. Under the same
96-view VPR + voxel-max + fixed top-2% protocol, alpha weighting drops macro
mIoU/Acc@0.25 to 0.2978/0.5389 and alpha-depth weighting drops to
0.2967/0.5345, compared with the uniform top2% VPR baseline at 0.3850/0.6428
and the fixed 128-view meanstd2p5 + floor0.005 + cap0.018 VPR selector at 0.4227/0.6906. This
negative result means the paper should keep uniform VPR as the main primitive
readout and describe contribution-weighted registration as future work that
needs true rasterization-contribution assignment rather than center-sampled
alpha weighting.

Additional component/voxel diagnostics from the Expert (5) pass are also
negative for the main Waldo bottleneck. Top-score component filtering with one
or two retained components reduces Waldo mIoU to 0.1757/0.1827, `voxel_mean`
reduces it to 0.0931 under the earlier paper selector, and `voxel_max_dilate`
reduces it to 0.1286. The current 128-view VPR + voxel-max + meanstd2p5
floor/cap protocol remains the strongest fixed, GT-free direct-3D readout.
The RGB-snap query audit in
`output/radio_gs/reports/lerf_direct_3d_query_audit_rgb_snap_sil0p60.md`
confirms that Waldo Kitchen remains the failure scene: 0.2406 mIoU, 0.4091
Acc@0.25, 0.2273 zero-prediction rate, and a 95% bootstrap mIoU interval of
[0.1459, 0.3440].

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
`output/lerf_sam_dino_tasks/formal_v6_dino_topk_area200_peak/lerf_sam_dino_task_report.md`.

| Task | Teacher LocAcc/Hit | Teacher mIoU/Score | Rendered LocAcc/Hit | Rendered mIoU/Score |
|---|---:|---:|---:|---:|
| SAM3 point prompt | 1.0000 | 0.3700 | 1.0000 | 0.4169 |
| SAM3 box prompt | 0.8702 | 0.6560 | 0.8221 | 0.6638 |
| SAM3 mask propagation | 0.7872 | 0.3583 | 0.6667 | 0.3756 |
| DINOv3 dense matching | 0.5723 | 0.8547 | 0.5393 | 0.9048 |
| DINOv3 mask propagation + robust readout | 0.7801 | 0.4806 | 0.7730 | 0.4456 |

Rendered CTF-GS features now exceed the frame-wise teacher on SAM3-adaptor
mask mIoU for all three prompt modes. The robust DINO readout combines
source-background contrast, foreground top-k pooling, 2.0x area scaling, and
peak-component cleanup; it raises rendered DINO mask-propagation mIoU from the
formal_v4 value of 0.3684 to 0.4456 and LocAcc from 0.7376 to 0.7730. This
exceeds the previous fixed-readout teacher mIoU reference of 0.3921, but the
same robust readout also improves teacher to 0.4806/0.7801. The paper claim
should therefore be "improves SigLIP2 grounding and SAM3-adaptor region mIoU,
substantially narrows the DINO propagation gap while preserving high DINO
similarity."

The mutual matching + homography RANSAC diagnostic is recorded at
`output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_report.md`.
It improves the visual reliability of DINO matches by reducing outliers and
raising rendered dense-match similarity to 0.9277, but the rendered DINO mask
propagation row remains below the teacher under the same readout
(0.4456 vs. 0.4806 mIoU). Use it as qualitative/diagnostic evidence rather than
as a main superiority claim.

### ScanNet v67 Direct Point Query

Use `output/scannet_pointcloud_eval/*_v67_teacherbalanced_fromv63_best_gidx_labelpoint/`.

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

The stronger direct point-readout support row uses the same v67 checkpoints but
queries each ScanNet vertex through local Gaussian context:
`query_mode=knn`, `k=8`, `candidate_k=32`, `logit_calibration=scene_mean`,
`alpha=0.5`.

| Split | contextual kNN mIoU | contextual kNN mAcc |
|---|---:|---:|
| 19 classes | 0.3637 | 0.6033 |
| 15 classes | 0.3708 | 0.6224 |
| 10 classes | 0.4512 | 0.7079 |

This is the current strongest balanced ScanNet evidence. A more aggressive
`alpha=0.75` setting raises split10 mIoU to 0.4534 but lowers split19/15 mIoU
and mAcc, so it stays diagnostic.

Label-free ScanNet prompt/calibration ablations:

| Variant | split19 | split15 | split10 | Use |
|---|---:|---:|---:|---|
| v67 baseline | 0.3538 / 0.6076 | 0.3573 / 0.6203 | 0.4293 / 0.7051 | Main conservative row |
| scene-mean calibration, alpha=0.5 | 0.3575 / 0.6101 | 0.3604 / 0.6227 | 0.4353 / 0.7074 | Positive supporting ablation |
| kNN contextual readout + scene-mean alpha=0.5 | 0.3637 / 0.6033 | 0.3708 / 0.6224 | 0.4512 / 0.7079 | Promoted balanced support row |
| kNN contextual readout + scene-mean alpha=0.75 | 0.3620 / 0.5994 | 0.3692 / 0.6187 | 0.4534 / 0.7078 | Higher split10 mIoU, weaker balance |
| ScanNet aliases | 0.3592 / 0.6191 | 0.3561 / 0.6192 | 0.4234 / 0.7002 | Mixed; not promoted |
| aliases + scene-mean alpha=0.5 | 0.3617 / 0.6180 | 0.3554 / 0.6174 | 0.4295 / 0.7026 | Mixed; not promoted |
| scene-mean calibration, alpha=1.0 | 0.3528 / 0.5834 | 0.3541 / 0.5935 | 0.4386 / 0.7048 | Hurts 19/15 and mAcc |

Targeted DINOv3 cross-view diagnostics are positive but not yet a 10-scene
replacement:

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
