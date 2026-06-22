# RADIO-GS Current Paper Draft

Status: 2026-05-14 submission draft plus calibrated rendered grounding,
adaptor/cross-view diagnostics, and a registered OpenGaussian-style direct-selection result. Frozen mainline numbers must
stay consistent with `output/radio_gs/reports/submission_freeze_report.md`;
promoted diagnostic candidates are tracked in `docs/PROJECT_MAINLINE.md`.

The active LaTeX draft is `paper/radio_gs_draft.tex`.

## Working Title

GaussFM: Compact Foundation-Feature Gaussian Memory for
Open-Vocabulary 3D Gaussian Scene Understanding

## Abstract

We study whether dense vision-foundation features can be reconstructed as a
3D Gaussian scene representation and reused at novel views for open-vocabulary
scene understanding. Instead of training a scene-specific classifier or storing
raw high-dimensional features directly on every Gaussian, RADIO-GS distills
frozen RADIO features into a hybrid Gaussian feature field with a compact HCD
codec, screen-space refinement, geometry-aware frozen-head supervision,
explicit feature-quality/visibility readouts, and a View-to-Primitive
Registration (VPR) bridge for primitive-level querying. The
resulting scene representation renders feature maps that remain compatible with
text grounding and other frozen downstream probes. On LERF-OVS, the current
frozen RADIO-GS package with the GT-free peak-connected-component mask readout
reaches 0.8598 macro localization accuracy and 0.5707 macro mIoU across four
scenes. On the VALA-aligned eight-scene ScanNet direct point-query protocol,
the paper-facing DINO-CV contextual kNN readout with scene-mean calibration and
spatial logit propagation reaches 0.3806, 0.3871, and 0.4711 mIoU on the 19-,
15-, and 10-class splits. A follow-up proposal-memory readout improves the
19-class detail split to 0.3931 mIoU / 0.6255 mAcc, but it is kept as an
ablation because the 15/10-class splits decrease. A VPR-registered LERF direct 3D object-selection
readout with a fixed global softmax-score threshold, 0.5% floor, 1.8% cap, and
GT-free RGB boundary snap reaches 0.4801 macro mIoU / 0.6760 Acc@0.25. With
frozen official SAM3 box-prompt boundary readout, the compact direct field
reaches 0.5705 / 0.6835 under a fixed global threshold, while the scene-locked
diagnostic upper bound reaches 0.5972 / 0.7009. These results support the
central claim that 3D Gaussian scenes can serve as reusable foundation-feature
memories across rendered-view and registered primitive-level interfaces, while
the remaining submission risk is strongest around Waldo Kitchen, exact baseline
provenance, and final paper formatting.

## Main Contributions

1. We formulate foundation-feature reconstruction for 3D Gaussian scenes: the
   target is not only photorealistic RGB rendering, but novel-view feature maps
   that remain usable by frozen open-vocabulary and task heads.
2. We introduce a unified multi-head hybrid RADIO-GS representation that
   combines per-Gaussian latent storage, a coarse spatial branch, an HCD
   bottleneck codec, explicit quality/visibility heads, and screen-space
   refinement to reconstruct 1280d RADIO features efficiently.
3. We use frozen-head supervision as a geometry-aware regularizer, allowing
   downstream head behavior to shape the feature field without turning the model
   into a task-specific classifier.
4. We provide a conservative frozen evaluation package covering LERF-OVS,
   VPR-backed LERF direct 3D object-selection diagnostics, ScanNet direct
   point-query transfer, and profiled runtime/memory evidence.

## Method Narrative

Given a pretrained 3DGS scene, RADIO-GS keeps the Gaussian geometry fixed and
learns a feature field attached to that geometry. Training views are first passed
through frozen RADIO C-RADIOv4-H to produce 1280d dense target
features. The feature field renders compact latent maps from Gaussian splats and
fuses them with a coarse spatial branch queried from 3D positions. An HCD decoder
maps the compact representation back into the RADIO feature space, after which a
screen-space refiner improves pixel-aligned feature quality using rendering
guidance.

The training objective combines feature reconstruction with auxiliary
task-head consistency. In the FDH warm-start setting, a no-FDH feature field is
first trained to convergence, then refined with a frozen depth-head loss. This
schedule keeps early optimization focused on feature reconstruction and applies
the frozen head as a late geometry-aware regularizer.

At test time, the model renders a novel-view feature map. For LERF-OVS grounding,
RADIO-GS compares the rendered feature map with SigLIP2 text embeddings and
localizes each query from the relevancy heatmap. For ScanNet, it evaluates
direct point queries on the VALA-aligned ScanNet-8 split with the promoted DINO-CV
contextual kNN readout. For direct LERF object selection, VPR scores
registered-view primitive embeddings against text before the selected Gaussians
are rendered only for mask evaluation. The raw Gaussian-center readout remains a
diagnostic lower bound, not the paper-facing direct-selection method.

## Experiments Draft

### LERF-OVS

The frozen main LERF-OVS result uses the GT-free peak-connected-component
readout in `paper/artifacts/lerf_rendered_grounding_peak_component_20260524.json`.
It starts from the threshold-0.60 heatmap mask and keeps only the connected
component containing the query heatmap maximum.

| Scene | LocAcc | mIoU | Temperature |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.5134 | 50 |
| Ramen | 0.9014 | 0.6249 | 40 |
| Teatime | 0.8983 | 0.6177 | 25 |
| Waldo Kitchen | 0.8182 | 0.5268 | 25 |
| Macro | 0.8598 | 0.5707 | - |

The paper should present this as the primary internal RADIO-GS mask-quality result. External
LERF/LangSplat/LEGaussians rows should remain clearly marked as published or
reproduced only after their exact source and protocol alignment are closed.
An adaptive mean+std boundary threshold was tested as a no-GT alternative, but
its macro mIoU is 0.4939 at unchanged 0.8712 LocAcc, so it should remain a
negative diagnostic and not replace the threshold-0.60 main row.

Promoted adaptor/cross-view candidate:

| Scene | Selected branch | LocAcc | mIoU | Temperature |
|---|---|---:|---:|---:|
| Figurines | DINO cv + spatial text heatmap | 0.8214 | 0.4343 | 50 |
| Ramen | DINO relation + SAM3 region | 0.9014 | 0.5873 | 40 |
| Teatime | DINO relation + SAM3 region | 0.8983 | 0.5592 | 28 |
| Waldo Kitchen | Baseline | 0.8636 | 0.4106 | 25 |
| Macro | - | 0.8712 | 0.4979 | - |

### 2D Frame-wise RADIO-vs-GaussFM Feature Usability

Use `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md` and the compact
table snippet `paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex`. The
consolidated 2D evidence shows that rendered GaussFM features improve the main
LERF text-grounding mIoU over frame-wise RADIO features (0.5707 vs.
0.4634) and win 6/6 primary selected frozen-head metrics. The claim should stay
qualified to primary feature-usability metrics: the DINOv3 mask-propagation row
now favors GaussFM with the multi-head DINO-support/SAM-boundary readout
(0.4677 vs. 0.4606 mIoU), while secondary SAM LocAcc and DINO dense HitRate
caveats remain.

### LERF Direct 3D Object Selection

The direct-selection evaluator follows an OpenGaussian-style protocol: score
3D primitives with text, render selected primitives into binary masks, and
evaluate against LERF-OVS object masks. We now keep the VPR/RGB-snap primitive
row for strict primitive-mask comparison and a compact direct-field SAM3-box
boundary readout with a fixed global `thr0p25` selector. The SAM3 candidate mask
is chosen by overlap with the rendered prediction, not by ground truth. The
scene-locked and legacy pad0 SAM3-box rows remain diagnostics because their
thresholds are post-hoc or scene-specific.

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| GaussFM | SigLIP2 | VPR + fixed threshold 0.25 + RGB snap mIoU | 0.5309 | 0.5805 | 0.5662 | 0.2429 | 0.4801 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout mIoU, pad16 fixed `thr0p25` | 0.6136 | 0.6409 | 0.6130 | 0.4142 | 0.5705 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout mIoU, pad16 scene-locked diagnostic | 0.6422 | 0.6494 | 0.6528 | 0.4444 | 0.5972 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout mIoU, pad0 legacy diagnostic | 0.5924 | 0.6830 | 0.6556 | 0.3949 | 0.5815 |
| GaussFM | SigLIP2 | VPR + voxel context fixed top0p02 mIoU | 0.4055 | 0.4491 | 0.4862 | 0.1991 | 0.3850 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| GaussFM | SigLIP2 | VPR + fixed threshold 0.25 + RGB snap Acc@0.25 | 0.7857 | 0.7465 | 0.7627 | 0.4091 | 0.6760 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout Acc@0.25, pad16 fixed `thr0p25` | 0.6964 | 0.7465 | 0.7458 | 0.5455 | 0.6835 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout Acc@0.25, pad16 scene-locked diagnostic | 0.7321 | 0.7465 | 0.7797 | 0.5455 | 0.7009 |
| GaussFM | SigLIP2+SAM3 | compact direct field + official SAM3 box readout Acc@0.25, pad0 legacy diagnostic | 0.7321 | 0.8028 | 0.7797 | 0.5455 | 0.7150 |
| GaussFM | SigLIP2 | VPR + voxel context fixed top0p02 Acc@0.25 | 0.6786 | 0.7324 | 0.7966 | 0.3636 | 0.6428 |

This result should not be mixed with rendered-view LERF mIoU. It is best used
to show dual readout usability: direct primitive scores give the 3D selection,
while official SAM3 supplies a frozen, promptable boundary readout. The strict
fixed-global pad16 row improves the previous direct-field diagnostic from
0.4363/0.6191 to 0.5705/0.6835 mIoU/Acc@0.25 and exceeds the VPR/RGB-snap row.
The scene-locked pad16 diagnostic reaches 0.5972/0.7009 with boundary-F 0.6817
and trimap IoU 0.4043, but it should stay appendix-only unless selected by a
held-out validation protocol. The legacy pad0 best-by-scene export reaches
0.5815/0.7150 and remains a diagnostic for the same reason.

GPU4/GPU5 diagnostics tested KNN point readout, semantic/geometry scoring heads,
scene-softmax calibration, LERF-style relevancy re-ranking, adaptor-promoted
checkpoints, voxel score aggregation, seed-expand component selection, and
GT-free mean+std score-distribution thresholds. Seed-expand components lower
Teatime/Waldo, while the latest 128-view threshold-0.25 selector with fixed
0.5% floor, 1.8% cap, and GT-free RGB snap improves the primitive-score VPR
result to 0.4801 / 0.6760. The paper should still avoid implying raw Gaussian-center
superiority or global direct-3D SOTA.
The direct compact field also has a confidence-weighted VPR-to-field transfer
result: normalized `log1p(view_counts)` sample weights improve the
threshold-sweep direct-field diagnostic from 0.4119 / 0.5876 to 0.4363 /
0.6191. With official SAM3 box-prompt readout refinement, the same direct-field
branch becomes the strongest strict fixed-threshold LERF direct-3D boundary
readout, while post-hoc threshold choices remain diagnostic.
The RGB-snap query audit shows that Waldo Kitchen remains the limiting scene
because of zero-prediction and primitive fragmentation: 0.2429 mIoU, 0.4091
Acc@0.25, and 0.1818 zero-prediction rate.
The confidence/coverage mechanism table adds a GT-free explanation: scene-level
mean valid VPR views correlate with strict Direct3D mIoU at Pearson r=0.7588,
and the high primitive-score bucket reaches 0.6358 mean IoU / 0.8551 Acc@0.25
versus 0.4345 / 0.6143 for the low bucket. Text-margin stratification gives a
matching ambiguity trend: distinct-margin queries reach 0.6202 / 0.8261 while
ambiguous queries are 0.4329 / 0.6286.
The boundary-error readout adds the measured boundary side of this failure
analysis for the strict pad16 SAM3-box result. Across 208 query instances,
IoU correlates with boundary-F at Pearson r=0.9148 and with trimap IoU at
r=0.8412; balanced-size predictions have mean boundary error 0.0637, while
under-selection and over-selection are 0.7042 and 0.6919. The report does not
claim a causal alpha/depth discontinuity mechanism. The geometry-map rerun now
adds 208/208 per-query alpha/depth overlays; boundary error has weak positive
correlation with alpha-edge, depth-edge, and combined discontinuity statistics
(Pearson r=0.1515, 0.1178, and 0.1438), so this is mechanism context rather
than causal proof.

### ScanNet Direct Point Query

The current fair cross-domain table uses the promoted DINO-CV contextual kNN
point-query readout on the VALA-aligned eight-scene ScanNet subset:

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3806 | 0.6129 |
| 15 classes | 0.3871 | 0.6315 |
| 10 classes | 0.4711 | 0.7200 |

The row uses the DINO-CV compact field with contextual kNN point readout
(`k=16`, `candidate_k=80`) plus label-free scene-mean calibration at alpha 0.45
and one-step spatial logit propagation (`k=12`, alpha 1.0).
This should be framed as VALA-aligned split-aligned cross-domain feature
usability evidence, not as a fully standard ScanNet semantic segmentation
leaderboard comparison. Older v67 and label-informed ScanNet diagnostics must
stay out of the main fair table.

### Efficiency Evidence

The current freeze package includes five formal profile workloads:

| Workload | Wall Time | Peak VRAM |
|---|---:|---:|
| LERF Figurines overlay | 26.198 s | 1568 MiB |
| LERF Ramen overlay | 40.474 s | 1762 MiB |
| LERF Teatime overlay | 36.997 s | 1850 MiB |
| LERF Waldo Kitchen overlay | 21.101 s | 2076 MiB |
| ScanNet legacy 10-scene eval profile | 150.903 s | 1666 MiB |

The paper should use these as evaluation-profile evidence. Training-cost claims
need a separate table because training logs and evaluation profiles are different
measurement types.
The compression/downstream audit adds a mechanism guardrail: compact checkpoint
saving has only weak positive correlation with rendered mIoU (r=0.3606) and
negative correlation with Direct3D mIoU (r=-0.6158), so compactness and
direct-query robustness should be argued as separate properties.
The feature-error/text-relevance audit supports the reconstruction thesis:
`1 - best validation cos_decoded` correlates with rendered mIoU error at
r=0.9568 and with LocAcc error at r=0.8713 across the four frozen LERF scenes.
The nearest-view cache baseline is now measured under the same LERF evaluator:
an unwarped nearest cached RADIO frame reaches only 0.2722 macro LocAcc /
0.1545 macro mIoU, versus 0.8598 / 0.5707 for rendered GaussFM features. This
should be used as a cache-only control, not a 3D scene-memory baseline.
The full per-Gaussian 1280-D explicit RADIO-memory baseline is also now
measured under the same evaluator: it registers cached frame-wise RADIO features to
Gaussian centers and reaches 0.5642 macro LocAcc / 0.3182 macro mIoU, with
0.2020 mean registered-Gaussian fraction and 1039.7 MiB mean fp16 feature
storage. This closes the raw-feature controlled row; the compact GaussFM row
remains substantially stronger and should be framed as the main 3D feature
field result.
The boundary-error audit supports the SAM3-box readout framing: boundary-F and
trimap IoU move with query IoU, and
`output/radio_gs/reports/alpha_depth_boundary_alignment_report.md` now records
208/208 alpha/depth geometry-map records for the strict pad16 row. The high
discontinuity bucket has lower mean IoU (0.5394) and higher mean boundary error
(0.3699) than the low bucket (0.6361 / 0.2473), but the query-level
correlations are weak, so the paper should present this as boundary mechanism
diagnostics rather than causal occlusion proof.

## Limitations Draft

RADIO-GS currently depends on a pretrained Gaussian geometry backbone and does
not claim to improve RGB reconstruction. Its open-vocabulary quality is limited
by the resolution and alignment of the frozen RADIO features, which is most
visible on small-object scenes such as Figurines. The ScanNet protocol used here
is the VALA-aligned eight-scene direct point-query transfer test for the
learned feature field, using every-20-frame ScanNet training splits and mIoU/mAcc
evaluation on GT semantic point clouds. It should still be framed as
VALA-aligned protocol comparison rather than a full ScanNet leaderboard. The
LERF direct 3D object-selection result should be compared only against the
published public baseline rows we keep in the table; the unpublished
protocol-source method row is excluded from the comparison. External comparison
numbers should be cited from the original published tables, while our own rows stay fully auditable under the
local evaluator.
The training entry point is now explicitly audited in
`output/radio_gs/reports/train_feature_field_audit.md`: manifest, split,
checkpoint, metrics-history, and lock guards are present. Feature/text/cache
tensor loads now go through `load_training_tensor_cache`, the support code is
split under `radio_gs/training/`, and the audit passes with a 3735-line entry
script.

## Submission Gaps

1. Move the draft into the target venue template and tighten related work.
2. Keep the final qualitative figures synchronized with the current main rows:
   rendered grounding, direct-field + SAM3 box readout, and SAM/DINO probes.
3. Decide whether to include the measured per-Gaussian 1280-D explicit baseline
   in the main paper or keep it as appendix evidence.
4. Decide whether the alpha/depth boundary-case montage belongs in the main
   failure-analysis section or appendix.
5. Split the training entry point enough for release: move data/loss helpers
   into importable modules.
6. Rerun LERF/LangSplat/LEGaussians under the local evaluator only if the paper
   wants to make strict same-evaluator SOTA claims.
