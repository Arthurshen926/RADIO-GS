# RADIO-GS Current Paper Draft

Status: 2026-05-12 submission draft plus adaptor/cross-view diagnostics and a
registered OpenGaussian-style direct-selection result. Frozen mainline numbers must
stay consistent with `output/radio_gs/reports/submission_freeze_report.md`;
promoted diagnostic candidates are tracked in `docs/PROJECT_MAINLINE.md`.

The active LaTeX draft is `paper/radio_gs_draft.tex`.

## Working Title

CTF-GS: Compact Teacher Feature Fields with View-to-Primitive Registration for
Open-Vocabulary 3D Gaussian Scene Understanding

## Abstract

We study whether dense vision-foundation features can be reconstructed as a
3D Gaussian scene representation and reused at novel views for open-vocabulary
scene understanding. Instead of training a scene-specific classifier or storing
raw high-dimensional features directly on every Gaussian, RADIO-GS distills
frozen RADIO features into a hybrid Gaussian feature field with a compact HCD
codec, screen-space refinement, geometry-aware frozen-head supervision, and a
View-to-Primitive Registration (VPR) bridge for primitive-level querying. The
resulting scene representation renders feature maps that remain compatible with
text grounding and other frozen downstream probes. On LERF-OVS, the current
frozen RADIO-GS package reaches 0.8712 macro localization accuracy and 0.4941
macro mIoU across four scenes; a promoted adaptor/cross-view candidate keeps the
same localization accuracy and raises macro mIoU to 0.4979. On a 10-scene
ScanNet direct point-query protocol, RADIO-GS reaches 0.3538, 0.3573, and
0.4293 mIoU on the 19-, 15-, and 10-class splits, respectively. A VPR-registered
LERF direct 3D object-selection readout raises fixed-protocol macro mIoU from
0.0804 to 0.3850 and Acc@0.25 from 0.0932 to 0.6428 under an OpenGaussian-style
protocol, after GT-free voxel context aggregation. These results support the
central claim that 3D Gaussian scenes can serve as reusable foundation-feature
memories across rendered-view and registered primitive-level interfaces, while
the remaining submission risk is strongest around Waldo Kitchen, exact baseline
provenance, and final paper formatting.

## Main Contributions

1. We formulate foundation-feature reconstruction for 3D Gaussian scenes: the
   target is not only photorealistic RGB rendering, but novel-view feature maps
   that remain usable by frozen open-vocabulary and task heads.
2. We introduce a hybrid RADIO-GS representation that combines per-Gaussian
   latent storage, a coarse spatial branch, an HCD bottleneck codec, and
   screen-space refinement to reconstruct 1280d RADIO features efficiently.
3. We use frozen-head supervision as a geometry-aware regularizer, allowing
   downstream head behavior to shape the feature field without turning the model
   into a task-specific classifier.
4. We provide a conservative frozen evaluation package covering LERF-OVS,
   VPR-backed LERF direct 3D object-selection diagnostics, ScanNet direct
   point-query transfer, and profiled runtime/memory evidence.

## Method Narrative

Given a pretrained 3DGS scene, RADIO-GS keeps the Gaussian geometry fixed and
learns a feature field attached to that geometry. Training views are first passed
through a frozen RADIO C-RADIOv4-H teacher to produce 1280d dense target
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
direct point queries under the v67 teacher-balanced protocol with Gaussian-index
lookup and label-point positions. For direct LERF object selection, VPR scores
registered-view primitive embeddings against text before the selected Gaussians
are rendered only for mask evaluation. The raw Gaussian-center readout remains a
diagnostic lower bound, not the paper-facing direct-selection method.

## Experiments Draft

### LERF-OVS

The frozen main LERF-OVS result uses the rendered-feature best-scene summary in
`output/radio_gs/lerf_summary_tables/current_best_lerf_ovs_per_scene.csv`.

| Scene | LocAcc | mIoU | Temperature |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.4308 | 50 |
| Ramen | 0.9014 | 0.5862 | 40 |
| Teatime | 0.8983 | 0.5486 | 25 |
| Waldo Kitchen | 0.8636 | 0.4106 | 25 |
| Macro | 0.8712 | 0.4941 | - |

The paper should present this as the primary internal RADIO-GS result. External
LERF/LangSplat/LEGaussians rows should remain clearly marked as published or
reproduced only after their exact source and protocol alignment are closed.

Promoted adaptor/cross-view candidate:

| Scene | Selected branch | LocAcc | mIoU | Temperature |
|---|---|---:|---:|---:|
| Figurines | DINO cv + spatial text heatmap | 0.8214 | 0.4343 | 50 |
| Ramen | DINO relation + SAM3 region | 0.9014 | 0.5873 | 40 |
| Teatime | DINO relation + SAM3 region | 0.8983 | 0.5592 | 28 |
| Waldo Kitchen | Baseline | 0.8636 | 0.4106 | 25 |
| Macro | - | 0.8712 | 0.4979 | - |

### LERF Direct 3D Object Selection

The direct-selection evaluator follows an OpenGaussian-style protocol: score
3D primitives with text, render selected primitives into binary masks, and
evaluate against LERF-OVS object masks. The current main result uses
rendered-view SigLIP2 features registered back to visible Gaussian primitives
with depth/alpha checks; masks are used only for final evaluation.

| Method | Text head | Protocol | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---|---|---:|---:|---:|---:|---:|
| OpenGaussian | CLIP | official paper mIoU | 0.3929 | 0.3101 | 0.6044 | 0.2270 | 0.3836 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed meanstd2p5 + floor0.005 + cap0.02 mIoU | 0.4879 | 0.4536 | 0.5111 | 0.2008 | 0.4133 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 mIoU | 0.4055 | 0.4491 | 0.4862 | 0.1991 | 0.3850 |
| CTF-GS | SigLIP2 | VPR + voxel context best-by-scene mIoU | 0.4879 | 0.4536 | 0.5111 | 0.2138 | 0.4166 |
| OpenGaussian | CLIP | official paper Acc@0.25 | 0.5536 | 0.4225 | 0.7627 | 0.3182 | 0.5143 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed meanstd2p5 + floor0.005 + cap0.02 Acc@0.25 | 0.8036 | 0.7324 | 0.7966 | 0.3636 | 0.6741 |
| CTF-GS | SigLIP2 | VPR + voxel context fixed top0p02 Acc@0.25 | 0.6786 | 0.7324 | 0.7966 | 0.3636 | 0.6428 |
| CTF-GS | SigLIP2 | VPR + voxel context best-by-scene Acc@0.25 | 0.8036 | 0.7324 | 0.7966 | 0.3636 | 0.6741 |

This result should not be mixed with rendered-view LERF mIoU. It is best used
to show VPR-backed primitive usability; the macro now slightly exceeds the
OpenGaussian official reference, but Waldo Kitchen remains the main
weakness and should be analyzed as object fragmentation / registration coverage.

GPU4/GPU5 diagnostics tested KNN point readout, semantic/geometry scoring heads,
scene-softmax calibration, LERF-style relevancy re-ranking, adaptor-promoted
checkpoints, voxel score aggregation, seed-expand component selection, and
GT-free mean+std score-distribution thresholds. Seed-expand components lower
Teatime/Waldo, but meanstd2p5 with fixed 0.5% floor and 2% top-ratio cap improves
the paper-facing VPR result to 0.4133 / 0.6741. The paper should still avoid implying raw Gaussian-center
superiority or global direct-3D SOTA.

### ScanNet Direct Point Query

The current fair cross-domain table uses the v67 teacher-balanced direct
point-query protocol. The 10-scene macro results are:

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

This should be framed as cross-domain feature usability evidence, not as a fully
standard ScanNet semantic segmentation leaderboard comparison. Older
label-informed ScanNet diagnostics must stay out of the main fair table.

### Efficiency Evidence

The current freeze package includes five formal profile workloads:

| Workload | Wall Time | Peak VRAM |
|---|---:|---:|
| LERF Figurines overlay | 26.198 s | 1568 MiB |
| LERF Ramen overlay | 40.474 s | 1762 MiB |
| LERF Teatime overlay | 36.997 s | 1850 MiB |
| LERF Waldo Kitchen overlay | 21.101 s | 2076 MiB |
| ScanNet v67 10-scene eval | 150.903 s | 1666 MiB |

The paper should use these as evaluation-profile evidence. Training-cost claims
need a separate table because training logs and evaluation profiles are different
measurement types.

## Limitations Draft

RADIO-GS currently depends on a pretrained Gaussian geometry backbone and does
not claim to improve RGB reconstruction. Its open-vocabulary quality is limited
by the resolution and alignment of the frozen RADIO features, which is most
visible on small-object scenes such as Figurines. The ScanNet protocol used here
is a fair direct point-query transfer test for the learned feature field, but it
is not yet a full replacement for a standardized semantic segmentation benchmark
with reproduced external baselines. The LERF direct 3D object-selection result
is now competitive in macro Acc@0.25 and close in macro mIoU, but Waldo Kitchen
is still below OpenGaussian-style primitive-selection numbers. Finally, the
current main LERF comparison still needs exact external baseline provenance
before the paper can make strong SOTA-style claims.

## Submission Gaps

1. Close external baseline provenance for the main LERF table.
2. Convert profile and training logs into one polished efficiency table.
3. Freeze the fixed-protocol LERF seed-robustness table beside the best-scene
   main table.
4. Assemble the final qualitative figure from
   `output/radio_gs/reports/submission_freeze_figure_shortlist.md`.
5. Add a concise Waldo Kitchen failure/coverage analysis for the registered
   direct-selection readout.
6. Convert this draft into the target venue template and add related work.
