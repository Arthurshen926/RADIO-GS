# RADIO-GS Current Paper Draft

Status: 2026-05-02 submission-freeze draft. Numbers in this file must be kept
consistent with `output/radio_gs/reports/submission_freeze_report.md`.

## Working Title

Foundation Feature Reconstruction in 3D Gaussian Scenes for Open-Vocabulary
Scene Understanding

## Abstract

We study whether dense vision-foundation features can be reconstructed as a
3D Gaussian scene representation and reused at novel views for open-vocabulary
scene understanding. Instead of training a scene-specific classifier or storing
raw high-dimensional features directly on every Gaussian, RADIO-GS distills
frozen RADIO features into a hybrid Gaussian feature field with a compact HCD
codec, screen-space refinement, and geometry-aware frozen-head supervision. The
resulting scene representation renders feature maps that remain compatible with
text grounding and other frozen downstream probes. On LERF-OVS, the current
frozen RADIO-GS package reaches 0.8712 macro localization accuracy and 0.4941
macro mIoU across four scenes. On a 10-scene ScanNet direct point-query protocol,
RADIO-GS reaches 0.3538, 0.3573, and 0.4293 mIoU on the 19-, 15-, and 10-class
splits, respectively. These results support the central claim that 3D Gaussian
scenes can serve as reusable foundation-feature memories, while the remaining
submission risk is mainly baseline provenance and final paper formatting rather
than core method viability.

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
   ScanNet direct point-query transfer, and profiled runtime/memory evidence.

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
lookup and label-point positions.

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
with reproduced external baselines. Finally, the current main LERF comparison
still needs exact external baseline provenance before the paper can make strong
SOTA-style claims.

## Submission Gaps

1. Close external baseline provenance for the main LERF table.
2. Convert profile and training logs into one polished efficiency table.
3. Freeze the fixed-protocol LERF seed-robustness table beside the best-scene
   main table.
4. Assemble the final qualitative figure from
   `output/radio_gs/reports/submission_freeze_figure_shortlist.md`.
5. Convert this draft into the target venue template and add related work.
