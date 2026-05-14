# RADIO-GS Current Paper Draft

Status: 2026-05-14 submission draft plus calibrated rendered grounding,
adaptor/cross-view diagnostics, and a registered OpenGaussian-style direct-selection result. Frozen mainline numbers must
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
frozen RADIO-GS package reaches 0.8712 macro localization accuracy and 0.5243
calibrated macro mIoU across four scenes. On a 10-scene ScanNet direct
point-query protocol, RADIO-GS reaches 0.3538, 0.3573, and 0.4293 mIoU on the
19-, 15-, and 10-class splits, while the contextual kNN readout raises them to
0.3637, 0.3708, and 0.4512. A VPR-registered LERF direct 3D object-selection
readout reaches 0.4227 macro mIoU / 0.6906 Acc@0.25 under the primitive-score
protocol and 0.4554 / 0.7014 after optional GT-free RGB boundary snapping of the
rendered evaluation mask. These results support the
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

The frozen main LERF-OVS result uses the paper-facing threshold sweep in
`output/radio_gs/reports/lerf_rendered_grounding_paper_ckpt_threshold_sweep.json`
with the GT-free threshold-0.60 readout.

| Scene | LocAcc | mIoU | Temperature |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.4244 | 50 |
| Ramen | 0.9014 | 0.6201 | 40 |
| Teatime | 0.8983 | 0.5760 | 25 |
| Waldo Kitchen | 0.8636 | 0.4769 | 25 |
| Macro | 0.8712 | 0.5243 | - |

The paper should present this as the primary internal RADIO-GS result. External
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

### LERF Direct 3D Object Selection

The direct-selection evaluator follows an OpenGaussian-style protocol: score
3D primitives with text, render selected primitives into binary masks, and
evaluate against LERF-OVS object masks. The current main result uses
rendered-view SigLIP2 features registered back to visible Gaussian primitives
with depth/alpha checks; masks are used only for final evaluation.

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

This result should not be mixed with rendered-view LERF mIoU. It is best used
to show VPR-backed primitive usability; the macro now slightly exceeds the
OpenGaussian official reference, but Waldo Kitchen remains the main
weakness and should be analyzed as object fragmentation / registration coverage.

GPU4/GPU5 diagnostics tested KNN point readout, semantic/geometry scoring heads,
scene-softmax calibration, LERF-style relevancy re-ranking, adaptor-promoted
checkpoints, voxel score aggregation, seed-expand component selection, and
GT-free mean+std score-distribution thresholds. Seed-expand components lower
Teatime/Waldo, but the 128-view meanstd2p5 selector with fixed 0.5% floor and
1.8% top-ratio cap improves the primitive-score VPR result to 0.4227 / 0.6906;
the optional GT-free RGB snap row improves the rendered evaluation mask to
0.4554 / 0.7014. A 1.5% cap gives an accuracy-oriented diagnostic at
0.4184 / 0.7013. The paper should still avoid implying raw Gaussian-center
superiority or global direct-3D SOTA.
The RGB-snap query audit shows that Waldo Kitchen remains the limiting scene
because of zero-prediction and primitive fragmentation: 0.2406 mIoU, 0.4091
Acc@0.25, and 0.2273 zero-prediction rate.

### ScanNet Direct Point Query

The current fair cross-domain table uses the v67 teacher-balanced direct
point-query protocol. The 10-scene macro results are:

| Split | mIoU | mAcc |
|---|---:|---:|
| 19 classes | 0.3538 | 0.6076 |
| 15 classes | 0.3573 | 0.6203 |
| 10 classes | 0.4293 | 0.7051 |

The stronger balanced support row uses contextual kNN point readout
(`k=8`, `candidate_k=32`) plus label-free scene-mean calibration at alpha 0.5:
0.3637/0.6033, 0.3708/0.6224, and 0.4512/0.7079 on the 19/15/10-class splits.
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
now exceeds the OpenGaussian official macro reference, but it remains below
newer published-context rows and Waldo Kitchen is still the weakest scene.
Finally, the current main LERF comparison still needs exact external baseline
provenance before the paper can make strong SOTA-style claims.

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
