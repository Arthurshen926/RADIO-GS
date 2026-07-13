# Query-Consistent Canonical RADIO Field: implementation and evidence

Date: 2026-07-13

This note records the current canonical-field implementation and the real
experiments that have completed. It does not promote partial scene results to
dataset-level benchmark claims. Target masks are used only by evaluators after
prediction; no target-set threshold calibration is used.

## Final method path

1. **Query-free multiview primitive reconstruction (MPR).** Official
   C-RADIOv4-H spatial features from permitted reconstruction views are lifted
   with depth/alpha-checked raster responsibility and averaged on fixed
   Gaussian rows. Query strings, benchmark masks, and held-out RGBs are absent.
2. **One compact canonical field.** Each Gaussian stores a 384-D coefficient.
   A pointwise affine PCA decoder reconstructs the 1280-D RADIO vector. The
   decoder has no screen normalization or token mixing, and commutes with an
   alpha-normalized splat.
3. **Query-free render-fidelity correction.** Local coefficients are optimized
   for 256 steps on reconstruction views, anchored by the MPR loss. Checkpoint
   selection uses four frozen non-benchmark development views and a maximum
   MPR-cosine drop of 0.001. The seven official ramen evaluation views remain
   unseen during selection.
4. **Frozen official capability views.** The canonical RADIO vector is mapped
   by the official C-RADIOv4 `dino_v3_7b` and `sam3` feature projections. These
   modules are loaded from the official RADIO checkpoint and never retrained.
5. **Official-first semantic ladder.** The official SigLIP2-G spatial output
   and official crop summary were tested first. Because neither passed the
   frozen oracle gate, the current text capability is the explicitly labelled
   Level-3 variant: one global bridge trained on generic COCO crops, frozen for
   all scenes, followed by the official SigLIP2-G summary head. It was not
   trained on LERF/ScanNet scenes or their test vocabulary.
6. **One primitive support graph and solver.** Text, unregistered image crop,
   registered 2-D prompt, and world-space 3-D point compilers all produce
   primitive evidence. The same query-independent graph and diffusion solver
   produce the final 3-D support before projection.

Feature signatures distinguish provenance (`spatial`, `summary`, and
`primitive`) from final-space comparability. Comparison is allowed only when
the official adaptor name, RADIO checkpoint, output dimension, and
normalization match. Field, capability, and graph rows and signatures are
validated fail-closed.

## Semantic alignment decision

| Level | Teacher/readout | 2-D oracle mIoU | Decision |
|---|---|---:|---|
| 1 | Official SigLIP2-G spatial adaptor | 1.50% | Insufficient |
| 2 | Official re-encoded crop summary | 12.83% | Insufficient |
| 3 | Global generic-crop bridge -> official summary head | 44.87% | Current optional semantic variant |

Level 3 is not called an official SigLIP2 adaptor. The custom component predicts
a RADIO summary token; the final text-space projection remains the frozen
official head.

## Reconstruction evidence: ramen

The final field is
`output/canonical_fields/ramen_canonical_radio_d384_verified_pose_renderft256.pth`
(SHA-256 `318cb7a3d35850d0e1974a8be20a06506ebca30206e3f423c293f79996bbca4b`).

| Audit | Initial PCA field | Final render-corrected field |
|---|---:|---:|
| Primitive mean cosine vs MPR | 0.994042 | 0.993891 |
| Primitive p05 cosine | 0.984794 | 0.984736 |
| Non-benchmark dev-view render cosine | 0.740227 | 0.746503 |
| Unseen official 7-view render cosine | 0.734796 | 0.740665 |

All seven unseen official views improve. The small gain together with nearly
unchanged primitive fidelity shows that remaining render error is dominated by
visibility/occlusion and unobserved primitives rather than 384-D compression.
Only 221,608 of 382,687 ramen Gaussians (57.91%) have valid MPR observations.

## Query evidence completed

### Text query: LERF-3D ramen

- Exact official query strings and official SigLIP2-G text embeddings.
- Scene softmax followed by fixed query-peak normalization.
- Fixed shared support solver, unary center 0.6, support threshold 0.5.
- Selected primitives only are rendered; silhouette threshold is 0.7.
- No GrabCut, SAM mask decoder, connected-component oracle, or threshold sweep.

Final result over 71 ramen instances: **33.22% mIoU, 59.15% Acc@0.25,
26.76% Acc@0.50**. The pre-render-correction canonical field gave 32.94%,
56.34%, and 26.76%, respectively.

### Pose-free real-image crop query

The query crop is encoded by the official C-RADIO crop summary path and the
official `dino_v3_7b` spatial adaptor. It uses neither camera pose nor a target
mask. A query-independent canonical-scene mean supplies the fixed negative
baseline; this prevents the positive-only cosine unary from saturating the
whole scene and is part of the method, not test calibration.

On the ramen egg crop from `frame_00123.jpg`, the final field selects 6,307
primitives. As a diagnostic performed only after prediction, 4,802 (76.14%) of
their centers project into the source crop. This establishes a functioning,
crop-dependent 3-D interface, but it is not yet an UnCoCo benchmark number.

### Registered 2-D prompt: NVOS fern

- Strict prompt registration with the target view excluded from MPR.
- Deterministic CPU raster-responsibility accumulation.
- Official DINOv3/SAM3 capability prototypes and the same support graph/solver.
- No target mask access at query time and no post-hoc calibration.

One-scene result: **74.8769% foreground IoU and 91.5815% pixel accuracy**.
Removing graph diffusion gives 74.8262% IoU, so the graph contributes only
+0.0507 points on this deterministic path; it is not the main source of the
gain. This is a fern result, not the eight-scene NVOS macro.

### World-space 3-D point: ScanNet scene0000_00

- One deterministic point from each ground-truth instance is the declared
  query input; instance masks are otherwise excluded from field/graph/query
  construction and are used for metrics only.
- The point is lifted by Gaussian covariance, compiled with official
  DINOv3/SAM3 views, and solved by the shared support solver.
- Output is projected to the official ScanNet mesh-vertex domain with adaptive
  local-sigma kNN; Gaussian and mesh rows are never assumed identical.

Across 63 instances: **30.17% macro IoU, 47.62% Acc@0.25, and 17.46%
Acc@0.50**. The earlier best same-scene path was about 24.13% IoU.

## What is and is not yet a benchmark claim

- Completed real canonical evaluations cover ramen text, one ramen image-crop
  diagnostic, NVOS fern, and ScanNet `scene0000_00`.
- Full four-scene LERF, eight-scene NVOS, and multi-scene ScanNet aggregation
  still need to be run before filling a paper main table.
- Formal ten-scene SPIn-NeRF evaluation remains fail-closed because the Fork
  original RGB/camera asset is missing. A nine-scene diagnostic must not be
  reported as the formal benchmark.
- No comparison method has been reproduced in this iteration. Literature
  numbers can only be placed beside these results after checking query inputs,
  support-view access, projection domain, binarization, and metric aggregation.

## Verification

The focused regression suite covering the canonical field, signatures,
official capability cache, shared graph/solver, text compiler, pose-free image
compiler, NVOS prompt path, ScanNet point path, semantic ladder, and LERF
projection passes: **125 tests**. `git diff --check` also passes.
