# Next66: joint source reconstruction improves LERF response, not yet selected extent

## Outcome

The four-scene development mean improves from **58.087 to 64.882 2D IoU**, **74.965 to 81.021 AP**, and **82.245 to 87.328 localization** after expanded-source RGB+MoGe reconstruction and fresh native feature reconstruction. This is a meaningful system-level gain; extra legal source RGB is part of the method/input budget, not an equal-input architecture claim.

The actual selected-only Gaussian projection remains worse: **48.518 retained versus 43.748 new**, or **45.862 after matched 240-step source fitting**. Consequently this is a research candidate, not a joint default replacement or a claim that LERF3D improved. No new ScanNet, NVOS, UQIS, image-query or prompt-query result is reported here. These are reused development scenes, not independent test results or proof of SOTA/CVPR readiness.

## What changed in the method

The previous next52 carrier used sparse source RGB, fixed Gaussian count/rotations, bounded centre movement, and 1,500 updates. Next66 warm-starts that carrier and independently optimizes centre, quaternion, scale, opacity and RGB spherical harmonics (degree 0→1→2), then freezes geometry before rebuilding semantics. Semantic losses do not update geometry. This remains the geometry-first feature-field architecture, with MoGe as geometric prior rather than an assumed ground truth.

Three four-scene reconstruction arms received 3,000 additional updates each:

1. **Fixed count:** original source frames; train all geometric parameters and RGB.
2. **Adaptive:** same source frames/updates; add gradient-driven densification and pruning.
3. **Expanded RGB adaptive:** all eligible RGB frames, excluding every annotated frame and the two reserved source validation frames; retain original calibrated MoGe source frames. Depth sampling remains on that original cohort. This arm can require an extra depth render per update; it is not equal compute/input to arm 2.

The photometric objective is 0.8 L1 + 0.2 SSIM, plus 0.1 relative calibrated-MoGe depth error, at width 512. Cameras stay fixed. The initial scene radius used by the optimizer is the 90th percentile Gaussian radius, not camera radius. Its scale sensitivity remains a possible confound, not an established bug or tested fix.

On each changed carrier, rebuild native 1536D SigLIP features from the original source-region observations; recompute visibility and source scope fitting. The existing next53 confidence distribution is transported with separately computed old/new source visibility masses. Old row-indexed features/memberships are not copied onto new Gaussians. Semantic confidence still depends on the old distribution; this is not a wholly independent model or a completed four-teacher compact student.

The evaluator accepts an explicit semantic geometry/support binding so changed Gaussian counts can render their own field. The score calculation, annotations, thresholds and aggregation remain unchanged. Both semantic response and selected-set rendering use the new carrier. The latter still renders only the hard-selected Gaussian subset and thresholds alpha bytes above 10; it is not silently replaced by full-scene compositing.

## Results: fixed recipes, no per-query stitching

Four-scene macro means, percent:

| Recipe | 2D IoU | AP | Localization | Selected projection IoU |
|---|---:|---:|---:|---:|
| Retained next63 | 58.087 | 74.965 | 82.245 | 48.518 |
| Fixed count, sparse RGB | 60.447 | 76.831 | 82.923 | 38.954 |
| Adaptive, sparse RGB | 61.942 | 78.281 | 84.265 | 40.474 |
| Adaptive, expanded RGB | **64.882** | **81.021** | **87.328** | 43.748 |
| Same expanded carrier, 240-step source fitting | 64.882 | 81.021 | 87.328 | **45.862** |

The retained comparison includes extra geometry training and new membership initialization; it does not isolate a single optimizer parameter. Fixed versus adaptive more directly measures the added topology adaptation. Adaptive sparse versus expanded RGB measures a source-budget change, with the stated depth-render compute difference.

Expanded RGB per-scene results (120-step set decoding):

| Scene | 2D IoU | AP | Localization | Selected projection |
|---|---:|---:|---:|---:|
| figurines | 66.711 | 84.722 | 87.500 | 42.523 |
| ramen | 61.477 | 76.117 | 81.690 | 37.351 |
| teatime | 71.328 | 88.146 | 98.305 | 45.027 |
| waldo_kitchen | 60.012 | 75.097 | 81.818 | 50.091 |

“Selected projection” is a 2D annotation comparison after selecting and rendering 3D elements. It is not an independently measured 3D-GT IoU.

## Source coverage was a real bottleneck

Reserved-source PSNR (two fixed views per scene):

| Scene | Previous carrier | Fixed | Sparse adaptive | Expanded adaptive | Expanded RGB frames |
|---|---:|---:|---:|---:|---:|
| figurines | 14.00 | 14.55 | 14.41 | 20.39 | 293 |
| ramen | 16.09 | 16.41 | 16.41 | 20.89 | 122 |
| teatime | 19.96 | 19.70 | 19.52 | 24.80 | 169 |
| waldo_kitchen | 11.69 | 11.60 | 11.56 | 18.90 | 180 |

Original feature/depth training cohorts remain 20/25/23/22 frames. Expanded carrier counts are 554,467 / 592,709 / 1,829,615 / 1,496,561 Gaussians. Sparse adaptive training PSNR reaches 29.06/27.44/30.26/35.61 but does not improve reserved views over the fixed arm: densification on sparse inputs alone mainly fits those views better. Do not compare expanded and sparse train means as if evaluated over the same cohort.

The expanded carrier visibly restores previously missing/blurred structure and texture. This agrees with better retrieval response, but RGB appearance does not prove physical geometric correctness. Reserved MoGe AbsRel improves in three scenes; **waldo worsens from 0.0242 to 0.0809**, while coverage rises from 0.9137 to 0.9915. Coverage and the different contributing pixel sets complicate this comparison. Neither metric is ground-truth depth accuracy.

Visualization: [reserved RGB / old / sparse adaptive / expanded adaptive](../../output/joint_carrier_next66/source_budget_comparison.png).

## Why this is not yet a complete solution

Semantic response and hard element selection consume geometry differently. A normalized full-scene feature response can improve while selected-only alpha accumulates contributions from too many inappropriate elements. New topology invalidates old membership coordinates; source masks do not uniquely identify a geometrically clean set across views.

Observed diagnostic on the sparse adaptive arm: figurines selected precision/recall changes from **52.45/90.44 to 38.09/94.82**; teatime from **66.19/88.46 to 42.39/91.69**. This identifies substantial extra false positives, not simply missing targets. Annotation-based precision/recall was diagnostic only, not a training target. It does not by itself prove which geometric parameter or source label caused each false positive.

Completed mechanism controls:

| Selected-set control | Sparse adaptive | Expanded adaptive |
|---|---:|---:|
| Normal 120 steps | 40.474 | 43.748 |
| Transfer old soft membership through source projections, then fit | 37.414 | Not run |
| Admit teacher-complement negatives despite old native positives | 40.561 | 43.968 |
| 120 steps old actual hard-subset source-output distillation + 120 source fitting | Not run | 44.206 |
| Matched 240 steps source fitting, same annealing | Not run | **45.862** |

Soft transfer on fixed-count fig/ramen also failed (29.839/29.495); it remains a two-scene pilot, with no fabricated four-scene macro. Output distillation uses the consumed hard selected-only render, unlike soft-state transfer, but remains a source-view approximation at the existing 96×128 fitting raster. It still uses views eligible under the current teacher supervision. Its benefit does not exceed the matched continuation budget. Simply relaxing negatives is almost neutral and risks labeling unknown scope as negative. None of these controls fixes independent target identity/whole-part scope supervision.

## Research decision and next priority

Keep the expanded RGB reconstruction as the strong **2D semantic candidate**, with explicit input/cost reporting. Do not stitch its semantic output to old geometry for selection and claim one unified repaired map. Keep the retained joint recipe available until the selected-set regression is resolved.

The next central method question is how to reconstruct object membership on changed Gaussian support from **visibility-aware, identity-consistent source constraints**, preserving exclusion as well as coverage. Optimize/evaluate the actual consumed selected render; separate occluded/unknown pixels from trustworthy background. A high-quality soft feature map alone does not establish a clean object set. Higher-resolution source extent evidence and opacity/support factorization are hypotheses to test, not completed fixes. The new carrier's depth inconsistency, especially waldo, also needs isolation before a claim of repaired geometry.

For a paper, full RGB reconstruction is currently an enabling baseline, not sufficient novelty. A defensible contribution would demonstrate that the proposed native-semantic/extent factorization solves this response-versus-membership mismatch under a common input protocol, then transfers to image/prompt queries and other datasets. Those results are still missing.

## Reproduction and artifacts

- `scripts/research/run_joint_carrier_next66.py`: three geometry arms.
- `scripts/research/run_joint_semantic_next66.py`: fresh native reconstruction, scope, confidence transport, set fitting, frozen evaluation.
- `scripts/research/run_joint_transfer_next66.py`, `run_joint_exclusion_next66.py`: source-membership and negative controls.
- `scripts/research/run_output_transfer_next66.py --arm adaptive_allrgb --seed-mode output --gpus 0,1,4,5`.
- Same command with `--seed-mode continuation`: matched optimization budget.
- `scripts/research/summarize_joint_carrier_next66.py`: complete-scene summaries only.
- `output/joint_semantic_next66/summary.json`: per-scene metrics, geometry budgets, macro results.
- `output/joint_carrier_next66/<arm>/<scene>/fit.json`: geometry logs and split IDs.

All launched reconstruction, semantic and selection jobs finished successfully. Validation here is actual reconstruction and benchmark execution; no defensive test suite or audit framework was added. Historical fields in copied decoder metadata may mention next54/readout inheritance; the actual new semantic/carrier bindings and executed commands identify these runs. The new output-transfer script descriptions now name that binding accurately.
