# Query-consistency iterative evidence (2026-07-12)

This artifact records controlled experiments performed after the unified-query
refactor.  Diagnostic oracle values are never promoted to protocol results, and
no target-set threshold calibration is used.

## Decisions backed by completed experiments

| Hypothesis / change | Controlled evidence | Decision |
|---|---|---|
| Token-wise HCD normalization fixes map/point inconsistency | Map/point parity tests pass after replacing spatially coupled GroupNorm | Keep |
| Primitive-first ordering alone fixes LERF | Ramen LERF2D primitive score/query readout: 0.24 mIoU; exact full-dimensional center-lift oracle: 0.03 mIoU | Ordering is necessary for a valid comparison but not sufficient for accuracy |
| True multiview center supervision fixes direct 3D | Ramen LERF3D top-2% mIoU improves 0.36 to 3.76 (10.4x), but remains far below usable | Keep as diagnostic evidence; do not replace the screen field |
| Per-view SigLIP projection before fusion fixes direct 3D | Query-space multiview checkpoint reaches 4.03 top-2% mIoU and collapses legacy LERF2D to 1.53 | Reject as a shared-field replacement |
| Frozen point adapter can preserve 2D and repair 3D | Ramen screen LERF2D remains 28.42 mIoU / 78.87 LocAcc, but direct 3D top-2% is 0.04 | Reject adapter as the direct-3D solution; retain the original screen branch |
| Gaussian-first NVOS is already stronger than screen-to-screen | Eight-scene SAM3 Gaussian-first macro IoU 55.60 versus screen 61.16 | Report separately; do not replace the stronger screen result |
| Raw RADIO is a universally better Gaussian region metric | Fern 71.10 vs SAM3 71.89; leaves 13.59 vs SAM3 11.09 | Insufficient and inconsistent; reject a global replacement |
| Four prompt prototypes solve multimodal appearance | Fern 71.03 vs one-prototype 71.89; leaves 12.23 vs 11.09 | Reject as default |
| Local multiscale one-click prototypes fix ScanNet | 63 instances: raw 11.6501 vs 11.6888; propagated 12.2062 vs 12.2314; connected 24.0004 vs 24.1280 | Reject as default; retain the simpler single-point readout |
| Register high-dimensional view features before text scoring | Corrected score-first registration reaches 34.48 scene-macro mIoU on LERF3D; feature-first lifting previously reached only 4.03 on ramen | Reject feature-first lifting; retain query-first scalar lifting |

## Valid positive results

- Region teacher oracle (474 queries): raw RADIO AP 0.733 / best IoU 0.615;
  SAM3-adaptor AP 0.475 / best IoU 0.367; DINO AP 0.705 / best IoU 0.589.
- Ramen text teacher oracle: 29.79 mIoU / 87.32 localization accuracy.
- True center-sampled multiview primitive supervision improves direct ranking by
  10.4x, identifying lifting/aggregation quality rather than only decoder
  normalization as the remaining bottleneck.
- Gaussian-first NVOS improves fortress by +2.21, horns-center by +3.47, and
  horns-left by +27.24 IoU points, proving that a view-independent support can
  work even though its current macro score is lower.
- ScanNet evaluation is now performed in the official mesh-vertex domain; the
  earlier same-row Gaussian/mesh assumption is not treated as valid evidence.

## Current candidate under test

`multiview_score_lift` performs the query in each protocol-permitted training
view first, backprojects only the scalar cosine margin, averages it on Gaussian
supports, and renders one target-independent 3D score.  It avoids averaging
view-dependent high-dimensional visual tokens.  Evaluation camera poses, target
RGB, and target masks are excluded from support construction.  The threshold is
fixed at zero.

This candidate is now validated on all eight NVOS scenes:

| Scene | Score-lift | Primitive Gaussian-first | Screen-to-screen |
|---|---:|---:|---:|
| fern | 72.34 | 71.89 | 72.00 |
| flower | 85.81 | 68.00 | 83.85 |
| fortress | 93.24 | 92.15 | 89.94 |
| horns-center | 58.31 | 56.09 | 52.62 |
| horns-left | 30.24 | 48.85 | 21.61 |
| leaves | 24.33 | 11.09 | 33.02 |
| orchids | 65.75 | 59.12 | 71.95 |
| trex | 65.14 | 37.63 | 64.29 |
| **Macro** | **61.89** | **55.60** | **61.16** |

Score-lift improves six of eight scenes over primitive-feature Gaussian-first
and also exceeds the screen-to-screen macro by 0.74 IoU points while producing
one target-independent 3-D score.  It is therefore retained as the new shared
support main path.  Horns-left is the main counterexample and motivates future
local/object-component support, not test-set threshold tuning.

## LERF3D text score-lift

The same query-first construction was applied to text queries.  Every annotated
evaluation camera is removed from the support set (`train_nonannotated`).  A
SigLIP relevancy map is computed in each remaining training view, raster
contributions lift only the independent scalar score of each text query, and
the resulting weighted mean is rendered from the selected Gaussians.  All
scenes use the same raw query strings, fixed top-2% selection, and silhouette
threshold 0.7; there is no per-scene or test-set calibration.

| Scene | mIoU | Acc@0.25 | Acc@0.50 | Instances |
|---|---:|---:|---:|---:|
| figurines | 29.49 | 50.00 | 14.29 | 56 |
| ramen | 44.37 | 77.46 | 46.48 | 71 |
| teatime | 44.70 | 71.19 | 45.76 | 59 |
| waldo_kitchen | 19.36 | 36.36 | 4.55 | 22 |
| **Scene macro** | **34.48** | **58.75** | **27.77** | — |
| **Instance weighted** | **37.81** | **63.94** | **33.17** | **208** |

An implementation audit caught and removed an invalid row-wise L2
normalization over the per-query scalar means.  The table above is the fully
rerun result after the correction; the earlier 43.50 ramen diagnostic is not
used.  Corrected ramen is 44.37, so the improvement is not caused by that
normalization.  This validates the attachment's non-commutation diagnosis:
high-dimensional view-token fusion destroys query information, whereas
querying first and registering scalar evidence preserves it.  Waldo remains
the principal failure case.

## SPIn-NeRF status

The nine-scene frozen queue is protocol-complete.  Fern geometry completed at
30k iterations with 1,314,717 Gaussians, and RADIO feature extraction / field
training is now running.  Two task-independent engineering
issues were fixed before restarting the queue:

1. exact 50k-point nearest-neighbour scale initialization now uses a one-worker
   cKDTree instead of a 50k-by-50k `torch.cdist` matrix;
2. full-resolution training images remain uint8 on CPU and are converted to
   float only for the sampled GPU frame.

Both changes preserve pixels, geometry statistics, and the frozen evaluation
protocol.  Dedicated unit tests pass.

The SPIn audit also found that annotation aliases such as `image001` do not
equal their resolved RGB/COLMAP camera names such as `IMG_4027`.  Support-view
exclusion now uses the frozen resolver's camera names rather than comparing the
aliases directly.  For fern this is verified before evaluation: 20 scene
training cameras minus 19 evaluation cameras leaves exactly the registered
prompt camera (`IMG_4026`) as query-time support.  Prompt prototypes default to
the observed, registered real-image RADIO→official-SAM-adaptor feature rather
than a re-rendered prompt feature.  Target RGBs, masks, and cameras remain
unopened/unused at query time.

The first formal SPIn result is now complete on fern.  Geometry uses 1,314,717
Gaussians trained for 30k iterations; the compact field's best checkpoint is
epoch 240, selected only by label-free decoded-feature cosine 0.8591.  With one
registered full-mask reference view and all 19 target cameras excluded from
query-time support, fixed zero threshold gives **56.12 foreground IoU** and
**93.07 pixel accuracy** across the 19 target views.  No connected-component,
GrabCut, SAM-decoder, or target-set calibration is used.

A same-checkpoint causal ablation replaces the observed reference feature with
the field's re-rendered reference feature: IoU drops from 56.12 to 55.81 and
pixel accuracy from 93.07 to 92.92.  The modest but consistent +0.31 IoU gain
confirms that a registered real-image prompt should enter through its observed
query encoder rather than incur an unnecessary render/decode round trip.

## Verification

The final focused regression suite covers HCD map/point decoding, unified
query propagation, LERF3D registration order, NVOS/SPIn prompt safety and
camera aliases, ScanNet mesh-domain point queries, COLMAP image mapping, and
direct point supervision: **154 tests pass**.  `git diff --check` also passes.
