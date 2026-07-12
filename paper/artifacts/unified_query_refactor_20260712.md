# Unified query-field refactor and first end-to-end runs (2026-07-12)

## Implemented architecture

The scene representation remains one compact RADIO-GS field.  Query-time
readout is now separated into two frozen spaces instead of benchmark-specific
heads:

- semantic space: decoded 1280d RADIO field feature -> frozen SigLIP2 summary
  head -> text prototypes;
- region space: decoded 1280d RADIO field feature -> official frozen RADIO
  `sam3` feature-projection adaptor (1280d -> 1024d) -> visual/prompt prototype.

`radio_gs/querying/unified_query.py` defines the common typed interface:

- `QuerySpec(kind, space, positive_prototypes, negative_prototypes, seeds)`;
- cosine query bank for multiclass text classification;
- positive-minus-hardest-negative cosine margin for text and binary prompts;
- a reusable spatial/feature k-NN support graph;
- fixed-parameter, label-free propagation and seed-connected support.

All thresholds and graph parameters are explicit protocol inputs.  The module
contains no ground-truth access and performs no threshold search or test-set
calibration.

## Real runs

### Text query

The LERF2D, LERF3D, and ScanNet text paths now call the shared GPU cosine
scorer.  One real-checkpoint scene from every evaluator was rerun after the
change.

| Task | Real smoke | Result | Frozen-result check |
|---|---|---:|---|
| LERF2D | ramen, 71 instances | mIoU 29.1737%, LocAcc 78.8732% | same displayed precision; mIoU delta +0.0014 percentage points |
| LERF3D | ramen, raw relevance > 0.6 | mIoU 0.1847%, Acc@0.25 0% | same displayed precision; mIoU delta +5.8e-8 percentage points |
| ScanNet text | scene0000_00, VALA Gaussian protocol | split19 mIoU 18.6869%, mAcc 40.9560% | exact scene metrics reproduced |

The shared stable relevance function is bit-exact to the previous formula on
identical normalized tensors.  The tiny LERF rerun deltas above arise after a
fresh GPU render/decode pass and do not change any displayed table value.

The previously frozen full cohorts remain:

- LERF2D four-scene macro: 37.61 mIoU / 81.27 LocAcc;
- LERF3D raw-0.6 four-scene macro: 0.12 mIoU / 0.00 Acc@0.25;
- ScanNet eight-scene provisional VALA-Gaussian macro: split19 20.76 mIoU /
  39.42 mAcc (still provisional because the splitwise pseudo-GT remap audit is
  not resolved).

### Registered 2D prompt

The NVOS/SPIn feature readout now constructs `QuerySpec` and calls the shared
cosine-margin scorer.  Prediction remains separated from the evaluator, so
target GT cannot be read while scores are produced.

NVOS was rerun on all eight real trained fields and all eight official target
views:

- foreground IoU: 61.1588245%;
- pixel accuracy: 91.2350088%;
- protocol hash: `d21b52d3c2155f82d63c18a6c9c4a56eb25c42f09c10306aa924135317cadaf9`.

The result is numerically identical before and after the refactor.

SPIn-NeRF remains incomplete for a formal result because the original Fork RGB
and camera bundle is absent.  A strictly labelled nine-scene diagnostic
manifest was built for the 423 available annotated frames, with
`formal_10scene_eligible=false`, `missing_scenes=[fork]`, and protocol hash
`d8a87284ddc2fde946a5d9de83aec190487e61c72259dc62656be603c2af6752`.
Its nine-scene train/render/predict queue is materialized, but no score is
reported because those feature fields have not yet been trained.

### ScanNet 3D point query

A new fail-closed evaluator reads official ScanNet `aggregation.json` and
`segs.json`, samples one deterministic GT point per eligible instance, and
reveals no other instance pixels to the query method.  The field uses official
RADIO SAM3-adaptor region features.  The declared method is:

1. positive prototype = feature at the sampled 3D point;
2. negative prototype = unlabeled scene-mean region feature;
3. shared cosine margin, fixed threshold 0 with `>=` comparison;
4. four iterations of fixed spatial/feature propagation;
5. retain the spatial component connected to the query seed.

On the real scene0000_00 checkpoint, all 63 instances with at least 100 mesh
vertices were evaluated:

| Readout | Macro IoU |
|---|---:|
| Raw cosine margin | 5.93% |
| Shared graph propagation | 7.75% |
| Seed-connected propagated mask | 19.00% |

Final Acc@0.25 is 26.98% and Acc@0.50 is 14.29%.  These are first-run
diagnostics, not paper-ready multi-scene results.  They show that the shared
interface runs correctly, while a one-point prototype plus scene-mean negative
is not yet a strong instance readout.

## Current release gates

1. Do not put the nine-scene SPIn diagnostic in a ten-scene main-table row.
2. Do not tune point-query threshold, graph scale, or component radius on
   scene0000 test instance masks.  Freeze them from a disjoint development
   cohort or report the current values as preregistered diagnostics.
3. Download the official instance annotation JSONs for the remaining ScanNet
   scenes and run the fixed protocol unchanged before reporting a dataset
   macro.
4. The full-reference-mask SPIn prompt is not the undisclosed sparse-click SAGA
   interaction and must remain a separately named diagnostic.

## Verification

- targeted query/prompt/LERF/ScanNet tests: all pass;
- repository-wide suite: 723 passed, 3 unrelated pre-existing failures;
- `git diff --check`: pass.

The three unrelated failures are a capitalization-sensitive documentation
string, the existing train-script line-count release audit, and the existing
OpenCLIP frame-order expectation.  None touches the files changed by this
refactor.
