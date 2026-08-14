# Five-benchmark core mainline — 2026-08-15

This is the single active experiment record for the post-grill cleanup. It
records development evidence, not a final five-benchmark claim.

## Frozen method decisions

- The persistent scene state is the schema-v2 factorized D512/L512 canonical
  RADIO field. Target reliability is an objective weight, not deployment state.
- Raw RADIO reconstruction owns the radial/log-amplitude gauge. SigLIP2,
  DINO, SAM, and region-summary objectives receive tangent-only gradients.
- Official SigLIP2 spatial projection runs on the complete 2-D token grid.
  Applying the SigLIP2 summary head independently to pixels or Gaussians is a
  retired proxy and is legacy-opt-in only.
- A valid region branch predicts a RADIO summary token from a region, then
  applies the frozen official summary head. Its teacher is generated from
  genuine source-RGB crops. Benchmark RGB, masks, queries, and labels are not
  field-training inputs.
- LERF primitive and region scores may be combined only by the global frozen
  1:1 rule in this development slice; no scene or query weight selection is
  permitted.

## Figurines source-only development slice

All runs use the four frozen LERF-OVS benchmark frames 41, 105, 152, and 195
only after training and candidate selection. The 295-frame feature and crop
teacher bundles exclude those frames.

| Candidate/readout | Raw validation | SigLIP2 spatial | Region summary | MPR probe | LERF2D mIoU | LocAcc |
|---|---:|---:|---:|---:|---:|---:|
| Initial D512/L512 field, primitive | 0.57957 | 0.62781 | — | 0.99362 | 0.37238 | 0.8750 |
| + 64-step official spatial, primitive | 0.58120 | 0.63653 | — | 0.99364 | 0.37298 | 0.8750 |
| + 64-step genuine region teacher, primitive | 0.58260 | 0.64182 | 0.50021→0.50914 | 0.99361 | 0.37388 | 0.8571 |
| Same field, region only | — | — | — | — | 0.17289 | 0.5357 |
| Same field, fixed 1:1 primitive+region scores | — | — | — | — | 0.34881 | 0.8750 |

The representation objectives improve their source-only validation metrics,
but the frozen LERF2D gain is negligible and the region branch hurts when used
directly or at fixed equal weight. This is a negative result for the current
bridge/readout, not evidence for per-scene fusion tuning.

## Current bottleneck and next candidate

The field is no longer the dominant demonstrated bottleneck on this slice.
The next method candidate is a global generic text-response objective with
listwise and sibling-negative supervision over genuine region summaries. It
must be trained without benchmark vocabulary and evaluated with the same
frozen primitive/region output operators.

## Five-benchmark readiness

- LERF2D: one legal Figurines development slice completed; full four-scene
  D512/L512 cohort remains to run.
- LERF3D: frozen protocol exists; needs the same full LERF field cohort and
  typed 3-D output execution.
- ScanNet OVS: genuine sparse region-view teachers and several D512/L512
  fields exist, but the paper eight-scene cohort is incomplete.
- NVOS: frozen protocol exists; unified D512/L512 fields and the public RGB/SAM
  transient prompt adapter are not yet materialized.
- Available-Nine SPIn-NeRF: frozen protocol exists; unified D512/L512 fields
  and legal RGB/SAM transient selection must replace the historical
  benchmark-assisted branch.

No joint five-benchmark row is eligible yet. Existing historical peak numbers
must not be combined into a virtual incumbent.

## Reproducible artifacts

- Spatial field: `official_siglip2_spatial_w005_s0_64.pth`
- Spatial+region field: `official_siglip2_spatial_region_w005_s0_64.pth`
- Source-only official spatial bundle:
  `canonical_teacher_features_v2/figurines_source_only_siglip2`
- Source-only genuine crop-summary bundle:
  `optimization_20260716/semantic_teacher_train`
- Frozen global region bridge:
  `global_region_summary_coco15000_full_context_local_scales_imageholdout.pth`
- Frozen evaluation reports: `lerf2d_eval_initial`, `lerf2d_eval`,
  `lerf2d_eval_spatial_region_field_raw_readout`,
  `lerf2d_eval_region_readout`, and `lerf2d_eval_typed_two_level` under
  `optimization_20260815/core_method_v1/figurines`.
