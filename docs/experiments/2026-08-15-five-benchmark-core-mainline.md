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
| + 64-step genuine region teacher, primitive | 0.58260 | 0.64182 | 0.50021→0.50914 | 0.99361 | 0.37394 | 0.8571 |
| Same field, region only | — | — | — | — | 0.17289 | 0.5357 |
| Same field, fixed 1:1 primitive+region scores | — | — | — | — | 0.34884 | 0.8750 |
| + generic response, frozen basis, primitive | 0.58363 | 0.64510 | 0.51885 | 0.99356 | 0.37558 | 0.8571 |
| + generic response, joint basis, early-stop step 1, primitive | 0.58564 | 0.64703 | 0.52567 | 0.99347 | 0.37580 | 0.8571 |

The generic response objective uses 806 target-blind generic text directions,
657 synonym directions, and 167 sibling relations. With the factorized basis
frozen, its validation loss falls from 0.32233 to 0.29122, response-profile
cosine rises from 0.11134 to 0.19667, and region validation rises from 0.50914
to 0.51885. Frozen LERF2D mIoU rises by only 0.00163 over the preceding field.
The same checkpoint's region-only and fixed 1:1 outputs fall to 0.15772 and
0.34517 mIoU respectively, so the current region readout remains rejected.

Joint basis optimization produces a slightly stronger early-stop checkpoint,
but its LERF2D increment over frozen-basis response training is only 0.00022.
Continuing to step 64 drops raw validation from 0.58260 to 0.57177 and MPR from
0.99361 to 0.98552. The persistent basis therefore remains frozen by default;
the one-step result is retained as an ablation, not promoted as a new method.

## Readout conclusion

The field objectives now preserve official spatial, genuine crop-summary, and
generic text-response capabilities without benchmark vocabulary. Their
source-only validation gains do not translate into a material LERF2D gain.
The demonstrated bottleneck is the typed region/query readout, not missing
field capacity. Per-scene fusion tuning, graph repair, and connected-component
repair remain prohibited because they would conceal rather than solve that
interface error.

## NVOS/SPIn query-transient RGB/SAM adapter

The common persistent/transient seam is now explicit in
`radio_gs/querying/transient_rgb_sam.py` and is emitted by both the audited
LUDVIG reproduction wrapper and the SPIn reference selector:

- the persistent field ends at a signed prompt and never stores target RGB or
  SAM state;
- ten deterministic trials use three positive and three negative points;
- target RGB and frozen SAM are query-transient, while target masks and target
  metrics are unavailable during proposal generation;
- SPIn may calibrate candidate/threshold on its one permitted reference mask;
  NVOS scribbles do not gain that full-mask calibration authority;
- exact positive/negative observations are clamped after SAM fusion; conflicts
  preserve the base posterior; no graph or connected component is applied.

Re-aggregating all nine sealed historical SPIn reports under the stricter
contract reproduces canonical 0.877162, SAM-only 0.946957, and reference-only
selected 0.948415 macro foreground IoU. The latter is +0.011214 over the local
LUDVIG-SAM reproduction. This validates the adapter and selector semantics,
but it is still a historical carrier rather than the new D512/L512 field.
NVOS's released-compatible target-RGB path remains the audited 0.912577
reference; it has likewise not yet been regenerated from the new field.

## Five-benchmark readiness

- LERF2D: one legal Figurines development slice completed; full four-scene
  D512/L512 cohort remains to run.
- LERF3D: frozen protocol exists; needs the same full LERF field cohort and
  typed 3-D output execution.
- ScanNet OVS: genuine sparse region-view teachers and several D512/L512
  fields exist, but the paper eight-scene cohort is incomplete.
- NVOS: the public-compatible signed RGB/SAM transient adapter is implemented
  and audited; unified full-eight D512/L512 fields are not materialized.
- Available-Nine SPIn-NeRF: frozen protocol exists; unified D512/L512 fields
  must replace the historical carrier. Legal reference-only transient
  selection is now implemented and reproduces its sealed full-nine result.

No joint five-benchmark row is eligible yet. Existing historical peak numbers
must not be combined into a virtual incumbent.

## Reproducible artifacts

- Spatial field: `official_siglip2_spatial_w005_s0_64.pth`
- Spatial+region field: `official_siglip2_spatial_region_w005_s0_64.pth`
- Frozen-basis generic-response field: `generic_text_response_w005_s0_64.pth`
- Joint-basis early-stop ablation:
  `generic_text_response_basis_w005_lr5e4_s0_64.pth`
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

## Verification

- 121 focused field, finetune, renderer, response, adapter, and frozen-protocol
  tests pass.
- 104 audited LUDVIG wrapper/full-carrier and SPIn adapter tests pass.
- Python compilation and `git diff --check` pass.
- The frozen five-contract validator still reports zero eligible joint rows;
  it correctly refuses to stitch historical peak numbers into a virtual
  five-benchmark result.
