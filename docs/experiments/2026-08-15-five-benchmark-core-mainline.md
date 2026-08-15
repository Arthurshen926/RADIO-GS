# Five-benchmark core mainline — 2026-08-15

This is the single active experiment record for the post-grill cleanup. It
records development evidence, not a final five-benchmark claim.

The frozen joint-development identity is
`paper/artifacts/five_benchmark_method_v1_authority_20260815.json`. The older
five-contract gap matrix remains useful as an evaluation inventory, but its
historical compiler identity no longer defines the candidate method.

The live materialization gap is recorded in
`paper/artifacts/five_benchmark_method_v1_asset_inventory_20260815.json` and
enforced per field by
`python -m radio_gs.scripts.validate_five_benchmark_method_v1_field`.

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

## Frozen Method-v1 LERF full-four results

All four shared LERF fields now pass the executable Method-v1 gate, including
the exact predecessor-file hashes for base, official spatial, and genuine
crop-summary stages. The first completed full-four evaluation rendered the
RADIO field and then applied the SigLIP2 head. It is retained as a useful
dense-readout diagnostic, but it is not the frozen Method-v1 readout and is
not eligible for a Method-v1 row.

| Scene | Samples | LocAcc | sample mIoU |
|---|---:|---:|---:|
| Figurines | 56 | 0.85714 | 0.37559 |
| Ramen | 71 | 0.81690 | 0.20185 |
| Teatime | 59 | 0.89831 | 0.37762 |
| Waldo Kitchen | 22 | 0.81818 | 0.29908 |
| Full four diagnostic | 208 | 0.85096 | 0.30877 |

The diagnostic scene-macro mIoU is 0.31353 and category-macro mIoU is 0.30849.

The exact frozen Method-v1 LERF2D readout scores text on each primitive and
then renders the scalar score, with no primitive-confidence modification and
no mask refinement:

| Scene | Samples | LocAcc | sample mIoU |
|---|---:|---:|---:|
| Figurines | 56 | 0.91071 | 0.37021 |
| Ramen | 71 | 0.85915 | 0.21985 |
| Teatime | 59 | 0.89831 | 0.37536 |
| Waldo Kitchen | 22 | 0.81818 | 0.31187 |
| Full four Method-v1 | 208 | 0.87981 | 0.31417 |

The Method-v1 LERF2D scene-macro mIoU is 0.31932 and category-macro mIoU is
0.31303. This exact readout is slightly stronger than the dense diagnostic,
but it is not a SOTA result; Ramen remains the largest 2-D bottleneck.

The frozen `vala_paper_3d` evaluation uses primitive relevancy, the fixed 0.6
score threshold, selected-only alpha projection, the released `>10/255`
silhouette rule, and no postprocessing:

| Scene | Samples | mIoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|---:|
| Figurines | 56 | 0.44840 | 0.78571 | 0.44643 |
| Ramen | 71 | 0.29173 | 0.50704 | 0.21127 |
| Teatime | 59 | 0.32871 | 0.54237 | 0.22034 |
| Waldo Kitchen | 22 | 0.19815 | 0.36364 | 0.18182 |
| Full four Method-v1 | 208 | 0.33450 | 0.57692 | 0.27404 |

The LERF3D scene-macro mIoU is 0.31675. Waldo Kitchen is the largest 3-D
bottleneck. These two complete LERF evaluations establish that field
materialization is no longer the immediate LERF blocker; the primitive-query
score geometry and typed readout are.

Waldo also exposed a legitimate grid-rounding boundary: the frozen crop
teacher is 46x62 while its native RADIO grid is 45x62. The shared semantic
alignment interface now permits only a one-cell full-extent bilinear alignment
and continues to reject larger or channel mismatches. Both region fidelity and
generic response use the same helper. The original Waldo construction then
completed, and the final generic stage improved generic loss 0.29299 to
0.27985, profile cosine 0.18656 to 0.22345, and region validation 0.55012 to
0.55796 while retaining MPR within 0.00003.

## ScanNet OVS Method-v1 scenes 0000, 0062, 0070, 0097, 0347, 0400, and 0590

Scene0070 is the first paper-eight ScanNet scene rebuilt as the exact
Method-v1 field. Scene0000, scene0062, scene0097, scene0347, scene0400, and
scene0590 now reproduce the same source-only chain. In scene order 0000, 0062,
0070, 0097, 0347, 0400, and 0590, their deterministic fidelity holdouts are
respectively frames 1100/2220/3340/4460, 140/280/440/580, 260/520/800/1060,
140/300/440/600, 200/420/640/860, 280/520/760/1000, and
540/1080/1620/2160; the exact-marginal base construction excludes only those
four views in each scene. All seven final
fields pass the strict schema-v2 D512/L512 lineage gate, and each
query-independent primitive cache is SHA-bound to its field and geometry.

For the original five-scene slice, the three capability stages all improve
their own held-out objective while retaining the exact-marginal MPR probe:

| Stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| Official SigLIP2 spatial | 0.74106 → 0.74520 | 0.75649 → 0.75703 | 0.99566 → 0.99567 |
| Genuine crop summary | 0.53660 → 0.54418 | 0.75703 → 0.75739 | 0.99567 → 0.99565 |
| Generic text response | 0.29498 → 0.27791 loss; 0.18467 → 0.23178 cosine | 0.75739 → 0.75764 | 0.99565 → 0.99561 |

The frozen Gaussian-center evaluator consumes the external primitive cache
directly, checks its cache hash and embedded field provenance, row count, XYZ
alignment, feature dimension, validity mask, method identity, and
no-postprocessing contract, then applies
the fixed five-prompt SigLIP2 text ensemble. It performs no target-scene
calibration or prediction postprocessing:

| Split | VALA volume mIoU | VALA volume mAcc | Row mIoU | Row mAcc |
|---|---:|---:|---:|---:|
| 19 classes | 0.28825 | 0.49546 | 0.31334 | 0.49073 |
| 15 classes | 0.30541 | 0.44857 | 0.32720 | 0.43890 |
| 10 classes | 0.32357 | 0.69809 | 0.40309 | 0.70776 |

The scene0070 row above was the initial valid single-scene development
sentinel, not a paper-eight aggregate or a SOTA claim. At that checkpoint the
legal inventory was 5/29 overall and 1/8 for ScanNet.

Scene0097 and scene0347 then completed the identical capability chain:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0097 official SigLIP2 | 0.77265 → 0.77807 | 0.77587 → 0.77612 | 0.99744 → 0.99749 |
| 0097 genuine crop summary | 0.54404 → 0.55346 | 0.77612 → 0.77627 | 0.99749 → 0.99748 |
| 0097 generic response | 0.28744 → 0.26044 loss; 0.20324 → 0.27869 cosine | 0.77627 → 0.77629 | 0.99748 → 0.99741 |
| 0347 official SigLIP2 | 0.75276 → 0.75715 | 0.76032 → 0.76107 | 0.99663 → 0.99671 |
| 0347 genuine crop summary | 0.54795 → 0.55849 | 0.76107 → 0.76161 | 0.99671 → 0.99672 |
| 0347 generic response | 0.32283 → 0.29430 loss; 0.10467 → 0.18411 cosine | 0.76161 → 0.76200 | 0.99672 → 0.99667 |

Scene0400 and scene0590 complete two further instances with the same frozen
weights, step budget, and selection policies:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0400 official SigLIP2 | 0.75151 → 0.75500 | 0.74618 → 0.74703 | 0.99706 → 0.99708 |
| 0400 genuine crop summary | 0.55965 → 0.56811 | 0.74703 → 0.74771 | 0.99708 → 0.99707 |
| 0400 generic response | 0.29610 → 0.27534 loss; 0.17828 → 0.23606 cosine | 0.74771 → 0.74818 | 0.99707 → 0.99704 |
| 0590 official SigLIP2 | 0.76952 → 0.77176 | 0.77409 → 0.77463 | 0.99622 → 0.99626 |
| 0590 genuine crop summary | 0.57323 → 0.57831 | 0.77463 → 0.77503 | 0.99626 → 0.99626 |
| 0590 generic response | 0.28262 → 0.26834 loss; 0.21525 → 0.25491 cosine | 0.77503 → 0.77532 | 0.99626 → 0.99622 |

Scene0000 then completes the same chain without a CLI frame allowlist. The
configured authority selects exactly 275 train frames from the 279-frame
source bundle after excluding the four frozen fidelity frames:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0000 official SigLIP2 | 0.70774 → 0.71080 | 0.69755 → 0.69799 | 0.99464 → 0.99478 |
| 0000 genuine crop summary | 0.57030 → 0.57574 | 0.69799 → 0.69828 | 0.99478 → 0.99483 |
| 0000 generic response | 0.29160 → 0.28569 loss; 0.19509 → 0.21134 cosine | 0.69828 → 0.69836 | 0.99483 → 0.99481 |

Scene0062 was rebuilt from a current 37-frame source bundle and its v14
geometry. Its exact-marginal authority covers 33 training views after holding
out frames 140/280/440/580; 49,720 of 51,610 Gaussians receive valid source
observations. The three frozen stages improve every primary objective:

| Scene/stage | Primary held-out change | Raw validation | MPR probe |
|---|---:|---:|---:|
| 0062 official SigLIP2 | 0.73346 → 0.73972 | 0.75124 → 0.75171 | 0.99766 → 0.99774 |
| 0062 genuine crop summary | 0.54672 → 0.55769 | 0.75171 → 0.75205 | 0.99774 → 0.99774 |
| 0062 generic response | 0.34217 → 0.30761 loss; 0.05218 → 0.14803 cosine | 0.75205 → 0.75222 | 0.99774 → 0.99766 |

Scene0400 also exposed a real cohort-authority defect: its source image
directory contains 64 extracted frames, but only 61 have registered training
poses. The finetune entrypoint previously ignored the configured
`train_frame_ids_path` unless `--include-frame-ids` was repeated on the command
line. It now resolves an explicit CLI allowlist first and otherwise requires
the configured frozen allowlist, failing closed on a missing or empty file.
The corrected path ran scene0590 without a CLI allowlist and selected exactly
its 135 registered frames.

Their frozen primitive-readout ScanNet metrics are:

| Scene | Split | VALA volume mIoU | VALA volume mAcc |
|---|---:|---:|---:|
| 0000 | 19 | 0.34441 | 0.70967 |
| 0000 | 15 | 0.32313 | 0.72529 |
| 0000 | 10 | 0.35116 | 0.79991 |
| 0062 | 19 | 0.27887 | 0.68145 |
| 0062 | 15 | 0.27821 | 0.68252 |
| 0062 | 10 | 0.36375 | 0.67099 |
| 0097 | 19 | 0.31038 | 0.72751 |
| 0097 | 15 | 0.29355 | 0.70608 |
| 0097 | 10 | 0.46517 | 0.69223 |
| 0347 | 19 | 0.36930 | 0.71201 |
| 0347 | 15 | 0.35192 | 0.68383 |
| 0347 | 10 | 0.64514 | 0.78158 |
| 0400 | 19 | 0.42121 | 0.68529 |
| 0400 | 15 | 0.39972 | 0.69110 |
| 0400 | 10 | 0.37690 | 0.80205 |
| 0590 | 19 | 0.35296 | 0.67779 |
| 0590 | 15 | 0.35396 | 0.65779 |
| 0590 | 10 | 0.43270 | 0.74171 |
| Seven-scene macro | 19 | 0.33791 | 0.66988 |
| Seven-scene macro | 15 | 0.32941 | 0.65645 |
| Seven-scene macro | 10 | 0.42263 | 0.74094 |

The seven-scene macro is development diagnostics only and is not eligible as
the paper-eight row. The legal inventory is now 11/29 overall and 7/8 for
ScanNet; the remaining paper-eight scene must be completed before an
aggregate or SOTA comparison is made.

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

- LERF2D: the legal D512/L512 full-four cohort is complete; the current
  0.31417 sample-micro mIoU demonstrates a readout-quality gap, not a missing
  field or protocol gap.
- LERF3D: the shared full-four field cohort and frozen typed 3-D evaluation are
  complete; 0.33450 sample-micro mIoU demonstrates the same readout-quality
  gap, with Waldo Kitchen as the worst scene.
- ScanNet OVS: scene0000, scene0062, scene0070, scene0097, scene0347,
  scene0400, and scene0590 now have complete D512/L512 Method-v1 fields,
  primitive caches, and frozen single-scene results. The paper-eight cohort is
  7/8; only scene0140 needs its current source bundle/config rebuilt.
- NVOS: the public-compatible signed RGB/SAM transient adapter is implemented
  and audited; unified full-eight D512/L512 fields are not materialized.
- Available-Nine SPIn-NeRF: frozen protocol exists; unified D512/L512 fields
  must replace the historical carrier. Legal reference-only transient
  selection is now implemented and reproduces its sealed full-nine result.

No joint five-benchmark row is eligible yet. Existing historical peak numbers
must not be combined into a virtual incumbent.

## Reproducible artifacts

- Per-scene spatial field: `official_siglip2_spatial_w005_s0_64.pth`
- Per-scene spatial+region field:
  `official_siglip2_spatial_region_w005_s0_64.pth`
- Per-scene frozen-basis generic-response field:
  `generic_text_response_w005_s0_64.pth` (Figurines uses the repaired
  `generic_text_response_w005_s0_64_lineage.pth`)
- Joint-basis early-stop ablation:
  `generic_text_response_basis_w005_lr5e4_s0_64.pth`
- Source-only official spatial bundle:
  `canonical_teacher_features_v2/figurines_source_only_siglip2`
- Source-only genuine crop-summary bundle:
  `optimization_20260716/semantic_teacher_train`
- Frozen global region bridge:
  `global_region_summary_coco15000_full_context_local_scales_imageholdout.pth`
- Frozen Method-v1 primitive caches: `primitive_query_method_v1.pth` under
  each scene directory.
- Frozen LERF2D Method-v1 reports: `lerf2d_eval_method_v1_primitive` under
  each scene directory. The older `lerf2d_eval_method_v1` and Figurines
  `lerf2d_eval_method_v1_lineage` directories are dense-readout diagnostics.
- Frozen LERF3D Method-v1 reports: `lerf3d_eval_method_v1/<scene>/` under each
  scene directory.
- Frozen ScanNet Method-v1 reports: `scannet_vala_method_v1/` under each of
  `optimization_20260815/core_method_v1/scene0000_00/`, `scene0070_00/`,
  `scene0062_00/`, `scene0097_00/`, `scene0347_00/`, `scene0400_00/`, and
  `scene0590_00/` in the results root.

## Verification

- The focused Method-v1, primitive-cache, finetune, generic-response,
  semantic-alignment, and ScanNet external-primitive evaluator regression
  suites pass 38/38, including the Waldo one-row alignment, fail-closed larger
  mismatch cases, configured ScanNet frame-authority fallback, and the full
  LERF3D authority formula fixture.
- 104 audited LUDVIG wrapper/full-carrier and SPIn adapter tests pass.
- Python compilation and `git diff --check` pass.
- The frozen five-contract validator still reports zero eligible joint rows;
  it correctly refuses to stitch historical peak numbers into a virtual
  five-benchmark result.
