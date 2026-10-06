# Complete-source cold-category system

This stage closes the missing ScanNet source-assisted scenes and replaces the
mixed source budgets with one explicit all-declared-source policy. It preserves
the current semantic map, native query coordinates and frozen geometry. It does
not introduce a learned compact student, claim public SOTA, or infer independent
generalization from these development scenes.

## Fixed mechanism and inputs

All ten scenes use the existing native1536 maps reconstructed from1158 source
images, supplied unlabelled mesh geometry, and the same19 class queries. The
cold prior is the exact one-vs-rest log-odds of temperature100 native cosine
scores, computed over the entire declared mutually exclusive vocabulary. No
historical scene threshold, competition head, semantic answer or test label is
loaded by inference.

Each declared source image receives literal-query SAM3 inference at the
existing1008 model resolution and96x128 scope raster. The installed official
SAM3 processor applies sigmoid before returning its historically named
`masks_logits`: these are probability masks, not logits to sigmoid again.
Native SigLIP foreground-minus-complement cosine uses the existing same-window
1.5x crop and384 resize. Its confidence remains a heuristic identity cue, not
independent correctness truth. Geometry contributions are freshly rasterized
from the same frozen map and explicit source cameras.

All nonempty detections are retained. Sharding partitions computation, never
selects source frames. The two previously missing scenes contribute46 and226
views; the other eight contribute886. Query-free region proposals are not
substituted for query-conditioned category observations.

## Complete-view scope inversion

The existing robust source-set solver is separable in query columns. The new
`source_query_batch_size=4` declaration bounds its transient memory while
retaining every source view. Category competition is computed before slicing;
recomputing softmax within each four-class batch would be incorrect and is not
performed. The former19-column implementation exhausted24GB on the226-view
scene0645 run; it was not recovered by dropping views.

An unbatched scene0200 prediction was retained for the actual batched rerun
comparison. This is a numerical execution comparison, not a new unit-test suite
or mechanism claim.

## Comparisons

1. Full10 map-only versus source-assisted: identical geometry/native maps.
2. Matched8 old176-source evidence versus all886 sources, using the same newly
   computed detector/identity tensors. This isolates source-set availability
   from inherited cache/teacher changes. Neither arm selects views using GT.
3. The full deployment recipe keeps the existing LERF4 and NVOS8 branches and
   replaces only the ScanNet adapter with the complete cold-category track.

The unchanged point evaluator saves predictions before reading labels, maps
GT vertices to nearest Gaussian centres, includes unsupported predictions as
void, ignores classes absent from GT, and macro-averages scenes.19/15/10 class
splits remain separate. These are the repository's frozen point metrics, not
verified equivalence to every public ScanNet-OVS protocol. LERF selected-set
projection likewise remains a2D evaluation of selected3D elements, not3D GT.

## Execution

```bash
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/run_scannet_cold_completion.py
PYTHONPATH=.:scripts/research /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/complete_scannet_all_source_cohort.py
PYTHONPATH=.:scripts/research /root/miniconda3/envs/cybersim_agent/bin/python \
  scripts/research/run_scannet_source_budget_control.py
```

The first command builds the two missing scenes; the second explicitly reuses
those tensors and builds the remaining eight. It exports
`configs/pipeline/system_complete_sources.json`, which replays inference and
unchanged evaluators using declared teacher caches. It is not a script that
reconstructs all geometric/native-map prerequisites from raw data.

## Measured information-budget effect

All10 completed with iterative scope inversion: point mIoU19/15/10 is
56.908217/58.074624/69.598625%, versus the identical map-only prior's
46.503650/47.976976/57.906700%. Previously missing source-assisted scenes now
complete: scene0200 reaches58.112391/58.159654/64.456933%, scene0645 reaches
52.614129/57.683020/69.200023%.

The matched8 fresh-teacher control isolates the source-frame set:

| Evidence budget, same new tensors and solver | 19 | 15 | 10 |
|---|---:|---:|---:|
| Frozen old176-frame subset |55.701539|56.515723|68.433510|
| All886 declared frames |57.294456|58.112946|70.291162|
| Difference, percentage points |+1.592917|+1.597223|+1.857651|

The old cached176-frame result55.705149/56.518796/68.437555 is close but not
identical to the fresh control; do not mix the two when attributing the source
budget effect. Fresh geometry rasterization and teacher execution are explicit.

The scene0200 batching comparison changes no19-class element prediction and
no19/15/10 point metric. Native logit maximum difference is0; final logit maximum
difference is0.000343323 from floating-point sparse operations. Four-column
scope inference is mathematically separable, not bitwise identical.

## A simpler category mechanism beats iterative scope inversion

Full10, same source tensors, same frozen geometry, native prior and confidence:

| Category source readout | 19 | 15 | 10 |
|---|---:|---:|---:|
|60-step robust inverse fit + soft cross-view consistency |56.908217|58.074624|69.598625|
|One-pass visible weighted fusion |57.921506|58.972766|70.905986|
|Difference, percentage points |+1.013290|+0.898141|+1.307361|

The simple method improves19-class IoU in all10 scenes. This is not universal
per-split improvement: scene0140's15-class metric drops0.524 points. On the old
matching8 scenes it reaches58.176196/58.832373/71.229359%, exceeding retainedv1
55.302891/55.822974/69.190160 by2.873305/3.009399/2.039199 points. That aggregate
includes both source-budget and inference changes; it is not a pure method gain.

For element i and category c, let A_v be the frozen visibility-normalized
projection, K_v the existing confidence-weighted known mask, and p0 the native
category prior. The simple readout is

`p = [sum_v A_v.T(K_v * target_v) + 0.1 * V * p0] /
     [sum_v A_v.T K_v + 0.1 * V]`,

where `V = sum_v A_v.T visible_v`. Unsupported entries preserve the exact native
logits. Supported probabilities become logits before the frozen evaluator's
class competition. There is no class-specific tuned parameter, per-scene winner
selection, new teacher or extra source RGB in this comparison.

This result rejects the assumption that the current iterative inverse-mask fit
is necessary for category partitions. It does **not** separately assign blame
to iteration versus the soft consistency weighting: both are absent in the
simple control. Ill-conditioned inversion and noisy masks are plausible
explanations, not proven individual causal attributions. Nor does it justify
replacing LERF whole/part selected-set inference or reference-instance tracking.

A weighted average is a baseline, not a novel CVPR contribution. The useful
research result is eliminating a more expensive, worse default and identifying
the stronger baseline a future mechanism must exceed with the same evidence.

Outputs: `output/scannet_cold_completion/full10_summary.json`,
`output/scannet_source_budget_control/summary.json`,
`output/scannet_source_vote/summary.json`. All were independently generated by
fresh query inference; the vote baseline never consumes the fitted solver's
answers. `source_category_fusion=visibility_weighted_vote` implements this
category-specific rule in the shared public runtime. `system_v2.json` combines
it with the retained LERF/NVOS adapters; its fresh22-job run is reported below.

## Public system promotion and full validation

`output/system_v2/summary.json` contains22 completed fresh deployment/evaluation
jobs: LERF4, NVOS8 and ScanNet10. Native/query readouts and source fitting are
rerun, using explicitly declared teacher caches; no historical final-answer
replay is used. The public ScanNet metrics match the independent one-pass
implementation exactly for every scene and each split (maximum metric difference0).

| Frozen local evaluation | v2 result |
|---|---:|
|LERF4 semantic IoU / AP |70.400676 /87.619278%|
|LERF4 actual selected-set projection IoU |66.961588%|
|NVOS8 positive-reference IoU / AP |86.513417 /98.022243%|
|ScanNet10 point mIoU19/15/10 |57.921506 /58.972766 /70.905986%|

LERF/NVOS tiny repeat differences are not gains. NVOS's retained reference path
uses frozen geometry and source tracks; those results do not demonstrate a
native semantic-field contribution to that benchmark. The four-query field
execution probes and the NVOS tracked-reference evaluation are distinct evidence.

Measured cached-evidence query time falls228.62→43.48 seconds on scene0140 and
292.03→46.28 on scene0645. These include fresh query encoding and evidence reading,
exclude source SAM/identity generation, and were run concurrently; report them
as observed system times, not a controlled hardware benchmark. All GPU jobs ended.

`configs/pipeline/system_v2.json` is promoted as the current executable recipe;
`system_v1.json` remains the frozen comparator. Promotion uses the complete
cohort, with a single ScanNet rule; no per-scene winner assembly. The input track
remains supplied unlabelled mesh, not RGB+MoGe.

A class-level diagnostic (mean only over scenes where the class exists) still
finds shower curtain0.5%(one scene), picture28.0%, counter29.1%, cabinet29.7%,
desk30.8%, table35.4%. These are not headline scene-macro metrics. They establish
remaining errors, not their unique cause; source recognition, extent coverage
and lifting still need to be separated before designing the next mechanism.

This stage completes system-level source coverage and removes an inferior
category inference default. It does not solve whole/part semantic mixing,
independent3D extraction, annotated same-map four-query accuracy, compact
multi-teacher training, new-scene generalization or public SOTA equivalence.
The next scientific baseline is the cheaper one-pass method, not the weaker
iterative solver. Beating only the latter would no longer support a contribution.
