# Next67: let subset-render evidence update elements hidden in the full scene

## Main result

On the same expanded-RGB next66 carrier, source masks, semantic field and 240-step optimization budget, removing the full-scene-visibility freeze improves selected-only projection IoU **45.862 → 51.414**, with every scene improving. The retained pre-reconstruction candidate was 48.518. Semantic IoU/AP/localization stay **64.882/81.021/87.328**, unchanged by construction.

This repairs an impactful mismatch between the membership optimization domain and the renderer consuming that membership. It does not establish complete identity/whole-part supervision, physical 3D accuracy, or SOTA. Teatime is still worse than the old-carrier candidate. All results use the existing four LERF development scenes; no independent test or other-benchmark result was added.

## Mechanism

The old decoder chose its trainable membership domain using

```
support = observed & (sum(full_scene_visible_contribution_over_active_views) > 0)
```

It froze membership outside that domain to the semantic-score initializer. However, deployment removes unselected Gaussians and renders the selected subset alone. Full-scene transmittance is then different: an element that was invisible behind another element can contribute to the selected subset. The original optimization made some such memberships impossible to correct, even when subset-render losses contained useful evidence against them. A larger/rebuilt map made this mismatch costly.

The new `exposed` control permits updates to all semantic-observed elements:

```
support = observed
```

It still learns only through the existing known source pixels and subset renderer. An element receiving no subset-render gradient is not automatically made a positive or a negative. Unknown pixel labels, source mask confidence, native qualification and query eligibility stay unchanged. Geometry, opacity, feature scores and the deployment scorer stay fixed. This is an optimization-domain correction, not an extra teacher, postprocessing threshold, GT-driven erasure or a change to visibility used for semantic feature fusion.

The positive/negative labels remain pseudo-labels. Removing this freeze does not make them independently verified identities or complete object masks, and inaccurate source labels can still cause deletion errors.

## Controlled results

All controls start from the same new-carrier semantic scores and receive 240 updates/query at learning rate .05. Source targets are the same detector/temporal union with the existing native qualifier. Membership relaxation anneals from .5 to .1; deployment remains membership>.5 and selected-only alpha bytes>10.

| Control | Selected projection IoU (%) |
|---|---:|
| Next66 matched parent | 45.862 |
| Hard membership forward, same restricted update domain | 45.870 |
| Allow subset evidence to update all observed elements (`exposed`) | **51.414** |
| Hard membership forward + exposed support | 51.509 |
| Parent with 4× linear source-render resolution | 45.833 |
| Exposed support with 4× linear source-render resolution | 51.220 |

The hard-forward arm uses a straight-through soft membership derivative through gsplat. It is a surrogate gradient, not an exact discrete optimizer; rasterizer culling can also affect its gradients. Its negligible isolated gain does not establish that every possible hard-set optimizer is ineffective. We retain the simpler exposed-support recipe, not the numerical maximum from an added near-neutral mechanism.

The resolution controls use 384×512 rather than 96×128 source renders and scale camera intrinsics correspondingly. Targets and known masks are nearest-neighbor expansions of the same labels. They add no teacher detail or new supervision. These controls do not support spending more render compute on those same labels. They do not test rebuilding teachers at native resolution.

| Scene | Old-carrier next63 | Next66 matched parent | Exposed support |
|---|---:|---:|---:|
| figurines | 49.095 | 44.709 | **56.185** |
| ramen | 35.275 | 39.161 | **41.386** |
| teatime | **59.116** | 48.262 | 55.283 |
| waldo_kitchen | 50.584 | 51.317 | **52.800** |
| Scene macro | 48.518 | 45.862 | **51.414** |

Against the matched current parent, changes are +11.476/+2.225/+7.021/+1.483 percentage points. Against old-carrier next63, three scenes improve and teatime remains below by 3.833 points. No per-query or per-scene winner stitching was used.

## What the improvement removes

Per-annotated-query macro precision/recall, measured after training using the existing masks and threshold:

| Scene | Precision before → after (%) | Recall before → after (%) |
|---|---:|---:|
| figurines | 47.20 → 59.74 | 93.34 → 92.92 |
| ramen | 42.18 → 44.55 | 76.84 → 76.80 |
| teatime | 50.29 → 57.85 | 91.86 → 91.87 |
| waldo_kitchen | 55.06 → 56.67 | 74.19 → 74.18 |

The improvement mostly removes false positives with limited recall change. These statistics are diagnostic only; annotation masks did not enter training or candidate construction. Together with the single-factor intervention they support the causal relevance of the restricted update domain, without attributing every remaining error to it.

[Diagnostic mask examples](../../output/subset_consistency_next67/selected_examples.png) show the largest improvement and largest regression per scene, explicitly selected after evaluation for explanation. For example, one figurines pink-ice-cream view changes20.9→89.7 by removing disconnected wrong objects; a ramen chopsticks view36.7→83.9 removes a distant false positive; a teatime coffee-mug view18.2→71.5 removes broad spurious fragments. These are individual views, not query-average gains. The plot also shows an old-camera regression and the still-failed knife case. No cherry-picked view score is used as the aggregate result.

## Candidate and research implications

The current LERF research candidate is **expanded-source joint RGB+MoGe reconstruction → rebuilt native semantic field → source scope inference → exposed-support subset decoding**. Response and membership bind the same new carrier. Its semantic/selected outputs are64.882/51.414, compared with58.087/48.518 before full reconstruction. The semantic improvement belongs to next66; next67 preserves it and repairs much of the selection regression.

This candidate uses the expanded source RGB budget, native1536D SigLIP feature representation, old confidence-distribution transport and query-time source supervision. It is not a compact four-teacher field or an equal-input/map-only comparison to other methods. Selected projection remains a 2D score of a selected3D set, not independent3D-GT IoU. No global cross-benchmark default was changed.

An implementation correction alone is insufficient CVPR novelty. The more general research question is how a shared semantic map preserves native retrieval while producing explicit object membership under the *changed visibility induced by selection*. This experiment gives evidence that those two consumed outputs require different optimization domains. A publishable claim still needs transferable identity/extent supervision, same-input comparisons and broader query/dataset results.

Remaining priorities:

1. Distinguish incorrect entity/whole-part source labels from unseen extent. The support fix cannot recover a query whose eligible source positives are missing/wrong; merely labeling unknown complements negative was already ineffective.
2. Resolve remaining new-carrier failures, including teatime versus the old carrier and waldo's MoGe disagreement, without discarding the semantic gains or silently using two different maps.
3. Apply the same explicit response/membership contract to image and prompt tasks and verify on the other accessible benchmarks before a universal-method claim. No such result is asserted here.

## Reproduction

```
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_subset_consistency_next67.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/run_raster_consistency_next67.py
/root/miniconda3/envs/cybersim_agent/bin/python scripts/research/summarize_subset_consistency_next67.py
PYTHONPATH=. /root/miniconda3/envs/cybersim_agent/bin/python scripts/research/visualize_subset_consistency_next67.py
```

The simpler recipe alone can be rebuilt with `run_subset_consistency_next67.py --modes exposed`. Outputs: `output/subset_consistency_next67/exposed/lerf_<scene>/typed_readout.pt`. Geometry is explicitly bound to next66 adaptive_allrgb; semantic scores are frozen from that recipe.

All 20 scene-level fits and 20 corresponding evaluations completed successfully. `summary.json`, `error_breakdown.json` and per-scene `fit.json` retain the numerical results. Validation was actual reconstruction/benchmark execution; no defensive test suite or additional audit framework was added.
