# Text union after restoring candidate coverage (next89)

## Fixed method and completed comparison

Apply next86's source-only secondary hypothesis rule to next88's soft geometric
proposal pool. Keep a second pair only when both views pass the unchanged Qwen
identity rule and intersection/min with the primary support is <=0.2. At most
two hypotheses/query; all authorized source frames are tracked. This changes
six queries. All four scenes were evaluated with the frozen protocol; no
query-wise GT winner assembly or manual acceptance was used.

| Scene | next88 single scope | Two-hypothesis union |
|---|---:|---:|
| figurines | 67.225 | 67.225 |
| ramen | 49.552 | 48.037 |
| teatime | 62.034 | 62.034 |
| waldo_kitchen | 58.093 | 56.881 |
| Macro | 59.226 | 58.544 |

All values are selected Gaussian 2D silhouette projection IoU percentages, not
independent3DGT scores. Reject the union replacement. Retain next88's59.226 as an
optional higher-mean LERF research candidate; next80 remains stable reference.

## What this establishes

Spatula improves11.17→28.70 when the red-spatula source scope, recovered by the
relaxed candidate proposal stage, is admitted alongside the wooden-spatula scope.
Old-pool union next86 never changed11.17. This is a concrete coupled failure:
the old candidate pool excluded useful scope evidence, and one winning scope
still omitted it after the pool was repaired. It does not establish exhaustive
instance coverage or annotation correctness for every admitted source mask.

The same fixed union rule hurts sake cup38.82→20.89, ketchup49.92→17.80 and
pour-over65.72→53.66; onion remains0 and plastic ladle remains0.02. Source crops
show false secondary identities, including a bottle/background region for sake
cup and cookware for pour-over. Correctly fitting a false secondary still creates
false-positive elements. The global result is58.544, below59.226.

The previously defined joint comparison (first/second/both/neither/unknown,
BOTH required across two views with reversed order) rejects all six secondary
hypotheses, including the red-spatula complement. No reconstruction was run for
this all-fallback policy: it equals next88 by construction. The single-region
and joint judgments are not reliable calibrated identity probabilities; a more
restrictive prompt cannot be claimed to solve identity admission.

## Research conclusion and next priority

Across next86–89, six complete four-scene variants (24scene evaluations) distinguish
candidate coverage, multi-view choice, and membership fusion. Neither stronger
spatial deduplication nor early/late union nor whole-view unknown censoring solves
identity. Soft geometric proposal association is the only small aggregate gain
(59.078→59.226), not a major improvement or SOTA result.

The next method work should preserve alternative source views within each scope,
retain multiple potentially matching scopes for text-union queries, and establish
identity/whole-part compatibility before allowing each scope to expand the set.
Do not transplant text-union behavior into reference-instance image/prompt paths.
Evaluate source identity discrimination and actual selected elements together;
more tracking, decoder steps or confidence transformations over wrong admitted
objects are not supported by these controls. A dedicated, source-only contrastive
identity evaluation must distinguish correct alternatives, similar distractors,
whole/part conflicts and unknown evidence. Do not use target annotations to
select individual scopes or assemble per-query best outputs.

The four-query compact-map deployment and NVOS/ScanNet/UQIS results remain
unchanged. These variants still use optional query-time source teachers and
transient inference. Native semantic response IoU/AP remain66.930/84.146.

## Artifacts and reproduction

- output/next89/summary.json: complete four-scene result.
- output/next89/source_hypotheses.jpg: accepted independent-judgment candidates.
- output/next89/lerf_*/joint_identity.json: joint judgments, all abstaining.
- output/multi_scope_next89/union/lerf_*/typed_readout.pt: actual predictions.

Run multi_scope_next86.py with --seed-base output/grounded_seeds_next88,
--primary-tracks output/grounded_tracks_next88,
--baseline-base output/grounded_scope_next88/source, --iteration next89,
--policies union, and an explicit --scene/--gpu. Each previous script keeps its
original defaults; new path arguments isolate these experiments.
