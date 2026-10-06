# Source-positive witnesses versus conflicting exclusions (next94)

## Decision

Completed all eight scene evaluations. Do not promote either witness policy:
next93 remains the optional LERF research candidate; next80/defaults remain unchanged.
Positive-witness protection does not repair the principal loss from incomplete scope.
This is a negative mechanism experiment, not a benchmark improvement.

## Mechanism and fixed conditions

The next93 leave-one-view-out negative vote can outvote a few positive views.
Here a source view witnesses a Gaussian only when eroded positive pixels account
for more than half of its full visible contribution footprint. A query/view must
have positive evidence and positive confidence. Count distinct source frames;
subtract the current frame. Protect a proposed extra negative pixel when more
than half its visible responsibility has at least K other-view witnesses.
Compare K=1 and K=2. Conflicting added negatives become unknown; no new positive
labels are fabricated, and existing baseline negatives are unchanged.

This deliberately isolates protection against next93's additional exclusions.
It does not test removing all historical negatives, nor certify complete masks.
Witnesses use source teacher masks, not verified identities; adjacent frames are
correlated. Witness calculation precedes the existing detector-only positive
admission stage, matching the position of the next93 exclusion stage.

Same geometry, native features, identity candidates, union tracks, eroded cores,
source initialization, optimizer budget and scorer. All authorized source frames:
293/122/169/180. Refit the same 13 regional queries; preserve other 54 columns.
Target GT is only consumed by evaluation and posthoc precision/recall analysis.
No per-query winner selection. This remains source-assisted transient scope
inference, not a newly learned generalizable compact feature field.

## Results

Selected Gaussian silhouette projection IoU (%), same frozen evaluation:

| Scene | next93 | K=1 | K=2 |
|---|---:|---:|---:|
| figurines | 67.065 | 67.033 | 67.021 |
| ramen | 54.234 | 54.219 | 54.224 |
| teatime | 62.098 | 62.090 | 62.095 |
| waldo_kitchen | 59.054 | 59.054 | 59.053 |
| Macro | 60.613 | 60.599 | 60.598 |

K=1 protects 1300/3341/78/177 proposed negative source pixels across the four
scenes; K=2 protects 928/1558/25/114. Counts precede the later detector admission
stage and include all query columns processed by evidence construction.

Waldo IoU only 78.22→78.36 for K=1, far short of the pre-exclusion 82.47.
Its mean recall rises about 82.8→83.1%, with slightly lower precision. Handle
51.82→50.73, sake cup 60.89→60.71. Kamaboko and pour-over are unchanged;
spatula remains 28.71 with precision30.1%/recall86.3%. Onion remains zero and
ladle near zero. Thus the rule is active, but does not resolve the major failures.

Both policies preserve semantic scores and the other54 selection columns exactly
in all four scenes. Semantic IoU66.930/AP84.146 therefore unchanged. These are
2D projections, not independent 3D-ground-truth results. No new ScanNet/NVOS/UQIS,
four-query, held-out or SOTA claim.

## Consequences for the method

Do not keep tuning vote counts. These two controls specifically reject the idea
that a simple source-positive veto is sufficient; they do not prove all possible
conflict reasoning ineffective. Remaining causes are still separable:

1. **Scope completeness:** when current positive masks do not cover a missing
   region, a positive veto cannot recover it. Next use genuinely new source-side
   object-completion evidence, preserving identity and whole/part distinctions.
   Measure coverage on views excluded from hypothesis construction. Waldo remains
   unresolved; do not declare its cause uniquely established by this experiment.
2. **Boundary representation:** spatula is insensitive to these exclusion rules.
   Prior replay attributes its background alpha to both extraneous and mixed
   primitives. Next isolate a source-only boundary-conditioned representation or
   responsibility split, retaining the frozen carrier control. Any carrier change
   must be explicit, query-independent and evaluated with its own semantic remap;
   it must not silently invalidate current geometry-bound selections.
3. **Identity:** onion/ladle failures cannot be repaired by voting over the same
   wrong or missing scope. Preserve independent candidate identity evidence.

The base multimodal architecture remains intact. These changes are optional
research decoder controls, not a substitute for eventual image/2D prompt/3D
prompt validation or generalizable map training.

## Reproduction

`run_scope_witness_next94.py --scene SCENE --gpu GPU --policies 1 2`
followed by `summarize_witness_next94.py` and `measure_witness_errors_next94.py`.
Decoder option `--positive-witnesses` defaults to0, preserving prior behavior.

Artifacts: output/next94/summary.json, error_decomposition.json,
retained_outputs.json; output/scope_witness_next94/{1,2}/lerf_*/fit.json,
typed_readout.pt and anchored/pipeline_calibrated/evaluation.json.
