# Multiple source hypotheses and candidate coverage (next86)

## Question and fixed conditions

Does a single source seed miss text-union instances, and does mixing their
supervision cause incorrect exclusion? Freeze next85's source candidate bank,
identity judgments, native field, geometry, decoder and evaluator. For each of
13 regional queries, retain the primary seed and admit at most one other pair
with positive identity margins in both views and support intersection divided
by the smaller support <=0.2. This overlap rule excludes strong containment as
well as duplicate coverage; it does not certify different physical instances.
All authorized source frames remain available. The two-hypothesis limit is an
explicit bounded mechanism experiment, not exhaustive retrieval.

Five queries admit a second hypothesis: sake cup, ketchup, plastic ladle,
pour-over vessel, spatula. Eight retain the primary evidence unchanged. Track every new
hypothesis through all source views with the same SAM3 provider.

Three comparisons:

1. Union masks before fitting one element membership.
2. Same union, but negative evidence is unknown whenever either hypothesis is
   missing. This conservatively censors an entire source view's negative mask;
   it is not a spatial visibility model.
3. Decode each membership separately and take their maximum afterward. This
   produces exact hard-set union at the fixed 0.5 membership threshold and uses
   additional solver compute. Semantic scores remain unchanged.

Seed frames do not count as independent tracked observations. Early-union arms
retain the primary native confidence, so the comparison of unknown handling
uses identical weighting. The factorized arm retains each seed's own confidence.
Source masks, candidates, confidence and admission never use evaluation labels.

## Completed result

All 12 scene evaluations completed. Selected-Gaussian silhouette projection IoU
is a 2D projection measure, not independently annotated 3D IoU.

| Method | Four-scene macro (%) |
|---|---:|
| Retained research candidate next85 | 59.078 |
| Early union | 58.493 |
| Early union with missing-hypothesis unknown | 58.175 |
| Independently decoded set union | 58.404 |

No method is promoted. Figurines and teatime have no admitted secondary and
retain identical scored projections. Early union reduces sake cup 38.82→28.55 and ketchup
46.85→15.39. Missing-hypothesis unknown is not a successful substitute for correct
identity. Factorization also fails to repair wrong identities. Plastic ladle
remains zero and spatula remains11.17. Query-level details are in the summary.

## Source identity and coverage findings

The source-only grid shows a glass admitted alongside sake cup, and a narrow-neck
bottle admitted alongside pour-over vessel. Independent binary-style identity
acceptance is insufficient for arbitrary union. These observations did not
manually alter any predictions.

A supplementary joint Qwen comparison showed both candidate crops, asked whether
first/second/both/neither/unknown matched, and required BOTH across two source
view comparisons with reversed candidate order. All five secondaries were
rejected. No scene reconstruction was needed for this rule: it retains next85
by construction. This is abstention, not a measured gain or independent proof
that every rejection is correct. No per-query winner assembly is used.

Spatula coverage was checked using its scored camera, AFTER predictions were
fixed and solely for diagnosis. All16 shortlisted candidate supports have zero
Gaussian centres inside this camera. The full source bank has13009regions, of
which1246have centres inside the camera; the equivalence graph has2880pairs,
of which222cover the camera. The first such pair has zero-based native rank38.
These are centre-coverage counts, not occlusion visibility, object correctness or
oracle IoU. They localize a shortlist coverage loss without proving that the
excluded candidates are correct spatulas. Neither camera-conditioned selection
nor evaluation RGB/GT is introduced into the method.

## Decision and next experiment

Keep next85 as the higher-mean research candidate and next80 as stable reference.
Do not further blend the same accepted false identities. A same-budget source-only
candidate diversity experiment (next87) will replace Jaccard>0.8 near-duplicate
suppression with intersection/min>0.2 exclusion, reducing repeated/contained
scopes' occupation of the16pair slots. It retains original native ranking and
the same verifier/decoder, changes no source frame cap and never uses the
camera-coverage diagnostic to choose candidates. This remains a hypothesis,
not proof of a general multi-instance solution.

## Artifacts

- output/next86/summary.json: all three full-scene comparisons.
- output/next86/source_hypotheses.jpg: primary and secondary source crops.
- output/next86/lerf_*/joint_identity.json: complete joint-comparison prompts/logits.
- output/next86/candidate_coverage.json and pool_coverage.json: coverage diagnosis.
- output/next86/retained_outputs.json: semantic and unintervened selection equality.

Semantic scores and the untouched54query columns are bit-identical. Recomputed
non-secondary memberships differ by at most0.000156, with identical hard
selection decisions. No new NVOS/ScanNet/UQIS results, compact-student improvement
or four-query claim.
