# Full-source text tracking and scope fitting, next79

## Mechanism gap and intervention

Native feature construction already consumes all293/122/169/180legal source
images, but retained LERF temporal evidence was produced by
`track_text_sources_next63.py` over `teachers.json.train_frames`: the old20/25
source images in the two pilot scenes. Retained extent fitting also uses its old
source cohort. These are distinct stages; all-source feature reconstruction did
not make temporal supervision or scope fitting all-source. This was not a LERF
protocol restriction.

Implement `track_full_text_sources_next79.py`: visit the complete authorized
source sequence from `source_coverage_next72/{scene}/all.json`, excluding target
images as before. Keep the SAME old source detections, eligible seeds, native
identity-margin check, winning seed view, model, tracking resolution and output
raster96x128. Only tracking sequence coverage changes. Query sharding distributes
work across GPUs; it does not limit scene sources. Default processes all queries.

After a query-index0 execution pilot, complete every query for figurines and
ramen, then extend to teatime and waldo_kitchen after positive pilot results. Index0 was chosen by index, not query outcome. Its outputs are kept as
pilot evidence, not a scene-level result to combine with winners elsewhere.

Two scope controls on the unchanged next75 semantic field:

1. **Old-fit views:** use new full-sequence predictions only on the original
   fitting views. Same240updates. This isolates the effect of sequence coverage
   on tracking predictions as far as this decoder permits.
2. **All-fit views:** use every legal source view. Original detection evidence
   remains available on old views; added views have temporal evidence only.
   Missing detections are empty observations with zero detector confidence,
   not invented masks or negative-label supervision. Views without a present
   track carry zero temporal confidence. The original seed, native conflict
   reference, visibility renderer, boundary abstention and objective are kept.
   Consume all eligible views: steps=max(240,number of source views), hence
   figurines293 versus its old240updates. This is extra evidence AND some extra
   compute, not an equal-compute claim. Ramen remains240updates.

This still uses the retained native-positive gate and old seed identity evidence;
more frames can consistently propagate a wrong seed/scope. No removal of those
constraints or new threshold tuning is silently included. Semantic2D IoU/AP
are frozen; measure the actual selected-Gaussian silhouette separately.

## Implementation and scope

- `scripts/research/track_full_text_sources_next79.py`: full-source tracking,
  optional explicit query pilot and non-overlapping worker sharding.
- `decode_expanded_extent_next72.py`: optional track source, explicit temporal-
  only extra views, and query subset for controlled pilots. Defaults unchanged.
- `run_full_tracks_decode_next79.py`: same frozen evaluator and output contract.
- Tracks: `output/full_text_tracks_next79/lerf_{scene}`.
- Evaluations: `output/full_text_decode_next79/{pilot0,complete}/{old,all}`.

All results are development evidence with additional authorized source-RGB
query-time compute. This is not a field-only, equal-budget SOTA comparison.
No independent3D-GT, new NVOS/ScanNet/UQIS, compact-student or same-map four-query
result is claimed. The initial two scenes are followed by the remaining two under the same recipe.

## Completed four-scene comparison

All67queries were processed with unchanged seed selection.54have valid seeds
and now contain293/122/169/180frame predictions for their respective scenes;
13have no eligible seed and remain explicitly untracked. This is not a claim
that all67queries acquired valid temporal supervision.

| Scene | Previous selected projection | Full sequence, old fitting views | Full sequence, all fitting views |
|---|---:|---:|---:|
| figurines | 59.432 | 59.535 | 61.135 |
| ramen | 44.018 | 44.017 | 44.433 |
| teatime | 58.333 | 58.296 | 60.159 |
| waldo_kitchen | 55.386 | 55.718 | 56.310 |
| **Macro** | **54.292** | **54.391** | **55.509** |

All four scenes improve selected projection with the SAME complete recipe.
Semantic2D IoU66.930/AP84.146 remains unchanged by construction. Most aggregate
gain appears only when the added views participate in fitting; the sequence-only
control changes macro selected projection by about0.10pp. Gains include porcelain
hand+25.17pp, coffee mug+8.92pp and yellow desk+8.04pp. There are query regressions:
bag−5.73pp, hooves−4.24pp and refrigerator−3.74pp. No per-query/scene stitching.

Machine-readable results: `output/next79/full_tracks_summary.json`; seed and
source counts: `output/next79/tracking_coverage.json`. The index0 pilots are
excluded from the four-scene comparison. The separate region-equivalence
propagation arm failed and is not included in this candidate.

## Remaining source-identity gap

Unseeded queries are figurines: pikachu, tesla door handle, waldo; ramen:
kamaboko, onion segments, sake cup; teatime: dall-e brand, paper napkin;
waldo_kitchen: ketchup, ottolenghi, plastic ladle, pour-over vessel, spatula.
These are source-admission observations, not GT-derived labels added to training.

A source-only diagnostic on available regenerated64-view detections for the
first two scenes still finds no admitted seed for any of their six missing
queries. Five have no detection proposals at all; pikachu has four candidates
but maximum foreground/context margin−0.00445, hence all fail the positive-margin
seed rule. This differentiates missing proposals from rejected proposals. It is
not an all-source grounding experiment, and does not prove whether those four
pikachu proposals are correct. See `seed_availability_diagnostic.json`.

Full video coverage cannot create missing seed identity or correct a consistently
wrong scope. Next work should obtain query-compatible candidates from the native
regional bank when grounding returns none, independently validate their identity
and whole/part extent, and revisit hard seed admission separately from proposal
absence. Do not merge these into one threshold-tuning problem. The regional bank
and typed relation prototypes are available, but such a candidate-selection
mechanism is not implemented by the current full-source tracking change.

The measured gain closes a real supervision-coverage gap, not a demonstrated new
CVPR-level method contribution. High-capacity native features, historical query-
specific calibration and compact four-teacher/four-query validation remain
unresolved aspects of the larger project. No new other-benchmark scores or SOTA
claim follows from these LERF selected-projection gains.

## Equal-step control and decision

Figurines full-sequence/old-view fitting at293updates yields59.591selected
projection, versus59.535at240updates and61.135for all-view293updates. The
all-view advantage persists at equal steps; it is not explained by the extra53
updates alone. This additional control is complete.

Retain the complete all-source temporal/all-view fitting recipe as the next
high-capacity LERF research candidate. All four scene averages improve, with no
per-query/scene winner assembly. Defaults and other-benchmark candidates remain
unchanged. Reproduction recipe: `configs/recipes/next79_research.json`. All jobs
launched for this iteration have completed: two graph pilots, two query-index0
pilots, eight full-scene control evaluations, and one equal-step control.
