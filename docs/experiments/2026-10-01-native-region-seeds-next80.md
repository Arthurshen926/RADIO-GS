# Native regional identity hypotheses for missing temporal seeds (next80)

The retained next79 reconstruction has 13/67 text queries without a temporal
seed. Adding views alone cannot fix an absent proposal. This experiment supplies
source-region candidates while retaining the frozen Gaussian carrier, native
semantic field, query encoders, score calibration and evaluator.

## Method

1. Preserve each automatic source region and its native SigLIP descriptor.
2. Use next79 mutual geometric region correspondence, requiring bidirectional
   coverage and a third source view, to admit paired scope hypotheses. Containment
   edges do not establish identity and are not used.
3. For a query without a next79 seed, maximize the minimum native cosine of the
   two corresponding regions. Seed SAM3 with the better-scoring member's mask.
   No evaluation masks, aliases or per-query score-selected policy is used.
4. Propagate over all legal source frames. Supply explicit native-pair confidence
   to the existing extent solver: previously its confidence came exclusively from
   the original detector, which has no evidence for these missing detections.
5. Fit only previously unseeded queries; retain the next79 element selections for
   all others. Semantic response scores are frozen.

The pair confidence is a nonnegative native cosine, **not** a probability of
correct identity. A single chosen scope is a bounded hypothesis and does not
solve exhaustive multi-instance text union. This is source-RGB-access transient
inference, not compact-feature-only retrieval.

Initial two-scene pilot improved selected projection in both scenes, so the same
recipe was extended to the remaining two scenes without changing its parameters.
A separate ablation, first run on two scenes then completed on all four, permits regional-track positives without requiring
old semantic positivity; exclusion evidence remains unchanged. This tests whether
the old semantic gate prevents valid new evidence, without conflating it with a
change of detector, tracker or reconstruction geometry.

## Source observations

`output/region_seeds_next80/source_crops.jpg` shows all 13 source candidates.
Waldo, the Tesla handle, the DALL-E label and the Ottolenghi book have plausible
source identity. The sake-cup candidate visibly covers a bottle; the pour-over
candidate includes a metal pan. Geometric consistency does not resolve semantic
identity. These visual observations did not alter candidate selection or the
experiment's per-query policy, and do not constitute target-label verification.

Candidate and scope supervision are separate: in the gated pilot, onion segments
still has no active fit view despite a candidate and full-sequence tracking.

## Reproduction

- `scripts/research/propose_region_seeds_next80.py`: pair-based seed proposals.
- `scripts/research/run_region_seeds_next80.py`: tracking, fitting, fixed evaluation.
- `scripts/research/expand_region_seeds_next80.py`: build missing banks/relations
  and run the same recipe for another scene.
- `scripts/research/summarize_region_seeds_next80.py`: all changed-query outcomes.
- `configs/recipes/next80_research.json`: experimental contract.

No unit tests, evaluation threshold sweeps, target-image supervision or changes to
NVOS/ScanNet/UQIS are included. Selected-Gaussian projection IoU is a 2D projection
metric, not an independent 3D ground-truth score.

## Completed results and decision

All eight scene evaluations completed under the frozen evaluator. Values below
are selected-Gaussian projection IoU (%). Each arm applies one fixed policy to
all four scenes; there is no per-query winner assembly.

| Scene | next79 | Regional seeds, old positive gate | Regional positives admitted |
|---|---:|---:|---:|
| figurines | 61.1348 | 62.3507 | 62.8787 |
| ramen | 44.4327 | 45.0132 | 44.9314 |
| teatime | 60.1592 | 62.1434 | 62.1447 |
| waldo_kitchen | 56.3098 | 56.9488 | 57.7357 |
| Macro | 55.5091 | 56.6140 | 56.9226 |

Retain the **regional-positive arm as a LERF research candidate**: all four scenes
improve over next79, with a macro gain of 1.4135 percentage points. Relative to
the gated new-seed arm, ramen slightly regresses; this is reported rather than
selecting different policies by scene. 2D semantic IoU remains 66.9301% and AP
84.1459%. This iteration improves set decoding, not semantic-field ranking.

| Changed query | next79 IoU | Candidate IoU | Active source views |
|---|---:|---:|---:|
| pikachu | 60.86 | 77.51 | 230 |
| tesla door handle | 19.12 | 21.84 | 127 |
| waldo | 10.90 | 30.53 | 124 |
| kamaboko | 11.42 | 13.50 | 117 |
| onion segments | 0.00 | 0.00 | 43 |
| sake cup | 16.52 | 19.99 | 86 |
| dall-e brand | 19.55 | 54.36 | 35 |
| paper napkin | 50.42 | 59.92 | 90 |
| ketchup | 21.76 | 40.93 | 22 |
| ottolenghi | 90.28 | 83.69 | 47 |
| plastic ladle | 0.00 | 0.00 | 33 |
| pour-over vessel | 33.22 | 52.01 | 35 |
| spatula | 11.17 | 11.17 | 28 |

All 13 previously seedless queries now have a tracking hypothesis and active
scope-fitting views in the regional-positive arm. That is coverage, not verified
identity or correctness. Onion segments and plastic ladle remain at zero; spatula
is unchanged. Ottolenghi regresses from 90.28 to 83.69 despite a plausible source
crop. Native recognition, physical extent and tracking integrity are distinct.

The ablation supplies evidence for a **circular-supervision limitation**: a new
identity provider cannot repair a missed concept if positive labels require an
old semantic response first. Removing that dependency improves Pikachu and
ketchup. It is not enough for all queries; admitting a track still does not
establish the correct semantic referent or whole/part extent. It is therefore
incorrect to describe the remaining gap as only a detector threshold problem.

The old source-based initialization also remains detector-derived; adding new
regional tracks does not initialize membership from their independently supported
scope. This is a concrete remaining mechanism to separate from candidate identity
in the next controlled experiment, rather than increasing training steps.

Next priority: separate wrong-reference tracks from correct-reference/incorrect
extent tracks using source evidence, and compare source-supported membership
initialization on the same candidates. Retain multiple disjoint hypotheses when
needed for text union; a single best region is still an incomplete query contract.
Do not claim same-map four-query completion, compact-student distillation,
independent LERF3D ground truth, or SOTA/CVPR readiness. NVOS/ScanNet/UQIS unchanged.

Machine-readable results: `output/next80/summary.json` and
`output/next80/positive_summary.json`. Candidate readouts:
`output/region_seed_positive_next80/<key>/typed_readout.pt`.
