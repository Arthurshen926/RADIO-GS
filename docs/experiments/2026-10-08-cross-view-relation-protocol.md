# Full-cohort source relation learning

Frozen before new fitting, 2026-10-08. This is substantive source training under
ADR0014, not another query rank/threshold sweep. Native construction losses and
typed category decoding remain separate unfinished work; this study does not
claim to implement the entire revised architecture at once.

## Fixed experiment

All30 existing fields: four LERF, seven distinct NVOS fields serving eight cases,
ten ScanNet, nine UQIS. Two new arms both warm-start the same physical checkpoint:

- `local_anchor_control`: render primary source partial masks from the original
  anchor, then a matched other-view soft partial target from its own interior anchor.
- `cross_view_anchor`: identical examples, second-view targets, scales, views,
  initialization and budget; condition both renders on the original 3D anchor.

Source mask regions are traversed without replacement for two epochs. Every
eligible region is presented exactly twice. Adam learning rate0.001, seed20261008,
two equally weighted actual subset-alpha losses per update. Both arms continue
training; neither is an equal-compute comparison to the old2000-step checkpoint.
The paired comparison isolates the support condition of the auxiliary loss;
the baseline-to-control effect also includes added budget and changed exposure.

Geometric contribution columns establish shared visibility; no column means no
evidence. In the other view, retain all eligible masks whose eroded interiors
contain >=0.5 of projected anchor mass and whose physical scale is within[.5,2]
of the primary scale. Normalize their support weights within that view. Combine
their partial positives and valid disjoint comparable-scale exclusions as soft
evidence; conflicts remain ambiguous soft supervision. Unseen pixels never become
negatives. Choose uniformly among supported views, not among all masks, so nested
regions do not manufacture independent observations. No supported second view
duplicates the original condition in both arms and is counted explicitly.

The local control's second anchor is the middle reliable interior pixel of the
most-supported compatible mask. It is itself uncertain source pseudo-supervision,
not an official complete-instance label. Geometric registration and masks can
still be wrong; this is not proof that pseudo instances are complete or correct.

Training uses only the parent's compiled training split, hash-bound to the same
geometry and native observations. No new teacher, view, development mask, benchmark
query or3D instance label. Only grouping and its existing scale/affinity parameters
are updated. Native/geometry/support buffers must be exactly unchanged. Standard
runtime remains the same source-free field and member operator; no region memory,
track cache or labelled v10 member model is exported.

## Complete evaluation and interpretation

All four LERF scenes (both retained readouts), eight NVOS cases, ten ScanNet rooms
and all268 UQIS inputs are predicted for both arms before opening new scores.
Use the unchanged source-free adapters, current typed readouts, thresholds,
default scales and query candidate rules. Do not fold in the previous image
reranking controls or native category bypass. Preserve empty/unsupported cases.

Report baseline/control/cross-view separately, and the matched-arm difference.
Require paired data/condition traces to agree, source region exposure to be complete,
protected buffers to match, and standalone source-denied replay. Actual sets,
responses and AP remain distinct. Report all scenes/modalities and paired scene
uncertainty; any regression prevents a blanket full-performance claim. No per-scene
or per-input checkpoint choice. The unchanged native-only control must reproduce.

Next decisions follow the full table, including opposite LERF over-selection and
NVOS under-selection errors. An improvement does not eliminate native evidence
loss or contract gaps to v10; a failure localizes this particular partial-relation
training design rather than refuting cross-view evidence or the single-field idea.
