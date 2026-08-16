# LERF typed extent posterior design — 2026-08-16

## Decision

LERF2D peak-connected extent and LERF3D quarter-retention peak extent are two
profiles of one query-peak-anchored conservative extent operator. They share a
formal interface and implementation, but the already-observed results are not
one exact-parameter method: the current dense-raster profile uses retention
floor `rho=0`, while projected primitive alpha uses `rho=0.25`.

This operator belongs after the Gaussian Query Posterior has been converted to
a raster support. It is therefore a typed extent-posterior/output-conversion
stage, not a replacement for the Gaussian-domain Typed Posterior Interface.

## Formal interface

Let `S` be the frozen thresholded raster support and `h` the query response map.
Let `a=argmax(h)` after the frozen coordinate mapping. If `a` lies outside
`S`, snap it to the nearest foreground pixel. Let `C_a(S)` be the 8-connected
component containing that anchor and

```text
r = |C_a(S)| / |S|.
```

For a globally frozen retention floor `rho`, return

```text
E_rho(S,h) = C_a(S), if r >= rho
             S,        otherwise.
```

The implementation is
`radio_gs.querying.typed_extent_posterior.apply_peak_anchored_extent` and emits
an auditable receipt containing the domain type, peak, support sizes, retained
fraction, decision, and negative declarations for RGB, ground truth, and
persistent-state mutation.

The two input adapters are:

- `dense_raster`: `S` is the frozen LERF2D probability threshold and `h` is
  the same dense query posterior.
- `projected_primitive_alpha`: `S` is selected-only-alpha after the frozen
  Gaussian decision and silhouette rule; `h` is the separately rendered,
  target-blind primitive query response.

No scene or query identifier is an operator input.

## Useful invariants

- Subset-only: `E_rho(S,h)` never invents foreground outside `S`.
- Identity fallback: an uncertain, low-retention peak component returns `S`
  exactly.
- Anchor preservation: an accepted component contains the score anchor or its
  frozen nearest-support projection.
- Idempotence: applying the same operator twice returns the same support.
- Bounded deletion: when `rho=0.25`, an accepted refinement retains at least a
  quarter of the frozen support.
- Query-transient: the result cannot update Universal Field v1 or another
  query.

These are structural guarantees, not an IoU guarantee. Connectivity is a
reasonable object-extent prior because disconnected high-score islands are
often false positives, while the fallback protects multi-part projections
against a weak or misplaced peak.

## Current evidence boundary

- LERF2D dense `rho=0` improved all four scenes in the existing CPU screen,
  but it is render-then-score mechanism evidence; the exact primitive-score
  mainline follow-up is pending.
- LERF3D projected-alpha `rho=0.25` has an exact retrospective replay estimate
  of `0.48451` sample-micro mIoU and `0.46538` scene-macro mIoU. All four
  scenes improve over fixed extent in that replay, including Waldo.
- All current LERF full-four labels or metrics have already been observed.
  Neither profile can now become blind confirmation on the same cohort.
- The LERF3D evidence remains single-level. It does not satisfy the frozen
  three-semantic-level contract.

## Failure modes

1. A legitimate multi-instance or articulated object may occupy disconnected
   components; peak selection can discard true support.
2. A wrong semantic peak can select a compact distractor. Retention fallback
   limits deletion but cannot identify the correct component.
3. Resize rounding or tied maxima can move the anchor across a component
   boundary; the coordinate map and tie behavior must be frozen.
4. Thin structures can be disconnected by thresholding before the operator.
5. A connected false-positive bridge makes unrelated islands one component;
   connectivity alone cannot remove it.
6. An empty support remains empty and cannot recover a missed object.
7. Selected-only alpha can split a single 3D object through occlusion, making
   projected connectivity view dependent.
8. Choosing `rho`, connectivity, or snap behavior after target metrics is
   target tuning even though inference itself is GT-free.

## Recommended blind preregistration

The next genuinely blind test must use a cohort whose masks and metrics have
not participated in this work. Freeze before opening it:

1. One shared operator identity:
   `peak_anchored_conservative_extent_v1`, 8-connectivity,
   nearest-foreground snapping, `rho=0.25` for both raster domains.
2. Exact upstream identities: the four Universal Field hashes, query-cache
   hashes, text/canonical embedding hashes, renderer hashes, thresholds, and
   projection rules. For new scenes, freeze the same roles before scoring.
3. LERF2D primary input must be the exact primitive-score Method-v1 output,
   not the existing dense render-then-score diagnostic.
4. LERF3D primary input must satisfy the chosen Evaluation Contract. If the
   three-level contract remains authoritative, first produce its valid frozen
   three-level posterior; do not promote this single-level diagnostic.
5. Run prediction-only for the entire cohort, write per-query operator
   receipts and prediction SHA-256 values, and seal the complete batch before
   any target mask or metric is opened.
6. Permit no parameter callback or scene fallback after scoring. Report every
   scene, sample-micro and scene-macro metrics, identity-baseline deltas, empty
   support counts, fallback rate, and retained-fraction distribution.
7. Treat success on only one output domain as partial evidence, not validation
   of the shared operator family.

If no untouched benchmark cohort is available, use source-only registered
views or a separately designated development cohort to choose policy, and
report the current full-four reruns only as reproducible Development Evidence.
