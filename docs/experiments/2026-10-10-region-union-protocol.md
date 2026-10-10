# Same-view set union instead of part competition

Follow-up full four-scene/68-text LERF, same exported native-region fields/query
fixtures/source budget, no state or training change, temperature10 and strict0.5.
Acceptance is identical to native-region study. Copy admission/native_parts/
native_protected as sealed controls; predict union_parts/union_protected(primary).
All four scenes/all controls before target scoring. This remains development.

Within each view, accept the same regions with positive native margin and nonzero
available Bpositive mass. View confidence remains max margin. Replace convex
positive/negative mixing with max Bpositive and min Bnegative across ALL accepted
regions, both multiplied by the same view confidence. Empty view abstains;
a region unknown about a Gaussian cannot supply reliable union-negative evidence.
No candidate truncation. Chunk16 is only a temporary-memory bound.

This is a union of Gaussian partial marginals. It is NOT exact v10 source-raster
union: lifting and max/min need not commute, boundary erosion and visibility also
differ. Negative reliability and instance identity are not solved by set algebra.
Shared P/N once for protected/unprotected updates; no other-view self-protection.

Native vs union is an explicit change of within-view composition including removal
of relative part weighting, not an isolated threshold change. Compare actual-set
projection and response/AP across every scene. No per-scene route, GT selection,
threshold search, source access during queries, default/full-five claim. Retain all
failures. A two-part synthetic counterexample validates semantics, not accuracy.
