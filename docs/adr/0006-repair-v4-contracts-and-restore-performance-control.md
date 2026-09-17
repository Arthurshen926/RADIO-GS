# Repair v4 contracts and restore the historical performance control

Status: Accepted for implementation (user authorization, 2026-09-14).

This amends the proposed sequencing in ADR 0005; historical ADRs and receipts
remain immutable. The v4 candidate is not the promoted performance baseline.

Restore the retained typed identity–extent baseline outside v4 method code.
Freeze inputs, renderer, query semantics and aggregation before comparison.
Allow fixed development end-to-end diagnostics alongside geometry gates;
never use diagnostic GT selections for deployment or final-set tuning.

New projection implementations have explicit, serialized contracts. Missing
version fields in historical bundles select the historical implementation,
never silently reinterpret old lifted evidence. Rebuild geometry/lifts for
new contracts. Pixel coordinates use image-edge coordinates: pixel (x,y)
has centre (x+.5,y+.5). Visible-scene rendering and selected-only rendering
remain distinct output definitions.

Overlapping candidate support is not categorical instance membership.
Canonical categorical composition requires explicit validated probabilities;
legacy normalization/top-2 remains available only as identified replay.
Preserve candidate evidence before query and separate compression loss from
unobserved mass. Object organization cannot replace local identity by fiat.

Nonoverlap and visible mask absence are not physical different-instance
labels. New relation paths require explicit negative evidence; frozen legacy
learned weights may only consume their original feature semantics in replay.
Correlated source masks must not accumulate as independent observations.

Completion and compression remain research options until real source-heldout
identity and extent diagnostics and matched end-to-end evaluation show value.
No benchmark improvement is inferred from unit-test success.
