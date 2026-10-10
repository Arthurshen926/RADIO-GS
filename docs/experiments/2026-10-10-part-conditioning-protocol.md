# Query-conditioned parts and independent-view exclusion protection

Exploratory full four-scene/68-text LERF study after union_admission_v1. Same
exported fields, native query vectors and semantic-prior scores. Existing benchmark
labels were seen in prior development; no current target labels enter prediction.
No new source frames, descriptors, masks, training, labels or state. All four
scene predictions (all controls) sealed before retained benchmark scoring.

One transferable partial-evidence operator has two factors:
1. Regional query acceptance: for Bpositive column r, average the current query's
   semantic foreground score over available Gaussian mass. Accept a strict
   positive margin a=max(2*mean-1,0). Within each source view normalize these
   preferences and retain max(a) as view confidence. No scale or top-k gate.
2. Lift Bpositive/Bnegative with those coefficients into per-Gaussian, per-view
   positive P and negative N. The control update is (prior+sum P)/(1+sum P+sum N).
   Primary protection uses N_v*(1-other_positive/other_known), excluding view v
   from both other sums. Zero other evidence gives no protection. Unknown source
   coordinates contribute neither positive nor negative evidence.

Controls: admission (exact previously verified semantic admission/rank baseline),
parts (region conditioning, no protection), protected_parts (same regional
acceptance with leave-one-view-out positive protection). Fixed .5 admission and
existing scorer; no threshold search, per-scene routing, iterative expansion or
GT-selected candidates. Report all controls, response and actual-set projection,
AP and all scene results. Compare primary vs parts to isolate protection, and
parts vs baseline as a coupled changed acceptance/readout intervention.

This migrates a limited responsibility of v10's other-view-positive exclusion
censoring. It is NOT exact v10 raster-space protection: field B coordinates are
partial, accepted source parts have no independently verified instance identity,
and reference probabilities are not calibrated. Unit replay verifies the stated
view algebra, not identity. Protection can also preserve false parts. Neither
compilation nor success here proves the full v10 function or paper novelty.

Prediction uses sealed current-query semantic evidence from the earlier study as
an input fixture; checks field/geometry/query bindings and reports this reuse.
It is not a fresh raw-text-encoder validation. Source-free cold validation copies
the field plus current-query fixture and denies original source/result roots.
The implementation is shared and optional; NVOS integration is a separate next
validation, not implied by LERF results. No default promotion or full-five claim.
