# Query authority high-value closure

## Scope

This round tests two remaining high-value gaps without opening benchmark masks
or labels during training: benchmark-disjoint generic-text extent transfer for
LERF, and counterfactual decision authority for ScanNet OVS. NVOS remains
frozen at the previously cold-start-confirmed `0.92555` RGB-assisted full8 row.
SPIn remains outside the active short-term scope.

## LERF generic-text extent authority

Official SigLIP2 ImageNet-1k text banks were split into fit/dev/audit banks.
For each source-only cross-view object episode, the compiler chooses a token by
maximizing the minimum query/target crop similarity. Eligibility requires the
token to lie in both crops' top-k sets and to pass a shared margin. The retained
episode counts are:

| scene | fit | dev | audit |
|---|---:|---:|---:|
| ramen | 76/82 | 66/82 | 78/82 |
| teatime | 54/58 | 54/58 | 54/58 |
| waldo_kitchen | 14/22 | 14/22 | 18/22 |

The joint decoder now alternates image-query and fit-text episodes, uses dev
only for checkpoint/calibration, and reserves audit for the final source gate.
Image and text posterior thresholds are typed because their score scales are
not exchangeable. A dev-calibrated continuous selective blend contains the
primitive identity score as an exact fallback.

Learned text projection fails all six seeds. Image-query heldout gains remain
strong, but Waldo generic-text audit regresses. Typed calibration improves the
best text macro result to `+0.04502`; selective fallback reaches `+0.03610`,
but neither is scene-wise noninferior.

A fixed, parameter-free cosine projection reduces vocabulary overfitting. Its
best seed obtains text audit macro `+0.05823` (ramen `+0.1006`, teatime
`+0.0905`, Waldo `-0.0164`). A different seed is strictly positive on all
three text-audit scenes (`+0.01211` macro), but regresses an image heldout scene.
No one seed passes both image and text gates, so no benchmark full4 expansion
or method promotion is authorized.

The result supports identity--extent factorization but rejects the claim that
benchmark-disjoint generic class tokens alone provide a stable physical-object
extent authority. The remaining LERF gap is an independent, complete
cross-view instance teacher with substantially better Waldo coverage.

## ScanNet counterfactual authority

The categorical decoder now adds replay loss on baseline-correct source rows.
Checkpoint selection directly measures source-only top-1 recovery mass minus
harm mass against the distilled teacher, weighted by query-independent
opacity-volume significance. This exposed that every raw candidate remains net
negative: replay weights `1/2/5/10` do not close the gap.

The former eligibility target was structurally wrong: it predicted whether the
teacher differs from the baseline, not whether adopting the actual candidate
helps. Two replacements were tested:

1. row-level recovery-versus-harm authority;
2. permutation-equivariant Gaussian/query-pair authority trained on whether a
   candidate coordinate reduces source-teacher error.

The row-level version preserves all seen scene-splits but fails query-holdout
in three seeds. The pair-level version fixes the all-query row coupling and
passes all seen scene-split noninferiority in six configurations, but still
fails strict query-holdout and does not improve every scene-split. It is not
promoted.

This falsifies score-error reduction as a sufficient proxy for categorical
mIoU. The next ScanNet authority must optimize query-set top-1 counterfactual
utility directly and remain equivariant to query permutation/cardinality. It
should not continue tuning replay weights or scalar thresholds.

## Method status

The new compilers, typed calibration, exact primitive fallback, counterfactual
checkpoint metric, and pair-equivariant authority are retained as research
infrastructure. None of the failed candidates replaces the current development
rows (`0.39584` LERF2D, `0.43730` LERF3D, and ScanNet
`0.36401/0.36189/0.46716`). The failures narrow the remaining problem rather
than constituting a benchmark improvement.
