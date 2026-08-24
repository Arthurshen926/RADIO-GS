# Anchor-conditioned Query-Native closure (2026-08-25)

## Scope

This round tested the three remaining structural hypotheses without retraining
the D512/L512 Universal Field:

1. text identity must compile a clean instance anchor before extent recovery;
2. extent residuals must be disabled outside mapping-time spatial authority;
3. generic class nouns must be replaced by source-only attribute/sibling text
   authority.

The implementation uses modality-specific identity retrieval, a modality-free
`AnchorPacket`, and one token-free shared extent decoder. Unknown rows replay
the frozen identity exactly. Positive-only physical episodes remain positive
and receive lower weight instead of being relabeled negative.

## Source-only mechanism evidence

The initial anchor audit found a large image/text gap. Global AnchorPurity@6
was `0.931/0.324` on Ramen, `0.844/0.467` on Teatime and `1.000/0.889` on
Waldo. Image/text anchor Jaccard was near zero. Peak-local selection removed
distant secondary instances but could not repair an incorrect primary peak.

The final shared extent seed (`radius_fraction=0.04`, authority radius `4x`)
improved source-heldout image IoU on all three scenes by
`+0.25283/+0.10440/+0.00412` (macro `+0.12045`). With text identity but oracle
image anchors it improved `+0.30603/+0.04823/+0.02088` (macro `+0.12505`).
This is direct evidence that token-free extent completion transfers across
modalities when anchor identity is correct.

A low-rank text retrieval adapter raised attribute-text anchor purity/peak
accuracy from approximately `0.45/0.45` to `0.7424/0.7576`. Reusing the adapted
identity as the posterior prior caused severe Waldo regression. Restricting it
to anchor compilation and replaying the frozen raw-text identity reduced the
Waldo regression from `-0.13946` to `-0.00153`. A query-independent 3D
agreement fallback at twice the raw local radius passed the final source text
gate: Ramen `+0.03611`, Teatime `+0.00873`, Waldo `+0.03711`, macro `+0.02732`.

## LERF full4 result: rejected

The same posterior and the source-selected threshold were evaluated under the
existing fixed LERF2D and LERF3D protocols.

| metric | result |
|---|---:|
| LERF2D sample-micro mIoU | `0.06178` |
| LERF2D scene-macro mIoU | `0.05988` |
| LERF2D sample-micro LocAcc | `0.87981` |
| LERF3D sample-micro mIoU | `0.03779` |
| LERF3D scene-macro mIoU | `0.04540` |
| LERF3D sample-micro Acc@0.25 / Acc@0.50 | `0.02404 / 0.00962` |

This does not exceed the retained development rows (`0.39584` LERF2D and
`0.43730` LERF3D) and is not promoted. Localization is preserved, so the
failure remains object extent rather than target identity. Source-selected
attribute phrases and source-visible proposal authority are not adequate
surrogates for benchmark-style language and full-scene object support.

## ScanNet constrained top-2 closure

The single fixed candidate changed only the baseline/raw-candidate top-1 pair
margin. It used source teacher top-1 decisions and query-independent
opacity-volume weights, with no benchmark labels or masks and no alpha sweep.
It produced positive net recovery mass (`+0.99377`) but failed unseen-query
noninferiority in multiple scene/splits, including severe split-19 regression
in scene0000 and scene0062. The Query-Native candidate therefore does not
replace the retained compact head (`0.36401/0.36189/0.46716`). Selector and
top-2 replacement search is closed unless a new candidate score itself passes
the source decision contract.

## Conclusion

The architecture is now identifiable and internally correct:

```text
modality-specific identity -> AnchorPacket -> token-free shared extent
```

The high-value remaining problem is not another readout threshold or a wider
decoder. It is missing training authority that jointly matches benchmark-like
referring language and complete 3D instance support. Further LERF work should
first obtain that authority (for example source region captions/contrastive
referring expressions plus multi-view object tracks with full-scene support),
then rerun the same frozen interface. The current candidate must remain a
mechanism ablation, not a paper result.

