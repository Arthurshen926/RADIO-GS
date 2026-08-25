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

## Post-rejection contract audit and next candidate

The failed full4 posterior has now been audited before any further training.
All four scenes fail the same two deployment contracts:

- the `4x` peak-local radius covers between about `92%` and `99.9%` of scene
  rows, so the nominal authority is effectively scene-global;
- no raw text-identity probability crosses the inherited `0.644475` posterior
  threshold. The selected support is therefore produced almost entirely by
  the extent residual, not by calibrated identity evidence.

Figurines additionally used `scene_canonicalizer_index=-1`, while the other
three scenes used learned scene codes. This confirms an unseen-scene forward
path mismatch rather than a valid scene-heldout canonicalizer experiment. The
immutable diagnostic is
`paper/artifacts/lerf_anchor_posterior_contract_audit_sampled_v2_20260825.json`.

Three implementation guards now block recurrence:

1. posterior construction restores the identity gauge, extent-conditioning
   mode and fixed decision threshold from checkpoint metadata;
2. unseen-scene identity canonicalization fails closed unless explicitly
   requested as a diagnostic;
3. peak-local spatial authority has a fixed maximum scene-row fraction.

The third guard is not promoted until it passes source-only full-track gates.
A modality-specific `CanonicalIdentityEvidenceCalibrator` is also introduced:
it converts positive-affine-gauge-invariant score and empirical-rank evidence,
anchor distance and reliability into a canonical membership logit. Its raw
and rank contributions are monotone and spatial distance can only reduce
evidence.

The direct-language ceiling is a distinct experiment from the rejected
query-independent SigLIP descriptor-before-MPR arm. Its materializer applies
the frozen official SigLIP2 head and text response per legal source view, then
lifts only query sufficient statistics through exact MPR. Figurines is the
low-memory sentinel; full4 evaluation remains unopened until all caches pass
their construction contracts.

The corrected Figurines sentinel covers 96,879 rows directly and 97,082 valid
rows after union with the frozen fallback authority. With fixed canonical
negative relevancy and threshold `0.5`, LERF2D mIoU is only `0.02427`, while
LocAcc remains `0.91071`. The mean exact-MPR response therefore retains sparse
identity peaks but is not a full-object membership teacher. It is rejected as
an extent posterior and is not expanded to the other three benchmark scenes.
Any robust multiview statistic must first pass source-only full-track
membership; target results cannot select mean/max/quantile aggregation.

## Text authority and physical-track closure

The remaining P1/P2 alternatives were tested without changing the frozen
field or benchmark threshold.

- A row-wise top-2 robust view aggregate reduced the Figurines direct-language
  sentinel from `0.02427` to `0.01588` mIoU. Mean-view dilution is therefore
  not the primary failure.
- Four raw/rank/distance/reliability canonical calibrators failed the
  three-scene source Brier gate. Class-balanced proper scoring did not rescue
  them: a single monotone map cannot turn generic text relevance into physical
  instance membership when the source language authority itself is wrong.
- Pinned Florence-2 captions were generated from three legal source crops for
  every available physical track. Under the frozen official SigLIP2
  cross-view, crop-caption and sibling-discrimination audit, only `3/19`
  tracks passed all three conditions. The captions are retained as an audited
  negative result, not training authority.

The query-free SAM hierarchy was then compiled into a maximum-confidence
physical-track forest with a strict invariant: a component may contain at
most one proposal from each registered view. This eliminates the ordinary
connected-component failure, where one transitive component absorbed
`245/277` proposals. Choosing exactly one physical track from identity before
extent raised the direct-language Figurines LERF2D result from `0.02427` to
`0.21713`, with LocAcc unchanged at `0.91071`. This is strong causal evidence
for identity/range factorization. Query-free per-view proposals without the
physical-track constraint remained flat at `0.02422`.

The paired LERF3D gate rejects promotion. Direct-language plus hard tracks
obtained only `0.2688` mIoU and D512 identity plus hard tracks obtained
`0.2542`, versus the matched primitive development baseline `0.57207`.
Moreover, the earlier source-gated soft physical association remains the
strongest single-scene 2D candidate (`0.42657`) but also regresses LERF3D to
`0.51544`. Hard tracks improve range coherence but do not yet provide pure
Gaussian membership. No full4 expansion is authorized.

The surviving high-value problem is now narrower: learn or construct a
signed, visibility-aware Gaussian membership for each physical track, with
explicit different-instance veto and unknown outside observed support. More
text calibration, robust averaging or graph closure is not justified by the
current evidence. ScanNet and NVOS remain frozen at
`0.36401/0.36189/0.46716` and `0.92555`; this LERF mechanism study does not
change their retained rows.

## Formal-readout contract recovery

A later code audit established that the `0.06178/0.03779` full4 collapse was
not a valid architectural comparison. The deployed extent checkpoint had
failed source validation noninferiority, lacked the final decision-gauge
contract, inherited an incompatible text threshold, and the builder replaced
the primitive query-cache validity mask by all-true rows. Figurines also used
the explicit unseen-scene diagnostic path instead of a source-gated
canonicalizer.

The builder now requires both source validation and source audit
noninferiority, requires calibrated identity logits together with fixed
posterior threshold `0.5`, preserves the immutable validity mask, zeros
invalid-row scores and excludes invalid rows from extent authority. Replaying
the corrected 60-step source contract gives validation macro delta
`-0.01567`, so the candidate is correctly rejected before benchmark
materialization despite a positive audit split.

The formal route is restored to the retained readouts. A current-code
Figurines replay obtains LERF2D mIoU `0.42624`, LocAcc `0.91071`, and LERF3D
mIoU `0.58403`, Acc@0.25/0.50 `0.85714/0.66071`. Therefore neither the frozen
field nor current evaluators lost their normal accuracy. Anchor-conditioned
Query-Native remains an experimental branch until it independently passes
the corrected source contract; it cannot replace the formal LERF posterior.
