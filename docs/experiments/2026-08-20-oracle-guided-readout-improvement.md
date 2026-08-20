# Oracle-guided typed-readout continuation (2026-08-20)

## Scope and method boundary

Universal Field v1 and the promoted typed instance readouts remain frozen.  All
target-label-dependent calculations in this note are development-only oracle
diagnostics and are not benchmark methods.  The purpose of the oracle ladder is
to identify which source-trainable component has enough remaining headroom to
justify an implementation change.

The LERF ladder changes one authority at a time:

1. `O0`: frozen deployable object posterior;
2. `O1`: oracle identity for one proposal, fixed lifted membership;
3. `O2`: oracle greedy distinct-view proposal association, fixed membership;
4. `O3`: `O2` with an oracle primitive-membership threshold;
5. `O4`: exact-adjoint Gaussian target without a SAM proposal-set constraint;
6. `O5`: `O4` with a per-sample oracle pixel threshold.

The implementation is
`radio_gs/scripts/eval_lerf_proposal_oracle_ladder.py`.  It writes a sealed
membership artifact before metric materialization and can resume from that
artifact without rerunning the expensive exact adjoint.

## Full-four oracle result

Figurines artifact (the other three scene receipts are siblings under the same
root):

`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260820/lerf_proposal_oracle_ladder_full4/figurines/result.json`

SHA-256: `68fb2e3a053d414f492714d7f55057704764352585e55537ddad04eb32aec1e7`

| Stage | Rendered sample-mean fixed IoU | Interpretation |
|---|---:|---|
| O0 current | 0.33338 | Diagnostic replay baseline; not the promoted full evaluator row |
| O1 oracle single identity | 0.26242 | One proposal is usually only a discriminative part |
| O2 oracle association | 0.45674 | Correct multi-view grouping has material headroom |
| O3 oracle membership | 0.47324 | Threshold calibration adds only 0.01650 after association |
| O4 exact adjoint | 0.83639 | The Gaussian carrier/renderer can express far more complete extent |
| O5 oracle boundary | 0.93580 | Pixel boundary calibration accounts for another 0.09941 ceiling |

The completed rendered fixed-IoU rows are:

| Scene | O0 | O1 | O2 | O3 | O4 | O5 |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.33338 | 0.26242 | 0.45674 | 0.47324 | 0.83639 | 0.93580 |
| Ramen | 0.30733 | 0.25425 | 0.32011 | 0.32771 | 0.85993 | 0.91891 |
| Teatime | 0.36424 | 0.33886 | 0.52725 | 0.54387 | 0.88628 | 0.95658 |
| Waldo Kitchen | 0.10054 | 0.17863 | 0.20248 | 0.20411 | 0.81919 | 0.95386 |
| Scene macro | 0.27637 | 0.25854 | 0.37665 | 0.38724 | 0.85045 | 0.94129 |
| Sample micro | 0.30861 | 0.27245 | 0.40321 | 0.41514 | 0.85676 | 0.93784 |

O2 improves over O1 in all four scenes.  O3 adds only `0.01059` scene-macro
and `0.01193` sample-micro IoU, whereas removing the finite proposal-set
constraint at O4 adds `0.46321` scene-macro IoU.  O4 is above `0.81` in every
scene and O5 is above `0.91` in every scene.  Therefore the conclusion is
full-cohort consistent: identity-conditioned multi-view association matters,
membership-threshold tuning is secondary, and the dominant gap is the finite
proposal set/coverage rather than the Gaussian carrier's ability to express
the target support.  O1 being worse than O0 in aggregate also proves that a
single high-identity SAM proposal often captures only a discriminative part.

This rejects “tune the membership threshold” as the next primary direction.
The largest actionable loss lies in proposal coverage and cross-view grouping.
The original source32 hierarchy contains only 277 proposals over 32 views.  Its
manifest used an 8-by-8 full-image point grid even though the sealed Stage-A
design registered a 16-by-16 grid.  The query-independent 16-by-16 Figurines
sentinel has now sealed all 32 source views and produced 579 proposals with
323,507 exact-MPR Gaussian--proposal memberships over 168,791 Gaussian rows.
The official masked-crop summary teacher is also sealed for all 579 proposals
(mean masked/context cosine `0.81928`).  Frozen typed-readout scoring and the
paired LERF2D/LERF3D evaluation completed from those two sealed artifacts.  No
target mask entered proposal construction, teacher extraction, or scoring.

The denser sentinel did not pass the joint gate.  Figurines LERF2D mIoU is
`0.40065`, versus `0.39719` for the current listwise candidate and `0.42624`
for the older source32 scorer.  LERF3D mIoU is `0.45402`, versus `0.46574` and
`0.51650`, respectively.  Thus doubling raw proposals adds only `0.00346`
against the current LERF2D candidate while regressing its paired LERF3D row by
`0.01173`.  The candidate is rejected and the remaining three scenes are not
materialized under this absolute-score arm.  Coverage is necessary, but denser
proposals expose the limits of the current hard weighted-Jaccard components:
proposal/null calibration and learned same/different/unknown association must
precede another proposal-density expansion.  A matched query-listwise grid16
arm is run once to avoid conflating density with the already accepted listwise
identity gate.

That matched arm also fails: Figurines LERF2D/LERF3D mIoU is
`0.39143/0.42094`, below the current listwise `0.39719/0.46574`.  The density
candidate is therefore closed after the sentinel.  The failure is not an
absolute-vs-listwise gate mismatch; the hard component posterior assigns excess
extent to newly available local/distractor proposals.

Teatime initially failed because another GPU-5 workload left less than 1 GiB
free for a whole-frame stable hit sort.  The streamed depth-range compositor
completed it without changing the front-to-back recurrence, closing the full4
ladder.  Teatime's result SHA-256 is
`d11ebe894afaab3dcf42943a885208880911019efbf7339905a0d0c09c7584c9`.

## ScanNet categorical screen

Artifact:

`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260820/scannet_categorical_minimal_screen_v1.json`

SHA-256: `b4bc78bedb0842d0a6fe766c1d07d6fa0fddc9827f59794de8e22e16ed197069`

The fixed 4-by-2 shrinkage/background screen selected
`shrink_0.75_with_background`.  Paper8 development metrics changed as follows:

| Split | Primitive-v0 mIoU | Development head mIoU | Delta | Primitive-v0 mAcc | Development head mAcc |
|---|---:|---:|---:|---:|---:|
| 19 | 0.33535 | 0.41023 | +0.07487 | 0.66525 | 0.68053 |
| 15 | 0.33370 | 0.43269 | +0.09899 | 0.64954 | 0.69520 |
| 10 | 0.42213 | 0.52560 | +0.10347 | 0.73871 | 0.78847 |

Heldout6 mIoU also improves on all splits (`0.34169 -> 0.42835`,
`0.34017 -> 0.45496`, and `0.45039 -> 0.55612`).  Background enabled and
disabled variants are numerically identical, so the gain is attributable to
constrained class competition/temperature/bias rather than unknown rejection.
Because the parent checkpoint used two paper8 development scenes and the
variant was selected on heldout6 metrics, this is mechanism evidence only.  It
cannot replace the frozen eligible ScanNet row until a source-only LOSO head
passes.

## Memory-equivalent execution fixes

The oracle and SPIn runs exposed three avoidable allocation peaks:

- front-to-back weights repeated hit-sized float64 group baselines; the new
  implementation uses an equivalent per-pixel baseline lookup and bounded
  in-place chunks, with bitwise equality to the old formula;
- Gaussian footprint alpha computation now evaluates the same pointwise
  equation in fixed hit chunks;
- exact-adjoint `A^T target` and `A^T 1` use bounded `index_add` chunks rather
  than materializing/transposing the full sparse CSR;
- high-overdraw frames now use gsplat's consecutive depth-range API and carry
  per-pixel transmittance between ranges, so accepted hits are reduced before
  the next range is materialized instead of sorting a whole frame at once;
- D512/L512 local codes are cast to the requested FP16 dtype on CPU before the
  first CUDA allocation, avoiding simultaneous complete FP32 and FP16 tensors.

Verification completed: 24 compositor/authority tests and 12 canonical-field,
optimizer-memory, and factorized integration tests pass.  Direct randomized
checks show bitwise equality for front-to-back weights and footprint alphas;
the chunked adjoint agrees with sparse matrix multiplication within `1.91e-6`,
and the split-depth recurrence agrees with whole-frame compositing on the
registered synthetic invariant test.

## Current decision

The next LERF candidate is the source-trained proposal/null scorer with
same/different/unknown cross-view association on the existing sealed proposal
sets.  Further raw density and membership-threshold tuning are deprioritized.
Reliability-conditioned fusion remains in scope only as a monotone precision
weight after the association gate.
ScanNet work targets a transferable shared competition rule, not a larger
background head.  SPIn field construction and the frozen full9 barrier continue
unchanged in parallel.

The explicit latent proposal/null marginal and same/different/unknown
co-membership authority are now executable public query operators with unit
coverage for convex marginalization, invalid-proposal zero mass, bitwise null
fallback, and visibility fail-closed behavior.  The learned source-only scorer
remains unpromoted until its cross-scene gate passes.

The stricter ScanNet single-development-scene to seven-heldout-scene attempt
stopped before training because one scene0347 primitive row did not exactly
replay the frozen argmax at a numerical top-two tie.  The replay authority was
not weakened and no checkpoint was selected.  Therefore the larger categorical
screen remains diagnostic; the promoted ScanNet row is unchanged.

## Registered RGB prompt contract correction

The `0.81776` NVOS target-frame box/point selector remains valid RGB-assisted
development evidence, but it is not the previously frozen all-view primary
compiler.  The primary NVOS/SPIn contract freezes ten reference candidates,
processes every registered captured RGB independently with official SAM3,
registers masks with exact compositor adjoints, and marginalizes candidates and
views symmetrically.  Its deterministic signed-point sampler and convex,
digest-canonicalized marginal are now implemented in
`radio_gs/querying/synchronous_multiview_candidate_marginal.py`; four contract
tests cover order invariance, probability normalization, signed support, and
fail-closed incomplete K.  The complete RGB/SAM3/adjoint materializer and dual
full8/full9 experiment remain open, so neither `0.81776` nor the pending SPIn
target-frame full9 sentinel is represented as that final all-view method.

## Analytic latent proposal/null full4 closure

The first probability-correct realization now replaces hard component
selection by one normalized null/proposal marginal.  It uses the existing
source-only official SAM3 proposals, exact-MPR memberships, masked-crop
SigLIP2 identity, immutable field-peak anchoring, and continuous cross-view
Gaussian overlap.  Proposal multiplicity receives a uniform cohort prior and
the null logit is fixed at zero.  No benchmark mask is read by score
construction and no per-scene parameter is used.

| Scene | Previous LERF3D mIoU | Latent marginal mIoU | Delta |
|---|---:|---:|---:|
| figurines | 0.51650 | 0.58403 | +0.06753 |
| ramen | 0.31056 | 0.30291 | -0.00765 |
| teatime | 0.41514 | 0.49970 | +0.08457 |
| waldo_kitchen | 0.32161 | 0.33020 | +0.00859 |

The exact 208-sample full4 result is mIoU `0.43730382`, Acc@0.25
`0.67788462`, and Acc@0.50 `0.48076923`, versus `0.39683586`, `0.61538462`,
and `0.40865385`.  Thus all three 3D endpoints improve, with three of four
scenes improving.  The same posterior is not promoted for LERF2D because its
projected raster behavior is not uniformly better: full4 mIoU is `0.37079632`
versus the retained `0.39584174`, while LocAcc is exactly unchanged at
`0.87980769`.  The 2D boundary/readout branch therefore remains separate.
This is a promoted development-method improvement, not prospective blind
evidence.
