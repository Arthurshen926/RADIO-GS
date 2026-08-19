# SAM3 probability/logit contract erratum (2026-08-17)

## Finding

`build_sam3_foundation_cache.make_sam3_cache_payload` stores the official SAM3
mask output after `masks.float()` under the historical key `mask_logits`.  For
the mapped LERF source caches these tensors are probabilities in `[0,1]`, not
pre-sigmoid logits.  A representative Figurines source frame has:

- minimum `1.15e-24`, maximum `0.9999999`, mean `0.01535`;
- fraction greater than `0.5`: `0.01551`.

The v1 LERF exact-MPR builder applied `sigmoid` again.  Near-zero background
therefore became approximately `0.5` and passed the registered membership
threshold.  On Figurines this produced `3,174,537` sparse pairs.  The corrected
probability-space v2 cache contains `46,407` pairs, a `68.41x` reduction.

The v1 artifact remains immutable.  The correction is implemented in
`build_lerf_sam3_exact_mpr_memberships_v2.py`, which:

1. requires finite values in `[0,1]`;
2. applies the identity transform before exact-MPR lifting;
3. stores proposal quality separately from conditional row membership;
4. declares the current masks query-conditioned and therefore not the final
   query-independent P0 hierarchy.

## Figurines paired result

All selector/readout parameters were frozen to the prior residual experiment:
query-matched proposals, alpha `0.25`, seed-support ratio `0.8`, at least two
source views, VALA kNN10/min-max, threshold `0.6`, and the same component guard.

| Input | mIoU | Acc@0.25 | Acc@0.50 |
|---|---:|---:|---:|
| no SAM residual | 0.576577 | 0.839286 | 0.696429 |
| v1 double-sigmoid residual | 0.590067 | 0.875000 | 0.678571 |
| v2 probability-correct residual | **0.606836** | **0.892857** | **0.750000** |

Thus the corrected residual is `+0.030260` mIoU over its no-SAM paired baseline
and `+0.016769` over the contaminated v1 residual.

## Corrected full4 transfer result

The same fixed v2 parameters completed Ramen, Teatime, and Waldo.  Across 208
frame-query samples, sample-micro metrics are:

| Metric | Primitive | V2 residual | Delta |
|---|---:|---:|---:|
| mIoU | 0.484513 | 0.492444 | +0.007930 |
| Acc@0.25 | 0.774038 | 0.798077 | +0.024038 |
| Acc@0.50 | 0.528846 | 0.543269 | +0.014423 |
| Boundary-F | 0.712012 | 0.705363 | -0.006649 |
| trimap IoU | 0.376743 | 0.369081 | -0.007663 |

Scene-macro mIoU rises from `0.465381` to `0.472757`, but transfer is unstable:
Figurines and Teatime improve by `0.030260` and `0.031087` mIoU, while Ramen and
Waldo regress by `0.024054` and `0.007790`.  This is not a promotion:
Boundary-F and trimap IoU regress, two of four scenes regress in mIoU, and the
masks remain benchmark-query-conditioned.  It is only evidence that a bounded
extent residual can sometimes help, not evidence that the proposed
query-independent hierarchy is complete.

A second pre-registered two-scene sentinel used proposal quality only for
within-view ranking and accepted-observation confidence, composed extent with
noisy-or, and preserved the text prior wherever proposal evidence was absent.
It also failed: Figurines fell from `0.576577` to `0.522815` mIoU and Ramen
fell from `0.404581` to `0.243642`.  This candidate is rejected with
`promotion=false`; monotone score addition is not monotone in IoU because it
can move background primitives across the fixed selection threshold.

An intentionally strong full-posterior replacement was also tested once.  V1
collapsed to `0.117486` because its proposals were nearly global.  V2 recovered
to `0.471110`, but remained below the guarded baseline because absence from a
sparse source mask is unknown rather than negative.  This is why the
strong-replacement candidate was rejected; any retained diagnostic must remain
a bounded positive residual until a visibility-complete, query-independent
hierarchy is built.

## Consumer audit

Affected consumers (do not silently reuse without an explicit tensor-semantics
contract):

- `build_lerf_sam3_exact_mpr_memberships.py`: directly applies a second
  sigmoid; all v1 membership caches and results using them are superseded.
- `models.sam3_proposal_registration.build_sam3_mask_memberships`: correctly
  consumes logits by its own API, but `smooth_scores_with_sam3_training_view_proposals`
  passes foundation-cache probabilities to it.  That caller path is affected.
- `models.foundation_cache._mask_boundary_response`,
  `_sam_region_compactness_loss`, and `_sam_region_separation_loss`: apply
  sigmoid to cache-derived probability masks.  Any training run with the
  corresponding non-zero foundation-cache mask/boundary/region weights needs a
  contract audit before promotion.
- The same foundation-cache path compares a projected value named logits
  directly with cached probabilities for `mask_logit_weight`; the output space
  is ambiguous and must be made explicit before reuse.

Not affected by this specific double-sigmoid bug:

- Current LERF2D feature-only prompt-mask heads use
  `target_activation=binary`, `target_threshold=0.5` in all four checkpoint
  configs.  Their training binarizes the cached probability directly.
- Sigmoid calls in `prompt_conditioned_mask_refinement.py` and the corresponding
  LERF evaluators operate on learned head output logits, not on foundation-cache
  probabilities.  The existing LERF2D boundary result is therefore not
  downgraded by this erratum.
- Query-free automatic-mask caches use packed boolean masks and are governed by
  a separate explicit contract.

## Remaining limitation

The corrected v2 cache is still generated from official SAM3 text-prompted
source masks whose prompt names are the benchmark queries.  It is a legal
source-only query-conditioned diagnostic, not a query-independent mask
hierarchy.  The final P0 implementation must lift official automatic
multi-scale masks, store per-view visibility/negative observation support, and
select hierarchy nodes using the immutable text identity seed.

The first query-free P0 grid-4 decode over eight source views is retained only
as an undercoverage smoke artifact: Figurines produced 19 proposals and one of
eight frames produced none.  It is not evidence against an instance prior.
The reviewed Figurines sparse pilot uses grid 12.  Earlier grid-12
materializations are retained as superseded artifacts because their manifests
lacked source-image SHA-256, output-cache SHA-256, or the complete producer
contract.  The final v5 producer hashes and decodes the same in-memory RGB
bytes, records both hashes per frame, and the exact-MPR lift requires the
canonical formula, every responsibility-shard SHA, a consumed source-to-MPR
preflight, uniform checkpoint/grid/schema/frame identity, packed-mask padding,
and row-aligned proposal attributes.

The result is 67 single-scale proposals and 44,538 memberships.  A non-circular
field-only identity audit finds one-view coverage for 7/21 queries and two-view
coverage for 1/21.  It is explicitly not a hierarchy or formal Stage-A result:
there is no crop pyramid or containment parent graph, and the eight-view sparse
undercoverage must not be generalized into a claim against SAM instance
priors.  See
`paper/artifacts/lerf_query_free_sparse_p0_grid12_coverage_20260817.json`.

## Contract closure and benchmark scope

New foundation-cache payloads now declare
`heads.sam3.mask_tensor_semantics=probability`.  The typed loader preserves that
field, and `models.mask_tensor_contract.mask_tensor_to_probability` rejects a
missing or unknown declaration instead of guessing from the historical key or
the observed range.  Existing cache files were not rewritten; an old cache can
only enter a new promoted experiment through a hash-bound local-v2 conversion.

The additional consumer audit is recorded in
`paper/artifacts/sam3_foundation_mask_semantics_consumer_audit_20260817.json`.
It confirms that current NVOS and SPIn paths consume official `predict_inst`
binary masks rather than these foundation caches, and that current ScanNet
paths use either adaptor features or separately contracted packed-boolean
automatic masks.  They are not downgraded by this erratum.  The current LERF2D
feature-only boundary heads are also unchanged because every materialized
checkpoint uses a binary pseudo target at probability threshold `0.5`.
