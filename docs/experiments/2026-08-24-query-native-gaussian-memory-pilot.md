# Query-Native Gaussian Memory pilot

## Question

Can the project replace foundation-feature reconstruction plus similarity with
a compact query-native memory that directly predicts Gaussian posteriors?

The first experiments deliberately freeze every L512 field.  They test the
decoder/query contract before authorizing a Universal Field v2 rebuild.

The categorical pilots consume the retained scalar baseline response as an
identity prior.  They do not reconstruct or output a 1536D descriptor, but an
unseen text still needs the v1 identity path to produce that scalar.  Results
below therefore validate posterior-level distillation and class-set
equivariance, not complete feature-decode-free deployment.

## Implemented contract

- `QueryPacket` separates decoder inputs from encoder identity.
- Text category decoding has no class-indexed parameter and is exactly
  permutation equivariant.
- Query cardinality is dynamic.
- Image/prompt posterior decoding separates identity retrieval and extent.
- Prompt seed probability clamps known evidence; NaN remains unknown.
- No experiment adds a per-Gaussian parameter or opens benchmark masks during
  fitting.

The v1 ScanNet decoder has `399,099` parameters (`1.52 MiB` FP32); its gate has
`124,587` parameters (`0.48 MiB`).  The larger sentinel has `531,051`
parameters (`2.03 MiB`).  These are constant per decoder, versus a
Gaussian-indexed high-dimensional capability sidecar.

## ScanNet per-scene query-native v1

Eight source-only scene decoders were trained in parallel.  The resulting
paper8 macro is:

| split | query-native v1 mIoU | retained compact mIoU | delta |
|---|---:|---:|---:|
| 19 | `0.36567` | `0.36401` | `+0.00166` |
| 15 | `0.36135` | `0.36189` | `-0.00054` |
| 10 | `0.46690` | `0.46716` | `-0.00027` |

This is not promoted because all three splits must be noninferior.  It does
show that removing the fixed 44-channel parameterization does not cause a
large accuracy collapse and improves the 19-class split.

## Cross-scene shared decoder

One decoder and one global gate threshold were trained across all paper8
scenes using the frozen source-distilled score caches as mapping-time teachers.
Its source gate passed all 24 scene/split checks.  Benchmark macro is:

| split | shared query-native mIoU | restored L512 baseline | retained compact |
|---|---:|---:|---:|
| 19 | `0.36118` | `0.35613` | `0.36401` |
| 15 | `0.36084` | `0.35300` | `0.36189` |
| 10 | `0.46434` | `0.45954` | `0.46716` |

The shared model improves the restored baseline but does not yet match the
scene-global compact student.  This measures the cost of removing all
scene-specific parameters.

A predeclared scene-query bipartite holdout improved 21/24 heldout pairs.  It
failed three pairs (`scene0000/split15`, `scene0062/split15`, and
`scene0590/split19`) by `0.0019--0.0029` MAE, so adapter-free arbitrary-query
transfer is not yet a valid claim.  In contrast, withholding the same words
from an individual scene degraded every split, demonstrating why global
cross-scene training is necessary.

## Larger per-scene equivariant sentinel

Increasing only the shared latent/query projections and pair MLP to `531,051`
parameters (`2.03 MiB`) gives:

| split | query-native v2 mIoU | retained compact mIoU | delta |
|---|---:|---:|---:|
| 19 | `0.36557` | `0.36401` | `+0.00156` |
| 15 | `0.36181` | `0.36189` | `-0.00008` |
| 10 | `0.46728` | `0.46716` | `+0.00012` |

The remaining 15-class gap is only `8.2e-5`, but strict promotion still fails.
A globally predeclared doubled 15-split sampling sentinel on scene0062 reduced
rather than removed the benchmark gap, so it is rejected; no outcome-based
split fallback is used.

## LERF direct posterior pilot

Figurines uses source-view residue 3 only for heldout evaluation.  The decoder
consumes official source SAM crop queries and frozen L512/reliability, and is
trained with BCE, Dice, Brier and identity ranking.  Invisible rows are
unknown.  Results are negative:

| variant | heldout proposal IoU |
|---|---:|
| primitive similarity control | `0.12457` |
| learned identity + SigLIP query | `0.04840` |
| learned identity + SigLIP/DINO query | `0.05174` |
| learned identity, 256D query | `0.02943` |
| identity-prior residual, SigLIP | `0.06404` |
| identity-prior residual, SigLIP/DINO | `0.04364` |

The identity-prior variant reaches about `0.197` on source-train calibration
but collapses on heldout proposals.  Direct posterior decoding therefore does
not remove the need for reliable cross-view object supervision.  Independent
same-view SAM masks teach proposal appearance and local support, not complete
physical-object extent.  Larger adapters or more loss weights are stopped.

## Current decision

The architecture remains a promising v2 candidate for categorical text
queries: it is efficient, structurally open-set, exceeds the retained result
on 19/10 classes and is within `8.2e-5` on 15 classes.  It is not a promoted
universal replacement because the strict three-split gate, LERF extent and
unseen-query transfer gates remain open, and the scalar identity prior still
comes from the retained v1 path.

The next justified LERF experiment must first create an independently gated
cross-view authority (for example, short-baseline mutual correspondence with
unknown handling).  Only then should the same Gaussian posterior be evaluated
on both LERF2D and LERF3D.  Training a larger direct mask decoder on the current
independent proposal carrier has low expected value.
