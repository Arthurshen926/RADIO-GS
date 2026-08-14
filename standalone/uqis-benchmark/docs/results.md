# Current candidate results

## Unified-Query Core

The evaluator-controlled v0.2 candidate contains 31 Core targets in all nine
scenes. LUDVIG CLIP+DINO diffusion obtained:

| Modality | AP | AP 95% CI | Oracle IoU | Fixed IoU@0.5 |
|---|---:|---:|---:|---:|
| text | 0.2330 | [0.1521, 0.3113] | 0.1937 | 0.0405 |
| image | 0.2988 | [0.1808, 0.4221] | 0.2416 | 0.0620 |
| point-2D | 0.5295 | [0.4671, 0.5913] | 0.4069 | 0.0649 |
| point-3D | 0.5387 | [0.4819, 0.5981] | 0.4104 | 0.0638 |

`UQ-Rank = 0.40004` with 95% CI `[0.33487, 0.46522]`.

`UQ-Mask@0.5 = 0.05778` is diagnostic only because there is no independent dev
calibration receipt.

## Same-expression text ablation

For the exact same 31 v0.2 expressions, deterministic direct CLIP mesh readout
obtained text AP 0.16365 and UQ-Rank 0.38269. DINO graph diffusion increased
text AP by 0.06939 (42.4% relative) and UQ-Rank by 0.01735 (4.53% relative).

The direct readout was produced after the evaluator claim to correct a detected
expression mismatch in the originally sealed legacy comparison. It changed no
parameters and uses saved pre-diffusion primitives, but remains explicitly
post-hoc and non-formal.

## Relational Text Challenge

On the 36 relation-required expressions across all nine scenes, LUDVIG
CLIP+DINO diffusion obtained scene-macro AP **0.16750**, with scene-clustered
95% CI **[0.08862, 0.25666]**. Oracle IoU was 0.14311 and diagnostic
fixed-IoU@0.5 was 0.02229. The 36-query prediction seal SHA-256 is
`b56f780f48d0ca26ed34a3541e052880da9d8bf2bdeb1e45b73fdec608d8c87a`;
the result SHA-256 is
`7c7a73f85a0396f6af83d2e8b593d5924eb968da4f509a9bca475300365d3fcd`.

This tier is deliberately excluded from UQ-Rank. Its lower AP than Core text
(0.23304) quantifies the additional burden of relational instance grounding
without making the four query interfaces incomparable.

The same-expression post-hoc direct-CLIP readout obtained AP 0.09705 (95% CI
[0.05202, 0.15467]). Graph diffusion therefore added 0.07045 AP, or 72.6%
relative. The ablation report SHA-256 is
`6854deb26fb7ab3bef53c0b9ea2b8c95665a7b2dc5ef453d5119ea6a0cea1e67`.

## Recovery disclosure

The first Core scoring process claimed private authority and then failed while
reading a metric at an incorrect report nesting level. The evaluator recovered
only with the identical prediction seal; no method output or parameter changed.
The ledger records `consumed_complete_recovered_scoring_code`, so this candidate
must not be described as an unqualified pristine one-shot formal run.
