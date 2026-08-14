# ScanNet-UQIS-9 v0.2 LUDVIG candidate result — 2026-08-14

## Claim boundary

This is an evaluator-controlled, non-formal result on the 31-target
Unified-Query Core construction candidate. It is not a public leaderboard row,
not an official LUDVIG reproduction, and not directly comparable to historical
LERF/LUDVIG paper metrics.

The unchanged image, point-2D and point-3D query inputs reuse their immutable
v0.1 predictions. All 31 v0.2 Core text expressions were rerun in independent
read-only workspaces and fresh processes on physical GPU1. Text consumes both
the per-scene OpenCLIP field and the per-scene DINOv2/PCA40 field. It opens no
query-time RGB, SAM mask/model or evaluator-private label.

## Core UQ-Rank result

All point estimates are equal-scene macro averages over nine scenes. Confidence
intervals are 2,000 scene-clustered bootstrap samples with seed 20260813.

| Modality | Core targets | AP | AP 95% CI | Oracle IoU | Fixed IoU@0.5 |
|---|---:|---:|---:|---:|---:|
| text (CLIP+DINO diffusion) | 31 | 0.2330 | [0.1521, 0.3113] | 0.1937 | 0.0405 |
| image | 31 | 0.2988 | [0.1808, 0.4221] | 0.2416 | 0.0620 |
| point-2D | 31 | 0.5295 | [0.4671, 0.5913] | 0.4069 | 0.0649 |
| point-3D | 31 | 0.5387 | [0.4819, 0.5981] | 0.4104 | 0.0638 |

The primary candidate `UQ-Rank` is **0.40004**, with scene-clustered 95% CI
**[0.33487, 0.46522]**.

The diagnostic `UQ-Mask@0.5` is **0.05778** [0.03712, 0.07824]. It is not a
valid calibrated mask score because no dev-only calibration authority exists.

## Same-expression diffusion ablation

The originally sealed “legacy no-diffusion” comparison was discovered after
evaluation to contain v0.1 expressions for 21 of the 31 Core targets. It is
therefore confounded and must not be interpreted as a diffusion ablation.

A deterministic correction was derived from the already saved, hash-bound
pre-diffusion CLIP primitive relevance for the exact same v0.2 expressions.
No parameter was changed after evaluator labels opened; because its mesh
readout was produced after the evaluator claim, it is explicitly post-hoc and
non-formal.

| System | Core text AP | Core text oracle IoU | UQ-Rank | Diagnostic UQ-Mask@0.5 |
|---|---:|---:|---:|---:|
| direct CLIP, same expressions | 0.16365 | 0.13791 | 0.38269 | 0.06882 |
| CLIP+DINO diffusion | 0.23304 | 0.19371 | 0.40004 | 0.05778 |
| difference | +0.06939 | +0.05580 | +0.01735 | -0.01103 |

Graph diffusion improves Core text AP by **42.4% relative** and raises the
four-modality UQ-Rank by **4.53% relative**. The lower fixed-threshold IoU is a
calibration effect: diffusion changes score scale and yields no score above the
same useful fixed boundary for several queries. It must be addressed with a
prospectively frozen dev-only calibration, not test-label threshold search.

## Authority and recovery record

- v0.2 text-profile candidate SHA-256:
  `62618e248a13eaf6360d008182be6969147a58a3a5f4c93758eb7ea7526caea3`
- sealed 268-query batch SHA-256:
  `6920f6c827b2d5e75a792de9281627ca95bc721edf3ee7c7d5f09dfe21f59699`
- controlled result SHA-256:
  `a5b8fa2556d7c175e24516b98caf58af73bb49dfb0b6a18df19d23541aa5b612`
- corrected direct-CLIP ablation SHA-256:
  `25a34440620379d34b6a5b81236c5f32bb3fe920a2e20607b3616720b3d74ec1`
- relational 36-query seal SHA-256:
  `b56f780f48d0ca26ed34a3541e052880da9d8bf2bdeb1e45b73fdec608d8c87a`
- relational challenge result SHA-256:
  `7c7a73f85a0396f6af83d2e8b593d5924eb968da4f509a9bca475300365d3fcd`
- relational direct-CLIP ablation SHA-256:
  `6854deb26fb7ab3bef53c0b9ea2b8c95665a7b2dc5ef453d5119ea6a0cea1e67`

The first scoring process failed after claiming private authority because the
report reader addressed a Core metric at the wrong nesting level. Predictions,
seal and parameters were not changed. The evaluator resumed only after
verifying the identical seal hash; the ledger status is
`consumed_complete_recovered_scoring_code`. This recovery prevents the result
from being described as an unqualified pristine one-shot run, but it does not
introduce method adaptation.

## Relational Text Challenge

All 36 relation-required v0.2 expressions were subsequently rerun with the
same frozen CLIP+DINO diffusion parameters in independent GPU1 processes and
sealed before this challenge score. The evaluator-private labels had already
been opened during the earlier Core evaluation, so this remains a controlled
non-formal extension rather than a pristine formal evaluation.

| Queries | Scenes | Scene-macro AP | AP 95% CI | Oracle IoU | Diagnostic fixed IoU@0.5 |
|---:|---:|---:|---:|---:|---:|
| 36 | 9 | 0.16750 | [0.08862, 0.25666] | 0.14311 | 0.02229 |

The result is lower than Core text AP 0.23304, quantifying the intended extra
difficulty of relational instance grounding. It is reported separately and
does not enter UQ-Rank.

For the exact same 36 expressions and saved pre-diffusion CLIP primitives, the
post-hoc direct-CLIP readout obtained AP 0.09705 (95% CI [0.05202, 0.15467]).
Graph diffusion improved relational AP by **+0.07045 absolute** and **72.6%
relative**. As with the Core ablation, this is deterministic and changes no
parameter, but was derived after evaluator-private labels had already opened.

## Immutable artifacts

- controlled result:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/controlled_evaluation_v1/result.json`
- recovery ledger:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/controlled_evaluation_v1/controlled_evaluation_ledger.json`
- corrected same-expression ablation:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/controlled_evaluation_v1/direct_clip_same_expression_ablation.json`
- sealed batch:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/sealed_ludvig_system_v1/sealed_prediction_batch.json`
- relational challenge result:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/controlled_evaluation_v1/relational_text_challenge.json`
- relational direct-CLIP ablation:
  `/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/controlled_evaluation_v1/relational_direct_clip_ablation.json`
