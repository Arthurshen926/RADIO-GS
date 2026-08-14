# ScanNet-UQIS-9 v0.2 construction candidate

Status: non-formal construction candidate. It must not be reported as a
leaderboard release or compared as an official LUDVIG reproduction.

## Purpose

v0.2 keeps the same nine ScanNet scenes and the same 67 physical target
instances as v0.1. It changes task decomposition, not scene difficulty:

- `Unified-Query Core Cohort` contains targets with at least one correct,
  class-mentioning, view-independent Nr3D expression explicitly annotated
  with `uses_spatial_lang=false`;
- `Relational Text Challenge` contains the remaining valid targets whose text
  requires spatial/relational language;
- image, point-2D and point-3D remain paired to all 67 targets, but the primary
  four-modality aggregate is computed only on Core targets;
- relational text is reported separately and never averaged into the primary
  four-modality result.

The first content-bound candidate contains 31 Core targets and 36 relational
text targets across all nine scenes. Its private construction artifact is
`/mnt/pool/sqy/results/RADIO-GS/output/scannet_uqis_9_v0_2_candidate/text_profile_v1.json`
with candidate SHA-256
`62618e248a13eaf6360d008182be6969147a58a3a5f4c93758eb7ea7526caea3`.

## Metrics

- `UQ-Rank`: equal-modality mean of equal-scene average precision on the Core
  cohort. It tests whether a method ranks the target mesh above background
  without requiring probability calibration.
- `UQ-Mask`: equal-modality mean of equal-scene fixed-threshold IoU on the Core
  cohort. The threshold/calibration must be fitted on a sealed dev cohort and
  bound by a calibration receipt before test predictions are opened.
- `Relational Text Challenge`: text-only AP and fixed-IoU summary on the
  relation-required cohort, with its denominator always reported.

Until a dev calibration authority exists, `UQ-Mask` is diagnostic and must be
marked `calibration_status=diagnostic_unverified`.

## LUDVIG comparator

LUDVIG is represented honestly as a per-scene field dependency set:

- a persistent OpenCLIP 512-D field for text;
- a persistent DINOv2/PCA40 field shared by image, point-2D and point-3D;
- text graph diffusion depends on both fields, aligning CLIP scores to the
  pruned DINO carrier through the frozen `source_indices.npy` artifact.

The benchmark-local diffusion path uses a spatial 64-NN graph weighted by
DINO similarity, 20 stationary iterations, feature bandwidth 0.5,
regularizer bandwidth 2.0 and seed quantile 0.999. These values are fixed
without opening evaluator labels. The query process opens no captured RGB, no
SAM model/mask and no evaluator-private instance labels.

The first real GPU1 integration smoke completed on `scene0203_00` and produced
a finite float32 probability vector on all 205,756 bound mesh vertices. This
is runtime evidence only; no evaluator metric was opened.

The subsequent 31-query controlled Core run obtained `UQ-Rank=0.40004` and
Core text AP `0.23304`. A same-expression deterministic direct-CLIP readout
obtained text AP `0.16365`, so DINO graph diffusion improved text ranking by
`+0.06939` absolute. See
`paper/artifacts/scannet_uqis_9_v0_2_ludvig_candidate_result_20260814.md` for
the non-formal claim boundary, calibration warning and evaluator recovery
record.

The separately sealed 36-query Relational Text Challenge obtained scene-macro
AP `0.16750` with scene-clustered 95% CI `[0.08862, 0.25666]`, oracle IoU
`0.14311`, and diagnostic fixed-IoU@0.5 `0.02229`. This result is not included
in UQ-Rank. The same-expression direct-CLIP readout obtained AP `0.09705`, so
graph diffusion improved relational text AP by `+0.07045` absolute (72.6%
relative).

## Remaining release blockers

1. Freeze dev scenes and a dev-only calibration receipt for `UQ-Mask`.
2. Promote the v0.2 target profile into versioned public method bundles and a
   separate evaluator-private bundle with independent one-query workspaces.
3. Seal the exact LUDVIG graph topology/cache as a charged persistent artifact
   or rebuild it independently in every query workspace.
4. Run all predictions, seal them before evaluator-private data is opened, and
   consume the cohort through a one-shot evaluation ledger.
5. Enable formal release only after the external immutable release authority
   and one-shot evaluation service are operational.
