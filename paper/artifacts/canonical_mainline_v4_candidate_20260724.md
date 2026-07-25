# Canonical query field v4 candidate — 2026-07-24

This is a promotion candidate, not a frozen main-table replacement. It keeps
one compact canonical C-RADIO primitive field and changes only shared,
query-independent readout contracts.

## Retained evidence

- Text query uses `SurfaceRegionContractV2`: symmetric typed surface regions,
  fixed physical radii 0.25/0.45/0.70 m, and independent normalized cosine.
  A ScanNet-frozen specificity rule selects the smallest scale within 0.02
  cosine of the numerical maximum. It raises LERF scene-macro mIoU/LocAcc from
  0.3148/0.5754 to 0.3196/0.5919 and the controlled ScanNet three-scene mIoU
  from 0.4694/0.4700/0.5848 to 0.4704/0.4718/0.5894 for the 19/15/10 splits.
  All four LERF scenes improve in mIoU and three improve in localization.
- Registered-2D and world-3D prompts use the same official DINOv3/SAM3
  primitive capabilities, signed hard-seeded confidence random walker, and
  fixed 0.5 threshold.
- Four query-free spherical background modes are fitted once from each frozen
  scene capability bank. They open no object, label, mask, click trajectory,
  prediction, or metric. On the three-scene development pilot they improve
  IoU@5 from 0.3771 to 0.5115 and IoU@15 from 0.6280 to 0.6767. On a separate
  three-scene holdout they obtain 0.5423 IoU@5 and 0.6768 IoU@15.
- AGILE3D uses native covariance-aware world clicks and continuous
  Gaussian-to-official-5-cm readout. The historical 3-NN/10 cm
  observation-domain bridge is absent.
- PFPR uses only the method-visible held-out RGB patch and the official DINOv3
  center 3x3 descriptor. It does not use instance identity, query pose/depth,
  support propagation, or evaluator-private anchors.

## Engineering closure

High-resolution COLMAP geometry now supports a fixed, label-free primitive
budget. After a densification event, lowest-opacity rows are deterministically
pruned with stable primitive-index tie breaking. A zero budget exactly
preserves the historical unlimited path. This addresses the SPIn-NeRF failures
where some scenes expanded beyond 1.6 million Gaussians and exhausted a 24 GB
GPU; it does not change cameras, RGB supervision, prompts, or metrics.

## Promotion gates still open

The candidate is promoted only after all of the following are complete:

1. ScanNet-PFPR-Small v2: 20 scenes / 200 queries.
2. AGILE3D ScanNet40: 312 scenes / 10,357 objects.
3. Background-4 transfer on all eight frozen NVOS scenes without retuning.
4. The nine available SPIn-NeRF scenes under the declared registered-prompt
   diagnostic protocol.

Until then, canonical-mpr-v3 and its frozen rows remain the last closed
mainline; partial scene results are diagnostic only.
