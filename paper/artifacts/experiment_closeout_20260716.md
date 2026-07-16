# Experiment closeout — 2026-07-16

All rows below use frozen, scene-independent protocol constants and no
test-set calibration.

## Completed current-mainline results

| benchmark | scope | primary result |
|---|---|---|
| LERF text | 4 scenes, 208 samples | scene-macro mIoU 0.248048, LocAcc 0.431605 |
| ScanNet text 19/15/10 | 3 official meshes | mIoU 0.292998 / 0.295680 / 0.396866 |
| ScanNet point | 3 official instance meshes | one-click macro mIoU 0.322015 |
| canonical-mpr-v2 | 3 held-out scenes | raw/DINO/SAM means and p05 non-inferior on every scene |
| boundary residual | 3 held-out scenes | rank-16 improves SAM mean and boundary retention on every scene |

LERF and ScanNet text use official C-RADIOv4/SigLIP2-G crop summaries lifted
query-free to primitives, official text embeddings, no learned query head, no
custom text projection, and no mask refinement.

## Compactness

On the same Ramen geometry and held-out frames, direct PCA d64/128/192/256/384
raw render cosine is 0.721350/0.733168/0.736923/0.738682/0.740227.  Official
SAM3 render cosine is 0.677604/0.685854/0.688143/0.689031/0.689751.  The
195.9-MiB local128-to-d256 fusion field obtains 0.742731 raw, 0.831868 DINO,
and 0.691673 SAM, exceeding the 194.1-MiB direct d128 field and even the
569.0-MiB d384 field on held-out rendering.  The fusion design therefore has
matched-storage Pareto evidence.  Larger dimensions still do not solve SAM3
boundary retention.

## Release gates that cannot be honestly marked complete

- NVOS has an earlier complete eight-scene frozen score-lift result, but only
  fern has been rebuilt with the latest canonical-MPR training chain.
- SPIn-NeRF has only fern completed with the latest canonical field.  The
  original Fork RGB/camera bundle remains absent, so a formal ten-scene row is
  fail-closed.  Full-reference-mask propagation is not the undisclosed SAGA
  sparse-point protocol.
- No UnCoCo or ScanNet++ held-out image-query dataset is present locally;
  pose-free image query therefore has no formal benchmark row.
- The official SAM3 boundary relation remains intrinsically weak.  The
  screen-only residual is a small observation-fidelity enhancement, not a
  solution to primitive boundary reconstruction.

## Primary artifacts

- `output/optimization_20260716/semantic_lifting/lerf_four_scene_primitive_score_summary.json`
- `output/optimization_20260716/scannet_text/three_scene_summary.json`
- `output/optimization_20260716/scannet_point/canonical_mpr_v2/`
- `output/optimization_20260716/compactness/`
- `paper/artifacts/canonical_mainline_v2.yaml`
