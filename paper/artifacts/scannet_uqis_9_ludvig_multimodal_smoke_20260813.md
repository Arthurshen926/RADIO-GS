# ScanNet-UQIS-9 LUDVIG multi-field smoke — 2026-08-13

## Scope

This is a construction/runtime validation on `scene0030_00`, not an official
metric row. All retained query manifests say `split_role=pilot`, and all run
manifests remain `result_eligible=false` and
`formal_benchmark_row_eligible=false`. No evaluator-private labels or pairing
were opened by an adapter.

LUDVIG is represented honestly as `modality_specific_multi_field`:

- one unpruned OpenCLIP 512-D field for `text`;
- one top-support-pruned DINO/PCA40 field for `image`, `point_2d`, and
  `point_3d`.

## Bound fields

- formal 30k geometry receipt:
  `29198126cdc9986265fa3d94720fdb45a25dfa09aacb73d4cbcfbee80e33d838`
- legal mapping-observation receipt:
  `3cd0c82b1cbfc00b758a438d73f4770ecbab3b432e77d7e29ad6a46c21d34192`
- DINO Phase-B manifest:
  `77fb15eb83bcbbb4fda5f173f0c3f8112cb725f173b9d22afd3e886488703020`
- DINO uplift manifest:
  `54aace8caaaeb287ca48c45313c00df060d1ded891a85efb4e9f172313f31478`
- CLIP field manifest:
  `5104eca9d313b9fbeb69bc72a47631d6ca293968cb256a2c75fdd2d75cfe29f6`

The DINO field keeps 600,000 of 2,235,426 Gaussians using the exact upstream
integer top-inverse-render-weight rule. The CLIP field keeps all 2,235,426
Gaussians, has shape `[2235426,512]`, and occupies 4,578,152,576 bytes before
the separate geometry and weight files. This storage is charged in addition
to the DINO field.

## Real query results

Every query ran in a fresh one-query workspace and process and produced a
finite `float32[293811]` vector on the official ScanNet mesh domain.

| Modality | Query | Core elapsed | Probability range | Query-time forbidden input check |
|---|---|---:|---:|---|
| image | `uq_39841577b6880ed5c659f5c5259175a6` | 29.52 s | 0.3018–0.6989 | no labels/pairing |
| point_2d | `uq_198c34dd4d4d92331cf092ff552b71e7` | 6.85 s | 0.2870–0.7310 | no captured RGB/depth |
| point_3d | `uq_02d99bf10f60321357fd7b514b424cb1` | 7.13 s | 0.2961–0.7288 | no camera/RGB/depth |
| text | `uq_011b2bd84b180505147722064da9ba7a` | 18.92 s | 0.3696–0.7341 | no image/pose/pairing |

The constructor-side paired-prompt audit covers all 67 targets. Reprojecting
each private paired world point through its public 2-D prompt camera gives a
maximum click error of 0.000227 px (median 0.000058 px). Only these aggregate
construction diagnostics are released; adapters never receive the pairing.

All eight image queries in the scene were also run independently. Descriptor
cosines span `[-0.729, 0.980]`, probability-map correlations span
`[-0.972, 0.998]`, and mean per-vertex cross-query standard deviation is
`0.0899`, ruling out a constant-response plumbing failure. However, the
default fixed sigmoid selects 28.1%–74.3% of scene vertices at threshold 0.5.
This is evidence of response diversity, not segmentation quality, and points
to the remaining dev-only calibration requirement.

## Remaining release gates

The nine-scene geometry and two-field inventories must finish, all query
outputs must be sealed before private evaluation, and an external immutable
release commitment plus actual sandbox/runtime receipts must authorize the
method. Official AP/IoU/UQ-Mean are intentionally unavailable until those
gates pass; the pilot evaluator cannot be used as an adaptive private-label
oracle.
