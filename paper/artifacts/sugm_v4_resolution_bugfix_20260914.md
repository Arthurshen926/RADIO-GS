# Resolution bug fix — 2026-09-14

The active repair is a source-resolution-aware surface projection. It fixes a concrete output failure, not the remaining semantic/object model quality.

## Defect and change

`SurfaceVoxelCarrier` capped a surface footprint at one **output** pixel, although evidence was constructed at a 46 × 62 reference raster. At approximately 730 × 990 output resolution this shrank its supported image area drastically. The existing projected-voxel radius formula did not help because the output-pixel clamp removed the scale dependence.

The carrier now accepts `reference_raster_shape`. It quantizes and caps the footprint in this mapping raster, scales the footprint independently along the output x/y axes, and retains its depth ordering, surface band and contributor limit. Same-reference projection is numerically identical. Historical configurations without the new field keep their previous behavior.

The optional reference shape is serialized by `SurfaceCarrierConfiguration`, checked against geometry receipt metadata, and restored by the normal bundle and LERF state loaders. Old configurations omit the optional field in their content digest to preserve historical bundle identities.

## Actual verification

- Synthetic resolution-change, nonuniform resize and foreground/background occlusion tests pass.
- All **128 source views** (32 per scene) reproduce exactly: element IDs, pixel IDs, depths and weights. Existing source evidence therefore remains applicable at its construction raster.
- Four scenes were evaluated with the same saved text queries, primary generic-negative fragment-consensus selector, all-token canonical composition, observed membership, threshold 0.2 and native GT dimensions. The old and fixed renderers consumed exactly the same element posterior.
- Exported binary PNGs were scored with the verified OpenGaussian-compatible scorer. All public object-frame observations were included. This is a development run; it does not certify all upstream benchmark assumptions.
- A separate process reopened all four sealed states/hypotheses through the normal loaders, restored the reference scale, and recomputed the complete text element posterior. Every value exactly matched the posterior used for the native-mask evaluation.
- Full v4 regression: **258 passed, 1 skipped**. Focused carrier/bundle tests: **28 passed**. `git diff --check` passed.

| Scene | Old native mIoU | Fixed native mIoU | Old support upper bound | Fixed support upper bound |
| --- | ---: | ---: | ---: | ---: |
| figurines | 0.010214 | 0.061825 | 0.114201 | 0.999199 |
| ramen | 0.013477 | 0.039958 | 0.124856 | 0.999999 |
| teatime | 0.020211 | 0.074950 | 0.158789 | 0.998564 |
| waldo_kitchen | 0.008930 | 0.122996 | 0.031543 | 0.894266 |
| Scene-equal mean | **0.013208** | **0.074932** | | |

The support bound uses an all-one surface field and perfect hypothetical selection of supported target pixels. It is not actual segmentation accuracy, purity, or evidence that enlarged footprints respect every physical boundary. These native-resolution numbers must not be compared as a direct delta to the earlier 8.73% coarse-grid/category-macro metric.

## Persisted inputs for subsequent work

Use the repaired scene-state and hypothesis paths from these manifests for subsequent native-resolution evaluation:

- `/mnt/pool/sqy/results/RADIO-GS/output/v4_resolution_repair_20260914/figurines/sealed/manifest.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_resolution_repair_20260914/ramen/sealed/manifest.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_resolution_repair_20260914/teatime/sealed/manifest.json`
- `/mnt/pool/sqy/results/RADIO-GS/output/v4_resolution_repair_20260914/waldo_kitchen/sealed/manifest.json`

Each repaired state records its original-state digest and validation-report digest. Fragment and hypothesis parent hashes were rebound after exact source replay, without changing membership or prototype tensors. The original artifacts remain available. Each scene directory also contains the old/fixed native masks, shared `query_posterior.pt` and complete `report.json`.

The native renderer currently uses the carrier's CPU projection/index-add implementation. GPUs 2 and 3 were assigned to the two scene groups for text scoring; no claim that the full projection was GPU accelerated is made.

## Remaining serious limitations

Actual mIoU remains only 7.49%. This repair removes the demonstrated resolution-dependent support ceiling but leaves incorrect identity ranking, fragmented/merged objects, coarse mapping-time boundary mixtures and noisy-association completion unresolved. Enlarged footprints may expose those errors over more pixels. Continue with fixed-posterior/source-mask boundary diagnostics and original crop → fragment → object retrieval isolation using these repaired inputs. Do not interpret the near-one support bound as completed geometry or declare the method restored to historical benchmark quality.
