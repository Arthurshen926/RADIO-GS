# Query-free regions and equal-count element-selection control

The previous full-four-scene source-caption region readout scored poorly for queries outside the caption route. This round tests a different, query-independent source inventory: SAM proposals extracted without a text prompt and their original SigLIP descriptors. The same frozen MoGe-bound Gaussian carrier, deployed text encoder, source-fitted threshold, query set, target frames, rendering and scoring code are used. Only source `train_frames` contribute proposal evidence. Benchmark query text is consumed at inference to form scores; target RGB or GT never enters score construction or parameter selection.

## Four-scene result, 208 target observations per arm

| Arm, scene-equal mean | Native-grid 2D IoU % | AP % | Selected-element render IoU % |
|---|---:|---:|---:|
| Frozen final field | **49.7408** | **68.4795** | **33.5676** |
| Query-free source regions, soft query-first averaging | 27.4596 | 65.1516 | 31.9752 |
| Query-free source regions, binary region gate | 28.4558 | 64.9653 | 31.3708 |
| Replace only 76 observations outside caption routing | 43.7661 | 66.5445 | 33.0601 |

The query-free proposals recover considerably more ranking and element-selection ability than the previous caption-region control (AP 62.2412→65.1516%; selected 27.5821→31.9752%), but none of the query-free arms beats the frozen final field jointly. Their thresholded 2D masks remain especially poor in ramen. Therefore a larger region inventory or original high-dimensional descriptor is not itself a performance fix.

To separate threshold effects from element ranking, a second fixed control shifts only each query's proposal-score logits so its number of foreground elements equals the frozen final field's count. This count is an inference-time prior from the existing map; no benchmark labels, target RGB, scene-specific winners, or parameter search are involved. Element ranking is preserved by the logit shift; rendered-pixel ranking can change after Gaussian composition. Counts match exactly for 65/67 queries; the other two have tied proposal scores and are explicitly recorded.

| Arm | Native-grid 2D IoU % | AP % | Selected-element render IoU % |
|---|---:|---:|---:|
| Query-free region, original threshold | 27.4596 | 65.1516 | 31.9752 |
| Same region ranking, parent foreground-element count | 38.6487 | 64.6938 | 28.2107 |
| Count-matched readout only outside caption routing | 44.8219 | 66.4082 | 32.2930 |
| Frozen final field | **49.7408** | **68.4795** | **33.5676** |

Matching the number of selected elements recovers **11.19 percentage points** of 2D IoU, confirming a large calibration mismatch. Yet it still falls **11.09 points** below the final field, and its selected-element rendering falls **5.36 points** below final. The residual is a spatial/semantic assignment problem: the new score ranks and selects the wrong elements, even when it selects approximately the right number. More threshold fitting cannot by itself close that gap. This result also explains why a target-IoU-tuned threshold would be misleading.

The two source inventories answer different questions. Caption-grounded regions depend on incomplete source phrase discovery and often merge a named whole with parts or different instances. Query-free proposals do not require those phrases and improve AP relative to caption regions, but a descriptor extracted from a proposal still has to be linked to the right physical element and semantic scope. The original field's source-trained threshold was fitted to fused element features, not to individual crop responses. Both the threshold transfer and query-to-element assignment must be checked, with source views and concepts excluded from hypothesis construction. The full target results here are development diagnostics, not an independent final-test estimate.

All four frozen baseline probability masks replay pixel-exactly; selected-object baseline records match, and all 1773 frozen source hashes remain unchanged. Two targeted equal-count tests pass. Both controls are rejected for deployment. This round does not alter ScanNet OVS/NVOS or UQIS. The selected-element render metric is not volumetric 3D IoU.

Machine-readable full cohorts: `output/query_free_region_readout_next26/summary.json` and `output/volume_matched_region_next27/summary.json`; receipts: `paper/artifacts/query_free_region_next26_complete.json` and `paper/artifacts/volume_matched_region_next27_complete.json`.
