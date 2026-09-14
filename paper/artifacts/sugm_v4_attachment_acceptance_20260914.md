# Latest attachment implementation and acceptance audit — 2026-09-14

Authority: user attachment `/root/.codex/attachments/4b8c75c4-87e5-4b0e-8454-65a41d1addc5/pasted-text.txt`.
Status: **not fully implemented or validated**. The September 11 and 14 repairs are partial. No claim that all bugs, geometry, identity, association or deployment have been solved is justified.

## Acceptance inventory

| Attachment requirement | Actual state | Missing acceptance |
| --- | --- | --- |
| Geometry / visibility | Sparse carrier and coarse registration tested; new native-resolution support audit completed | Resolution-consistent footprint, boundary leakage, cross-view visibility and source-mask round trips at final output resolution |
| Correct summary semantics | Invalid development entry quarantined; genuine crop summary implementation repaired and real sentinels reproduced | Native crop with paired text retrieval accuracy and cross-view identity evaluation |
| Local and object evidence | Raw fragment evidence and region keys retained; local-semantic and object paths exist | Verified part/material/background, multiple-instance, absent-target and ambiguous query cases |
| Reliable object association | Full versus geometry-only grouping trained on 12/4 ScanNet split and transferred to four LERF scenes | Real fragment errors, same-object disjoint parts, consistent descriptor domain, merge/fragmentation tradeoff |
| Completion under noisy identity | Synthetic noise and ambiguous-input handling implemented | Full soft memberships, actual association errors, object-equal small-object precision and supported-surface completion |
| Shared query output | Canonical sum composition and independent-process text evaluation exist | Three query types, cold reload equality, same element posterior consumed by final native-resolution outputs |
| Public evaluator | Compatible OpenGaussian metric adapter implemented, tested, replayed on 208 retained object-frame predictions | Current v4 native-resolution exported masks and source/rendering protocol audit |
| Compression / joint calibration | Deferred | Only after the functional chain has evidence; no fixed dimension requirement introduced |

## New repairs and verification

1. `OfficialCropSummaryRuntime` previously fetched backbone by version name while loading its summary head from an explicit path. It now loads both from that path and selects the genuine summary position from checkpoint teacher metadata. This closes a possible model identity mismatch. Checkpoint binding and teacher-order tests passed.
2. Reencoded three deterministic proposals (masked + context) from the first source frame of each of four scenes. All **24 descriptors were exactly equal** to existing half-precision cache values (maximum absolute error 0). Peak PyTorch allocation was 1,422,656,512 bytes. This is a real backbone run on GPU 2, but not a retrieval-accuracy test. Inputs were hash-checked; no labels/text were read.
3. Fixed annotation parsing for disjoint polygons of different lengths. Reject nonfinite coordinates, missing/invalid dimensions and mixed dimensions that the current global-scale interface cannot represent. Previously the latter could silently reuse the last frame's dimensions. This prevents incorrect scores; per-frame dimension support remains an extension if needed.
4. Added an OpenGaussian-compatible mask scorer: fixed public frames, binary threshold >10, missing predictions scored zero, observation-equal mean and strict accuracy cutoffs. It validates shapes rather than accepting accidental array broadcasting. It scores exports and does not certify upstream selection/rendering.
5. Full v4 plus relevant interface/protocol regression: **270 passed, 2 skipped**. The final metadata-only feature-mode receipt addition does not alter numerical behavior.

## Four-scene transfer: removing appearance did not improve the method

Full and geometry-only checkpoints from the previous turn were independently loaded, built and evaluated under identical source memory, association configuration, text queries, pixel threshold and coarse raster. Physical GPUs 2 and 3 were assigned to the two runs; construction/projection still use CPU code. The grouping model's input mask does **not** remove appearance from the downstream fixed assignment formula or text prototypes.

| Scene | Full grouping, primary mIoU | Geometry-only grouping, primary mIoU | Hypothesis count full → geometry |
| --- | ---: | ---: | ---: |
| figurines | 0.078612 | 0.065869 | 307 → 212 |
| ramen | 0.053050 | 0.049714 | 202 → 117 |
| teatime | 0.086874 | 0.081663 | 282 → 178 |
| waldo_kitchen | 0.130656 | 0.128643 | 191 → 154 |
| Scene mean | **0.087298** | **0.081472** | |

Primary = generic-negative distinct-fragment consensus, all-token canonical composition, observed membership, fixed threshold 0.2. These are coarse-grid category-macro development metrics, not public LERF scores. No LERF threshold selection was performed. Full grouping reproduces the previous result. Every scene regresses with geometry-only grouping. This rejects simple removal of appearance as the current fix; it does not prove descriptor domain shift is irrelevant.

The ScanNet validation split selected decision thresholds, so its reported validation metrics are development estimates, not independent test evidence.

## New critical finding: current radius-one projection does not support native resolution

Rendered an all-one element field with the unchanged frozen carrier at each public GT frame's native resolution. A query cannot predict outside the projection support. For each object, `intersection(all-surface support, GT) / area(GT)` is therefore an optimistic IoU upper bound for this fixed projection, even with perfect identity and no false positives.

| Scene | Native raster | Mean object-observation support IoU upper bound |
| --- | --- | ---: |
| figurines | 728 × 986 | 0.114201 |
| ramen | 731 × 988 | 0.124856 |
| teatime | 730 × 988 | 0.158789 |
| waldo_kitchen | 725 × 985 | 0.031543 |

The carrier computes a depth/intrinsics-dependent voxel footprint, then clips its radius to one pixel. A cap used at the coarse feature raster cannot retain equivalent support at native resolution. The support audit proves a severe limitation of the current native-resolution projection; it does not establish that changing radius alone will fix object quality. Larger footprints can increase foreground/background mixing, and projection changes invalidate source assignment assumptions unless rebuilding or replaying source evidence confirms consistency.

This corrects the earlier overconfident implication that geometry could be treated as closed while only object association remained. Do not seek 40–50% mIoU from the current direct native-resolution projection without addressing this bound.

## Public scorer replay

Replayed retained VALA predictions and GT from the August 1 protocol reproduction. No missing masks among 208 observations. Scene mIoU: figurines 0.583477, ramen 0.440339, teatime 0.694411, waldo_kitchen 0.446769. These reproduce the retained record's rounding (58.35/44.03/69.44/44.68%). They are **historical comparator predictions**, not new v4 results. Matching these scores validates this metric implementation on real masks, not the entire current method.

Reference inspected: https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/scripts/compute_lerf_iou.py . Two-dimensional GT can legally evaluate rendered 3D selection; there is no blanket element-GT prerequisite.

## Ordered next work and experimental requirements

1. **Projection first:** specify a reference image scale or physically projected surface footprint, keep depth ordering, and measure native-resolution support, cross-view alignment and boundary leakage. Use source masks/synthetic planes for choosing the rule. Before method promotion, rebuild source evidence under the same projection contract. A larger arbitrary mask dilation is not evidence of correct membership.
2. **Identity in parallel:** annotated source-crop paired-text ranking; then the same queries against fragments and object hypotheses. Report per-query ranks, difficult negatives and unavailable targets. A byte-identical crop cache does not establish identity accuracy.
3. **Association:** align training/deployment region descriptors or train on actual source fragment relations; keep reversible raw evidence. Test part/whole positives and genuine negative relations. Measure splitting and merging, as fewer hypotheses did not improve this round's query result.
4. **Completion:** first handle already reconstructed/source-supported surfaces. Train with deployed association errors, preserve soft/unknown evidence, and report object-equal precision/recall, small-object performance and false-positive growth. Defer guesses about never-observed surfaces.
5. **End-to-end acceptance:** freeze one configuration before evaluation; save and cold reload; test text/image/signed prompts and no-match/local queries; export 3D selection and its rendered native masks; score all four scenes through the public adapter. Report scene and observation aggregation explicitly. Keep validation used for tuning separate from a final held-out confirmation.

Artifacts: `/mnt/pool/sqy/results/RADIO-GS/output/v4_affinity_lerf_transfer_20260914/` contains `full_v2/`, `geometry_only_v2/`, `native_crop_reencoding.json`, `native_coverage_*.json`, and `public_metric_replay_*.json`. Each transfer directory contains exact commands, cold-loaded memory, extent ceilings and text reports. Initial non-v2 transfer directories preserve a CLI mismatch failure (`maximum_fragment_hypotheses` is a fixed capacity, not a builder flag), corrected before successful reruns.
