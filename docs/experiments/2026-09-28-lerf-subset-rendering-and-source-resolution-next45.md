# LERF subset rendering and source resolution — next45

2026-09-28. Completed development experiments; no default promotion. The request to bring both LERF metrics to mainstream performance remains unmet.

## Result

Keeping the geometry, field, native query encoder and evaluator fixed, increasing source evidence resolution does not yield a material improvement. Jointly fitting probability and soft-subset opacity also fails to improve the matched next44 candidate. Do not replace next44 with these variants or select different variants per scene.

| Full four-scene macro | next44 soft identity | next45 higher resolution |
|---|---:|---:|
| Probability-rendered IoU | 53.914 | 53.982 |
| AP | 71.211 | 71.456 |
| Selected-element projection IoU | 37.710 | 37.680 |

| Scene | Probability IoU | Selected projection IoU |
|---|---:|---:|
| figurines | 54.415 → 54.326 | 33.700 → 33.570 |
| ramen | 42.004 → 42.133 | 24.052 → 24.296 |
| teatime | 64.562 → 64.457 | 51.213 → 50.914 |
| waldo_kitchen | 54.674 → 55.011 | 41.877 → 41.942 |

Two-scene matched control (figurines + ramen), percentages:

| Method | Probability IoU | AP | Selected projection IoU |
|---|---:|---:|---:|
| next44 reference | 48.209 | 64.895 | 28.876 |
| probability | 46.068 | 63.832 | 26.454 |
| joint | 45.785 | 63.622 | 27.972 |

## Mechanisms implemented

1. `fit_subset_readout_next45.py`: optimize transient element memberships using source probability agreement and the actual gsplat opacity renderer, with geometry/opacities held fixed as map state. A soft membership multiplies original opacity only during query rendering. Anneal the membership temperature from 1 to 0.2. Unsupported elements keep the prior. This is a differentiable relaxation, not exact hard-selection optimization. The probability-only arm uses the same optimizer and initialization and isolates the extra opacity objective. Both use 60 steps, learning rate 0.05 and prior weight 0.1. No target labels are optimization inputs.
2. `fuse_highres_evidence_next45.py`: regenerate source SAM3 masks, native foreground/context identity evidence and fixed Gaussian contribution operators at 192×256 instead of 96×128. Keep the next44 solver, native confidence formula, source frames and next38 prior unchanged. All four LERF scenes complete detection, identity extraction, fusion and frozen evaluation. This increases source projection precision, not geometric reconstruction resolution. The same 3×3 erosion has a smaller relative width, so this is not a pure sampling-resolution ablation.

The initial joint prototype allowed unsupported transient parameters into the silhouette loss although final outputs restored their prior. The final version fixes these parameters in the forward membership as well; both joint scenes were rerun. Superseded outputs are isolated under `superseded_unmasked_joint` and excluded from the summary.

## Interpretation

- Higher resolution gains only 0.068 points in probability IoU and loses 0.030 in selected IoU. Coarse source rasterization alone does not explain the large gap.
- The opacity objective improves selected projection relative to its probability-only optimizer control, but both remain below next44. Thus adding a silhouette objective is insufficient, and these experiments do not establish the best optimization schedule or refute all possible set losses.
- A source mask can have a crisp contour while describing the wrong semantic target or whole/part extent. Stronger fitting cannot repair this information. It can spread that error consistently across the scene. This remains a mechanism hypothesis, not a measured allocation of responsibility.
- The existing single membership value per Gaussian must explain competing source observations and potentially overlapping geometry. These experiments retain that limitation; they do not establish whether separating semantic supports or preserving multiple identity hypotheses is sufficient.
- Existing native foreground/context confidence is crop-level evidence. It is not independent proof of a part identity or cross-view physical correspondence. Resolution and raster losses do not repair this missing evidence.

The next method intervention should retain distinct source instance/part hypotheses through geometric association, with independent native identity and held-out-source consistency, before writing membership to the field. A missing or conflicting source extent must remain unknown rather than become a confident negative. Compare identity/ranking failures separately from high-AP/low-IoU extent failures. This is outstanding work, not an implemented result of next45.

## SOTA reference and comparison limits

[LUDVIG official repository](https://github.com/naver/ludvig) combines uplifted 2D features with 3D/DINOv2 affinities. [VALA](https://vala3d.github.io/) studies visibility-aware association and robust cosine aggregation. These motivate controlling which observations write semantics and preserving reliable identity through aggregation; they do not justify indiscriminate propagation of conflicting same-name regions. The new opacity objective here is our experiment, not a reproduction of either paper's loss.

The local LUDVIG wrapper/lock exists, but its specified `/root/baselines/LUDVIG` checkout is absent. The [official source](https://raw.githubusercontent.com/naver/ludvig/main/ludvig_clip.py) was therefore consulted: its evaluator includes an optional SAM path that opens the target image and adaptive relevancy thresholds. Not every configuration uses that path. We do not silently import target-image refinement or adaptive thresholds into the frozen strict-source protocol to match headline scores.

The selected-only metric historically called LERF3D in this project is a selected-geometry **2D reprojection** score, not physical 3D ground truth. Its GT, alpha rendering and threshold differ from probability-rendered LERF2D. Their numerical difference cannot all be attributed to the selector. Results here use existing frozen evaluators unchanged. No matched public SOTA replication or new mainstream-performance claim is established.

All candidates additionally revisit authorized source RGB with SAM3 and native crop encoding at query time; they are online enhanced candidates, not equal-budget field-only improvements. No target RGB is used to generate these predictions. Results are development evidence. No new defensive tests or audit infrastructure were introduced.

## Artifacts

- `output/subset_readout_next45/summary.json`: exact matched comparisons.
- `output/subset_readout_next45/highres_execution.json`: completed four-scene run.
- `output/subset_readout_next45/joint_final_execution.json`: corrected joint runs.
- `scripts/research/run_highres_lerf_next45.py` and `run_subset_readout_next45.py`: experiment entrypoints.
- All reported evaluations use `hierarchical_pipeline`, not the saved historical `retained_pipeline`.
