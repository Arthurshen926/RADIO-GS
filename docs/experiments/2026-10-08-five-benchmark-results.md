# Complete five-benchmark architecture baseline

The fixed common physical architecture has now completed all five local benchmark families. This is a high-capacity development baseline, with no per-scene/input checkpoint selection. Full coverage precedes further architecture changes or staged error attribution.

[Protocol](2026-10-08-five-benchmark-architecture-protocol.md) · [Machine-readable results](2026-10-08-five-benchmark-results.json)

## Complete comparison

Values are percent; delta is candidate minus saved v10 in percentage points. The table is a system comparison with the input/teacher differences below, not an isolated architectural causal effect.

| Task | Metric | Candidate | Saved v10 | Delta (pp) | Cohort |
|---|---|---:|---:|---:|---|
| lerf2d | member response IoU | 49.4130 | 70.4007 | -20.9876 | 4 scenes |
| lerf2d | member response AP | 71.7326 | 87.6193 | -15.8866 | 4 scenes |
| lerf3d_local_projection | actual subset projection IoU | 32.3130 | 66.9633 | -34.6503 | 4 scenes |
| scannet_ovs | member mIoU 19 | 22.2853 | 61.0115 | -38.7262 | 10 rooms |
| scannet_ovs | member mIoU 15 | 23.3694 | 62.5276 | -39.1582 | 10 rooms |
| scannet_ovs | member mIoU 10 | 26.5462 | 72.4861 | -45.9399 | 10 rooms |
| nvos | signed response IoU | 57.6865 | 86.5134 | -28.8270 | 8 cases |
| nvos | signed response AP | 94.8988 | 98.0222 | -3.1234 | 8 cases |
| uqis/text | AP | 21.3937 | 26.5959 | -5.2022 | 9 rooms,67 targets |
| uqis/text | actual subset IoU | 12.0755 | 16.3865 | -4.3111 | 9 rooms,67 targets |
| uqis/image | AP | 24.7530 | 59.3088 | -34.5558 | 9 rooms,67 targets |
| uqis/image | actual subset IoU | 15.9861 | 39.8980 | -23.9119 | 9 rooms,67 targets |
| uqis/point_2d | AP | 47.9527 | 75.7901 | -27.8374 | 9 rooms,67 targets |
| uqis/point_2d | actual subset IoU | 28.3732 | 49.7357 | -21.3625 | 9 rooms,67 targets |
| uqis/point_3d | AP | 46.7809 | 75.3253 | -28.5443 | 9 rooms,67 targets |
| uqis/point_3d | actual subset IoU | 27.6913 | 49.0470 | -21.3557 | 9 rooms,67 targets |

All eight NVOS cases and all268 UQIS queries remain in the denominator. Unsupported UQIS registration/observation counts: {'text': 0, 'image': 0, 'point_2d': 6, 'point_3d': 8}. Empty outputs are scored, not removed.

Native controls are separate from the common member result: LERF native IoU/AP 43.8943/74.6332%; ScanNet native19/15/10 mIoU 36.9452/37.7524/44.1994%.

NVOS actual selected-subset projection IoU is 70.5360%, reported separately from its retained normalized-response benchmark score.

## Contracts and scope

- New fields18 training/2 development source views; retained LERF18 and fern12; v10 generally more observations.
- SigLIP/DINO/query-free SAM only; v10 ScanNet includes MaskAdapter and v10 UQIS adds labelled3D member training.
- One scene Gaussian field; no query-time source images, masks or tracks.
- LERF/NVOS saved geometry; ScanNet saved unlabelled-mesh Gaussians; UQIS same official unlabelled mesh converted to Gaussians.
- Same member core; typed category competition and fixed mesh transport. LERF3D is retained subset projection, not independent3D truth.
- Raw signed scribbles versus v10 positive-derived SAM reference/source tracks.

The reused four-scene LERF predictions are hash-bound to the accepted fields; fern is replayed against its saved output. All new cohort predictions are persisted before new scoring. The UQIS evaluator retains its original67 targets, all four modalities and hidden target pairing. Gaussian scores/selected rows are also saved before mesh transport.

New construction contains 25 fields, with horns left/centre sharing a field. Existing four LERF fields and fern are reused. No compression, latency tuning, support-mixture search or negative-rule experiment was conducted.

The only post-freeze integration correction accepts the official trex scalar scribbles stored as three identical RGB channels. It collapses exact replicas without changing evidence or the member method. The failed attempt, original implementation hashes and correction receipt are retained. The original integration suite passed 22 tests; all 8 signed-raster tests pass after the new regression case, and fern reproduces all four saved output tensors exactly.

## Next: attribute the measured gaps

The complete run supplies four initial observations; these do not establish a
single causal explanation:

- ScanNet native 19-class mIoU is 36.9452%, versus member/category output
  22.2853%. The member reader is lower in all ten rooms under the same field,
  query menu and mesh transport. Its additional loss is 14.6599pp; the native
  control itself remains 24.0663pp below v10 under different teachers and source
  budgets. Separate these two questions before changing the shared architecture.
- NVOS flower/orchids response IoU is 7.7530/9.2527%, while AP remains
  91.5207/93.0105%. Their actual subset projection IoU is 34.8390/23.1392%.
  Ranking, member-set extent and rendered response calibration need separate
  inspection. High AP does not validate the selected set, and these results do
  not authorize threshold fitting against the evaluated targets.
- The 6 unsupported 2D and 8 unsupported 3D UQIS prompts account for explicit
  empty outputs. Whole-scene observed mesh coverage ranges from 68.74% to 94.38%
  in UQIS and from 62.59% to 94.47% in ScanNet. Target-specific missing coverage
  must be measured before attributing point-query failures to member learning.
- ScanNet mesh correspondence has zero centre displacement at every evaluated
  vertex. UQIS correspondence is also zero through the 99th percentile in every
  room; two rooms have small nonzero maxima (1.34cm and 2.64cm). This does not
  establish perfect geometry or rendering, but it gives no evidence that a large
  mesh-coordinate displacement explains the full metric gap. UQIS same-target
  all-four IoU>=0.5 success is only 6.9444% under scene-macro averaging.

Use this fixed baseline to isolate source/geometry coverage, native observation fusion, query identity, member extent, and output projection/transport in that order. Save the complete cohort while changing one stage at a time; do not let local score gates prevent reporting other benchmarks. These are follow-up diagnostic questions, not causal conclusions from the current system table.

The existing development scenes and queries have been opened historically. No independent final-test, v10-equivalent supervision, compactness, or overall accuracy-promotion claim is made.
