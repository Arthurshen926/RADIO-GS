# Protected native locality: measured gains and a failed cross-input gate

Parent `b3b934b`, branch `codex/query-preserving-gaussian-field`. Fixed
[protocol](2026-10-07-native-local-member-protocol.md) and
[machine-readable evidence](2026-10-07-native-local-evidence.json).

## Decision

Keep the fitted physical metric as the common research candidate. A native
DINO directional metric improves real text subsets and, after refitting, image
diagnostics on both scenes. It fails the predeclared common-input gate because
ramen's 3D teacher-region point mean regresses by 0.5225 percentage points. The
frozen intervention has stronger text scores but substantial image/point losses.
Neither is promoted, no third-scene native-metric fit is launched, and no
modality-specific mixture or strength/threshold sweep follows this result.

The implementation remains available as the explicit research option
`training.local_metric: native_cosine`. Its coefficient is fixed; geometry,
native coordinates, observed members, eligible input anchors, source masks and
sampling episodes are held constant. It adds no trainable parameters or stored
tensors. It uses normalized **decoded** native support directions and paired
cosine scoring, so affine codecs keep the same metric. The existing default
`none` preserves the previous member decoder and old exports.

The metric has a simple common representation: for normalized group and native
local directions, their concatenation divided by sqrt(2) has cosine equal to
half the sum of their two cosines. The intervention therefore retains a fixed
native component alongside learned group directions in one member energy. It
does not add an image/text fallback or recover complete query uncertainty.
This algebra and module organization alone are not a novelty claim.

## Causal diagnostics before the intervention

On the fixed physical-metric fields, real selected projection IoU (%):

| Scene | Union member | Single anchor | Native-only diagnostic |
|---|---:|---:|---:|
| figurines | 41.7884 | 50.1398 | 46.4535 |
| ramen | 19.4642 | 22.0829 | 25.4241 |

The original union arrays replay exactly. Single-anchor selection undercounts
genuine category unions; the native-only output bypasses the member core. These
controls expose remaining losses but are not alternative deployed branches.
They do not independently establish ground-truth identity correctness.

The train-only exclusion audit covers all 1,455 / 1,320 eligible source episodes
and renders 32 evenly indexed episodes per scene using fixed median supports
and privileged source-mask scales. Adding disjoint masks of other sizes would
increase mean known-negative pixels from 2,703 / 4,459 by 4,296 / 3,599. Yet these
new pixels account for only 9.54 / 9.35% of sampled false additions; 75.22 / 81.51%
lie in the remaining unknown region. These are means over sampled episodes,
including zero-error episodes, and concern source SAM masks, not benchmark truth.
We therefore leave supervision unchanged instead of treating unknown pixels as
background. The audit motivates a locality study; it does not prove its success.

## Fixed local-metric comparison

All four text prediction packages precede new LERF scoring. Query vectors,
geometry and native responses are exact; the retained alpha-byte threshold of
10 and semantic scorer are unchanged. Values below are actual selected-Gaussian
**projection** IoU %, not independent 3D ground-truth accuracy.

| Scene | Physical reference | Native metric, frozen | Native metric, refitted | v10 |
|---|---:|---:|---:|---:|
| figurines | 41.7884 | 59.4702 | 47.6108 | 71.1139 |
| ramen | 19.4642 | 41.5179 | 26.9443 | 58.3074 |

The refitted candidate closes a further 5.8224 / 7.4801 points of the same-scene
v10 gap, leaving 23.5030 / 31.3631 points. v10 uses different source-access and
resource conditions. The frozen candidate's smaller text gap is not a shared
system improvement because it fails other input conditions.

| Scene / input | Physical reference | Frozen | Refitted |
|---|---:|---:|---:|
| figurines / region image | 23.6868 | 14.8723 | 24.6490 |
| figurines / 2D point | 37.4463 | 31.3454 | 41.0447 |
| figurines / 3D point | 32.2285 | 25.3407 | 37.1973 |
| ramen / region image | 31.7428 | 23.5224 | 33.7722 |
| ramen / 2D point | 40.3042 | 38.9684 | 40.7514 |
| ramen / 3D point | 38.8454 | 36.6613 | **38.3229** |

All 169 / 149 development regions and unsupported cases are retained. Region
images use source-SAM crop descriptors plus interior patches; 3D points are
dominant Gaussians derived from region-interior pixels. These are diagnostics,
not official reference-image or human 3D-point benchmarks. Ramen has 75 improved,
69 regressed and 5 unchanged 3D cases; the largest case drops 52.88 -> 10.97%.
The small negative scene mean must not hide these large individual failures.

Member-response semantic IoU (%), reference -> frozen -> refitted:
figurines 49.1988 -> 25.6557 -> 59.5939;
ramen 42.8233 -> 34.5528 -> 51.3901. Native semantic IoU stays exactly
47.2877 / 35.1404. A good hard-set projection and a good semantic response image
remain different requirements; the frozen control illustrates their divergence.

## Scope audit and its limits

With the fitted scalar parameters and default scale fixed, the native metric
has the member-logit bound `L <= T(cos_DINO - b) - d²/(2s²)` because learned
group cosine is at most one. The post-evaluation audit checks this bound on
observed dominant Gaussians of teacher foreground pixels for every 3D case.
It quantifies target directions the current scalar/native prior cannot include
even with maximal learned group similarity.

This is **not** a bound on rendered-mask recall: other selected Gaussians can
cover the same pixel. It is also not a proof that the native prior alone causes
the mean regression; temperature/bias changed during fitting, and exclusion
fractions are similar in improved and regressed cases. Both unknown supervision
and variation within a target remain relevant. A constant local appearance
penalty is not established as an adequate object-completeness model.

## Complete four-scene physical-metric reference

Waldo kitchen was constructed independently of the local-metric outcome, using
the fixed eighteen-source-view, 2,000-step physical recipe. Development frames
46/142 and uniformly selected training frames exclude all five public target
frames. Its field has 1,496,561 Gaussians, 1,112,269 observed elements and 46,579
eligible anchors; 1,247 training episodes. No legacy source-query calibration
or source-query solver is reused.

| Scene | Native semantic IoU | Member semantic IoU | Actual subset projection IoU | v10 subset |
|---|---:|---:|---:|---:|
| figurines | 47.2877 | 49.1988 | 41.7884 | 71.1139 |
| ramen | 35.1404 | 42.8233 | 19.4642 | 58.3074 |
| teatime | 48.6891 | 63.5209 | 32.8006 | 75.0496 |
| waldo_kitchen | 44.4599 | 42.1092 | 35.1987 | 63.3822 |
| Equal-scene mean | **43.8943** | **49.4130** | **32.3130** | **66.9633** |

This table uses the same physical method in every scene. It does not substitute
new native-metric results for whichever scenes improved. The remaining actual
subset gap is 34.6503 points. This completes the declared four-scene LERF
development cohort, not five benchmark families or independent final testing.
Waldo's 158 region diagnostics are image/2D/3D 33.1919 / 40.6422 / 38.1419%.

## Integrity, reproduction and next work

Run with the existing cybersim_agent Python, `PYTHONPATH=.`, and an explicitly
selected CUDA device. Configs are `*_native_local_metric.json`,
`native_local_metric_study.json`, `physical_readout_study.json`, and
`waldo_kitchen_{spatial_metric,public}.json` under `configs/query_field/`.
Fitting/evaluation use `python -m radio_gs.query_field`; matched packaging,
prediction and separate scoring use `python -m radio_gs.query_field.metric_study`.
The waldo source compiler and public predictor use their existing entry points.
Use fresh output directories; sealed outputs are immutable.

29 targeted tests pass. Both refitted native models and the new waldo physical
model pass actual four-input encoder/adaptor checks and source-denied copied
export replay. Full protected tensors and logged sampled-episode sequences are
exact for both refits; frozen controls retain every tensor. Native local
matching increases computation without adding stored tensors. Reported cold
fixture timings include transfer/hash work and shared-GPU noise; they are not
stable latency benchmarks. The waldo package is 7,174,517,428 bytes, excluding
shared encoders. No compactness claim is made.

Original-worktree branch, HEAD and index remain exact; parallel paper work is
preserved. All jobs are complete, with no third-scene native-metric run or
unreported hyperparameter search. `gh` remains unavailable; no remote issue/PR
was created.

The next decision needs real signed/registered point evidence and a model of
how local appearance variation changes with target scope. The present audit
does not justify a particular adaptive bandwidth or prove that such a change
will succeed. Preserve the physical reference, the native candidate's genuine
text/image gains, and its point failures as separate evidence. Continue the
pending ScanNet OVS/NVOS/UQIS integrations without source-assisted or labelled
legacy components entering the standard field implicitly.
