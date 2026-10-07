# Physical scale inside the shared Gaussian member energy

Protocol: [fixed two-arm study](2026-10-07-physical-scale-member-protocol.md).
Parent: `990d2c9`, branch `codex/query-preserving-gaussian-field`.
All work remains in `/root/RADIO-GS-query-field`.

## Mechanism and architectural change

The previous scale-conditioned group kernel had no spatial term. Two distant
Gaussians with the same group feature had the same membership score. A training
support mask improved the input identity domain but could not repair this output
extent failure. The new opt-in kernel combines the existing learned group energy
with a Gaussian log prior on physical distance to the weighted support centre:

`L(i,S,s) = T [cos(g_s(i), prototype_s(S)) - b]
            - ||x_i - mean_weighted(x_S)||² / [2(s u)²]`.

`u` is reconstruction units per metre, recorded by the observation compiler and
now stored in the deployment specification. A 2D ray retains all weighted
contributors. Native inferred anchors and registered supports enter the same
kernel, and negative supports use the same rule. Threshold, candidate count,
native identity ranking, union admission and full observed output carrier are
unchanged. Exact registered 3D constraints retain their declared semantics.

There are no additional learned parameters, per-query thresholds, source views,
teacher families or 3D labels. Existing temperature and bias can adapt during
training. At threshold 0.5, unconstrained members satisfy the upper distance
bound `d < s u sqrt(2T(1-b))` when `b < 1`, since the group cosine cannot exceed
one. This makes physical extent explicit; it does not prove object completeness
or geodesic connectivity. A Euclidean prior may truncate elongated objects or
misrepresent an ambiguous-depth ray. This standard geometric prior is a useful
architectural correction, not a standalone top-conference novelty claim.

## Fixed two-scene comparison

Text results below use actual selected Gaussian rows re-rendered by the unchanged
historical LERF scorer, with accumulated alpha byte > 10. They are 2D projections
of 3D selections, not independent 3D ground truth. All four prediction packages
were complete before any new LERF scoring. Identical cached native query vectors
were used; native responses and geometry replay exactly.

| Scene | Previous supported anchors | Spatial term, frozen group | Spatial term, refitted group | Saved v10 |
|---|---:|---:|---:|---:|
| figurines | 14.6841 | 27.8987 | **41.7884** | 71.1139 |
| ramen | 9.6614 | 15.2633 | **19.4642** | 58.3074 |

Values are IoU %. The refitted arm closes 27.1043 / 9.8028 percentage points of
the same-scene v10 gap, leaving 29.3255 / 38.8432 points. v10 is source-assisted
and has different construction resources; this measures distance to the local
target, not equal-resource superiority or five-task equivalence.

The default-scale source-SAM development diagnostics retain every region and
count unsupported prompts as zero. These image inputs are masked-crop SigLIP
plus interior DINO patches, not an external reference-image benchmark. Points
come from fixed region interiors and projected geometry, not human annotations.

| Scene / input | Previous | Frozen spatial term | Refitted spatial term |
|---|---:|---:|---:|
| figurines / image | 23.0668 | 21.9195 | **23.6868** |
| figurines / 2D point | 31.6277 | 36.8304 | **37.4463** |
| figurines / 3D point | 27.4599 | 30.2871 | **32.2285** |
| ramen / image | 26.3283 | 29.6466 | **31.7428** |
| ramen / 2D point | 31.7173 | 39.3700 | **40.3042** |
| ramen / 3D point | 29.5687 | 37.1253 | **38.8454** |

The frozen arm fails the predefined image nonregression gate on figurines. The
refitted arm passes every required metric on both scenes and is nominated for
teatime confirmation with exactly the same rule. This distinction matters:
adding a post-fit spatial restriction alone is not the accepted common solution.

## Third-scene confirmation

The nominated refitted arm was applied to the existing teatime eighteen-view
cache, with the same 2,000 steps and all hyperparameters unchanged. All 14 text
queries (59 query/view cases) were predicted before scoring. Actual selected
projection IoU rises **21.9424 -> 32.8006%**; saved v10 is 75.0496%, so the gap
shrinks by 10.8581 points and remains 42.2491 points. This is a previously opened
development scene, not an independent final test.

Default-scale teacher-region image / 2D point / 3D point IoU (%) rises from
27.7675 / 30.5706 / 29.4113 to **32.8021 / 36.3325 / 34.2485** across all 127
regions. Unsupported cases remain zero. The same common arm therefore passes
the declared text, image and point mean gates on all three scenes.

Teatime's native semantic IoU stays exactly 48.6891%; member-response semantic
IoU rises 58.9434 -> 63.5209%. Actual set precision/recall is 33.5376 / 99.0343%:
the high recall still coexists with substantial false additions. No claim of
complete extent recovery follows from passing these local gates.

## Error decomposition and remaining limits

For the refitted model, actual text-mask precision/recall (%): figurines
42.1660 / 98.2269, ramen 20.7656 / 84.0656. The previous values were approximately
14.74 / 97.84 and 10.20 / 86.95. Far fewer false additions improve the real subset;
ramen also loses some recall. Nearby unrelated members remain a major problem.
Every decomposed IoU is checked against the retained threshold-10 scorer.

Native semantic IoU stays exactly 47.2877 / 35.1404. Member-response semantic IoU
is 49.1988 / 42.8233, versus the previous 47.4479 / 44.2349: ramen's semantic
member readout regresses even though its actual selected subset improves. These
are different outputs; the local gate does not claim that every readout wins.
The older native-only subset diagnostic remains higher at 46.4535 / 25.4241,
but bypasses the common member kernel and is not deployed as a hidden fallback.

## Reproduction and integrity

Use the existing cybersim_agent environment with `PYTHONPATH=.` and a selected
CUDA device. The configurations are `figurines_spatial_metric.json`,
`ramen_spatial_metric.json`, and `spatial_metric_study.json` under
`configs/query_field/`. Fresh run roots point their `observations` directory at
the exact previous eighteen-view compiled cache; existing outputs are immutable.

1. `python -m radio_gs.query_field fit --config <scene-config> --output <fresh-root>`
2. `python -m radio_gs.query_field.metric_study package --config <study-config> --job <frozen-job> --output <public-root>`
3. Predict every fixed job using the same module's `predict` stage.
4. Evaluate each model with `python -m radio_gs.query_field evaluate`; score all
   public jobs with the metric study's separate `score` stage.
5. Run `subset_errors`, the four-input `smoke` check and the evidence collector.

Frozen controls retain every parent tensor bit for bit. Refitted models retain
all native/geometry/observed/eligible tensors exactly and use the same logged
sampled episode sequence as the original fit. Full deployment tensor bytes are
unchanged from the supported-anchor models; only manifest metadata grows.
Old exports without the new specification fields replay their previous scores.
The original worktree's branch, HEAD and index remain unchanged; parallel paper
edits there are preserved. No remote issue or PR was created (`gh` unavailable).

The third-scene recipes are `teatime_spatial_metric.json` and
`spatial_metric_teatime_study.json`. Final machine-readable results, protected
tensor checks, exact sampled-episode histories, tests and cold-export receipts
are sealed in [the evidence ledger](2026-10-07-spatial-metric-evidence.json).
The checks cover 26 distinct regression cases across the 25-test suite and the
four-test spatial/legacy compatibility run (three tests overlap).

## Next boundary

Use the refitted physical metric as the current opt-in research candidate; keep
the frozen control as a failed cross-input ablation and v10 as the deployment
comparator. The architecture remains one scene field and one member energy,
with no source-assisted or native-only query fallback. Storage is still about
2.66 / 2.84 / 8.77 GB per scene, excluding shared input encoders; compactness is
unproven. Preserve this high-capacity reference before any compression study.

The next mechanism problem is nearby unrelated membership and incomplete
identity/extent binding, now less confounded by arbitrary distant members.
Real reference-image and official signed point evaluations are needed beyond
teacher-region diagnostics. ScanNet OVS categorical competition, full NVOS and
UQIS integration, the remaining LERF scenes, and independent final evaluation
are still outstanding. Do not turn three local gates into a five-task claim or
resume per-benchmark threshold/strength searches.
