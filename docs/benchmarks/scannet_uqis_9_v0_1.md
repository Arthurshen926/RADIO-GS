# ScanNet-UQIS-9 v0.1

## Authority state

The official nine-scene construction is sealed separately from public formal
evaluation. A `Construction Authority` binds the ScanNet sources, Nr3D,
cohort ledger, target derivations, query exclusions, and evaluator-only
construction candidate. It authorizes content-bound method mapping. It does
not yet authorize method-visible formal query bundles, evaluator access,
metric release, or a formal benchmark row; those require the later Evaluation
Authority gate described in ADR 0003.

## Status

`scannet-uqis-9-v0.1` is implemented as a **result-ineligible pilot
harness**. It is ready for protocol integration tests and method-adapter
smokes, but it cannot yet mint a formal dev/test release. The formal path
fails closed until target eligibility, query-frame exclusion, mapping input,
and replacement decisions are recomputed from content-bound official assets
and sealed in stage receipts.

This distinction is deliberate. A hand-authored pilot manifest must not be
able to acquire the same authority as the final benchmark merely by selecting
`split_role=test`.

## Evaluation question

The primary benchmark asks one controlled question: given one persistent,
query-independent 3-D feature field, how well can the same field localize the
same physical object when the authorized query changes? A secondary
`modality_specific_multi_field` comparator track permits a complete method
system to use several modality-scoped fields, but reports that scope, field
count, and the sum of all per-scene persistent field bytes.

The three query families are text, pose-free image, and prompt. Prompt is
split into registered 2-D point and world-space 3-D point protocols, yielding
four scored modalities:

| Modality | Method-visible query input | Explicitly unavailable |
|---|---|---|
| `text` | scene ID and a natural, view-independent expression | target ID, paired queries, image, pose, mask |
| `image` | scene ID and an opaque 224×224 RGB exemplar crop | source frame name, bbox, pose, mask, paired queries |
| `point_2d` | scene ID, camera-to-world, intrinsics, raster size, one positive pixel | captured RGB, depth, mask, labels, image embeddings |
| `point_3d` | scene ID and one world-space positive point | camera, RGB, depth, mask, labels |

For `point_2d`, a method may render its already-frozen field/geometry into the
declared raster and interact with that rendering. It may not read a captured
RGB query image or invoke an RGB-consuming segmenter such as SAM at query
time. The 2-D click and 3-D point are derived from the same surface sample,
but the method never receives that pairing.

## Controlled pairing and firewall

For each Unified-Query Target, the constructor derives all four queries from
one ScanNet instance, one Query Camera, one official mesh ground truth, and one
paired surface point. Cross-modality pairing lives only in
`target_manifest.evaluator.json`.

Every public query has an unrelated opaque `uq_<digest>` identifier. A method
is run in an independent one-query workspace containing only:

- its modality-specific query record;
- the official ScanNet mesh XYZ output domain;
- the image crop only for image queries.

The workspace excludes evaluator labels, target identity, other modalities,
private depth, and field-exclusion details. Runtime orchestration must create
a fresh process/workspace per query and may not retain state across queries.

## Cohort and construction contract

The frozen nine-scene test order is:

1. `scene0030_00`
2. `scene0249_00`
3. `scene0353_00`
4. `scene0435_00`
5. `scene0700_00`
6. `scene0704_00`
7. `scene0246_00`
8. `scene0203_00`
9. `scene0535_00`

The content-bound cohort ledger records the four rejected primary candidates,
the accepted replacement, the two pre-evaluation additions, and the decision
not to mine the full validation split for a tenth scene.

Each final scene must contain 6–8 targets, at least four semantic categories,
at least three targets with a same-class distractor, and no more than two
selected targets per class. A target must have at least 500 official mesh
vertices, at least 1,000 query pixels or 1% image area, at least 0.90
projection purity, and at least 0.70 field-surface coverage under the frozen
distance and visibility rules. Text expressions must be correct,
class-mentioning, and view-independent; missing annotation flags fail closed.
Formal Nr3D ingestion binds the raw CSV and applies ReferIt3D's released
lexical view-dependence rule to the released `tokens` column; it does not
require a non-existent derived `dep_or_indep` column.

At most three Query Cameras cover a scene's targets. The field-construction
frame set excludes the union of every Query Camera, its ±5 temporal neighbors,
and near-pose frames within 0.10 m and 8°. Formal execution additionally
requires a content-addressed inventory proving that only the filtered Mapping
Observations were opened.

The local candidate audit is diagnostic:

```bash
python -m radio_gs.benchmarks.scannet_uqis.audit_candidate_assets \
  --sens-root /path/to/scannet \
  --annotation-root /path/to/scannet/annotations \
  --reference-annotations /path/to/nr3d.csv \
  --output /path/to/candidate_asset_audit.json
```

It inventories and hashes source assets, and reports coarse geometry and
expression counts. It never claims that visibility, purity, field coverage,
or exclusions have been derived.

## Output and metrics

Every query emits one `float32` array named `<query_id>.npy`, with one finite
probability in `[0,1]` per vertex of the bound official ScanNet mesh. The
single output-domain identifier is
`official_scannet_mesh_vertex_probability`.

Per query, the evaluator computes tie-aware AP, oracle-threshold IoU,
fixed-threshold IoU at 0.5, accuracy at IoU 0.25/0.50, selected purity,
positive coverage, maximum same-class-distractor IoU, and centroid error in
meters. Queries are averaged within each scene, then scenes receive equal
weight. Confidence intervals use a scene-clustered bootstrap with 2,000
samples and seed `20260813`.

`UQ-Mean` is the equal-modality mean of fixed IoU 0.5 and exists only for one
method-system identity that sealed all four modalities. Its row always states
whether the representation is `single_universal_field` or
`modality_specific_multi_field`; the latter is not eligible for a
single-universal-field claim. A single-modality comparator may report that
modality's scene macro, but its `UQ-Mean` is null.

## Release and evaluation workflow

A pilot can be frozen only with the explicit incomplete-pilot switch:

```bash
python -m radio_gs.benchmarks.scannet_uqis.build_benchmark \
  --scene-records scene_records.json \
  --target-records target_records.json \
  --query-id-salt-file private_query_id_salt.bin \
  --split-role pilot \
  --allow-incomplete-pilot \
  --output-dir /path/to/release
```

Audit all manifest and asset hashes:

```bash
python -m radio_gs.benchmarks.scannet_uqis.audit_benchmark \
  --benchmark-dir /path/to/release
```

`--skip-asset-hashes` is diagnostic and always returns invalid. Stage one
one-query method grant with:

```bash
python -m radio_gs.benchmarks.scannet_uqis.stage_query_workspace \
  --benchmark-dir /path/to/release \
  --modality point_2d \
  --query-id uq_... \
  --workspace-dir /fresh/query/workspace
```

Staging creates the minimal directory and receipt; it is not itself a process
sandbox. A formal orchestrator must expose only that directory as a read-only
mount, start a fresh process, record observed file access, and discard the
workspace after the query. That formal orchestrator is not yet implemented.

The prediction sealer can already inventory and hash all outputs without
opening private pairing or labels:

```bash
python -m radio_gs.benchmarks.scannet_uqis.seal_predictions \
  --benchmark-dir /path/to/release \
  --prediction-dir /path/to/predictions \
  --method-run-manifest /path/to/method_run_manifest.json \
  --row-scope universal_complete \
  --output /path/to/sealed_prediction_batch.json
```

For a one-modality baseline, use `--row-scope modality_comparator --modality
image`. Private evaluation is intentionally disabled for pilot releases;
repeated aggregate feedback can itself become a label oracle. The formal
evaluator must run once under evaluator-owned release/method authority after
the complete batch is sealed. Its public report contains scene/modality
aggregates only, while detailed rows remain evaluator-private.

## Initial LUDVIG integration

`run_uqis_image.py` is a benchmark-local image-query adapter. It executes the
real vendored DINOv2/PCA descriptor path, reads the frozen LUDVIG Gaussian
feature field, performs a continuous Gaussian-to-official-mesh readout, and
writes fixed-sigmoid scores in the mesh-probability output domain. The default
scale 1, bias 0 map is protocol-fixed, not empirically calibrated. The adapter
does not modify the historical LUDVIG PFPR reproduction.

The completed 2026-08-13 exact-runtime evidence, hashes, resource use, and
reproduction command are recorded in
`paper/artifacts/scannet_uqis_ludvig_image_smoke_20260813.md`.

The current smoke is deliberately marked:

- `result_eligible=false`;
- `formal_benchmark_row_eligible=false`;
- `official_ludvig_reproduction=false`;
- `paper_metric_comparable=false`;
- `benchmark_local_adapter=true`.

The LUDVIG target integration constructs two fields per evaluation scene: a
CLIP language field for `text`, and a DINOv2 visual field shared by `image`,
`point_2d`, and `point_3d`. Benchmark-local adapters are connected for all
four scored modalities. The strict 2-D prompt adapter renders only the frozen
DINOv2 field and reads its declared click pixel; it never opens captured query
RGB or depth. The 3-D adapter interpolates that field at the declared world
point. The historical SAM prompt path is therefore not substituted. The
complete LUDVIG row will be labeled
`modality_specific_multi_field` and charged for both fields.

## Remaining formal-release work

The official-asset constructor now derives target eligibility, Nr3D text,
Query Cameras, image crops, paired 2-D/3-D prompts, full-sensor union
exclusion, and post-exclusion surface coverage. It emits a content-bound scene
derivation receipt and cohort ledger. Formal release authority remains closed
until all nine scenes complete that constructor and the mapping-observation
runtime receipt, query-execution sandbox receipt, externally authorized method
identity, immutable release-root commitment, and dev-only calibration receipt
are bound.
Until those artifacts exist, this version remains useful for adapter
engineering and evaluator validation, not a leaderboard claim.
