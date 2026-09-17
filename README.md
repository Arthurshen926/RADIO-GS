# RADIO-GS


> Branch recovery (2026-09-14): the v4 object-candidate pipeline is a research
> candidate, not the performance baseline. The retained identity–extent LERF3D
> control has been re-rendered and independently scored at 43.739% sample mIoU.
> See [the recovery protocol](docs/adr/0006-repair-v4-contracts-and-restore-performance-control.md)
> and [the verified control](paper/artifacts/v4_repair_historical_control_20260914.json).
> This does not promote a joint five-benchmark baseline.

Geometry is now constructed and checked independently before semantic training;
see [ADR 0008](docs/adr/0008-finish-and-freeze-geometry-before-feature-reconstruction.md).
Frozen geometry candidates and their heldout source-depth audits do not yet
constitute a promoted semantic benchmark result.

The 2026-09-15 retrieval-field repair rebuilds text, image and registered-point
supervision from consistent source supports, with optional RADIO supervision.
Matched-budget experiments improve image and point retrieval in two development
scenes; ramen text AP still regresses. Native-direction constraints are not a
default. See the [implementation and evaluation report](docs/experiments/2026-09-15-component-pipeline-repair.md)
for all 42 training controls, repeated seeds and cold-query validation.

Reference image/point IoUs are GT-derived **2D category-union development
tasks**, not official LERF image-query or instance-tracking scores. The
[query-contract and reliability audit](docs/experiments/2026-09-15-reference-query-contract-and-reliability.md)
documents the exact inputs, source-query coverage repair, and 12 further
matched-budget controls, including regressions.

The current development direction restores the Gaussian semantic field as the
primary map, with MoGe-3 depth supporting observation registration. The first
bounded depth-weighted teacher builder is an unpromoted candidate; voxel and
surface carriers remain auxiliary. See [ADR 0007](docs/adr/0007-semantic-field-primary-with-moge-geometry-prior.md).

RADIO-GS reconstructs one query-independent C-RADIOv4 feature field on 3D Gaussian primitives and exposes several frozen query interfaces over that same field.

The executable historical mainline is the **one-field route trained with
`canonical-mpr-v3`**: the `canonical-mpr-v2` compact core plus one global,
scene-disjoint 3-D surface-region summary readout. The active five-benchmark
research target is a sidecar-free D512/L512 Single Compact Feature Field, but
that target has not yet completed an end-to-end cold-field benchmark run. The
older HCD, dual-stream screen decoder, and screen-space refiner remain legacy
baselines.

## Current method

```text
frozen C-RADIOv4 spatial observations
                 │
                 ▼
query-free multi-view primitive registration (MPR)
  canonical-mpr-v1: top-1 raster assignment + alpha/depth responsibility
  + per-view normalization + contribution-mean fusion
                 │
                 ▼
compact canonical Gaussian field
  local code → primitive-local residual fusion
             → affine RADIO basis → one 1280-D feature / Gaussian
                 │
        ┌────────┼───────────────┐
        ▼        ▼               ▼
 official     official      global frozen 3D surface-region readout
 DINOv3      SAM3 adaptor       → official SigLIP2 summary head
 adaptor                          (text capability only)
        └────────┬───────────────┘
                 ▼
 typed query compiler → primitive unary / seeds
                 ▼
 optional shared 3D support topology and fixed solver
                 ▼
 3D support or deterministic 2D rendering
  (optional rank-8 zero-mean view residual for 2D rendering only)
```

The canonical field reconstructs raw RADIO only. It never stores a text query, image query, prompt, benchmark mask, or task-specific score. Its affine decoder is point- and batch-invariant, so direct primitive reads and alpha-normalized rendered reads use the same decoded primitive truth.

All new fields require the versioned `canonical-mpr-v1` observation contract.
This is the lifting-layer contract.  The promoted field-training contract is
`canonical-mpr-v2`: initialize from a compliant v1 MPR field, replay the exact
renderer, jointly preserve raw RADIO and frozen official DINOv3/SAM3 outputs
and local affinities, and constrain primitive replay.  Promotion is fail-closed
through the held-out multi-capability Pareto gate; it does not use benchmark
masks, labels, query text, or test-set calibration.
The optional view residual is centered with exact replayed MPR weights and is
never read by primitive-domain text, image, registered-prompt, or 3D-point
queries. It only compensates view context when rendering a 2D feature map.

### Representation status

- `Canonical-D384` is a high-fidelity oracle, not the submitted compact design.
- The historical compact candidate uses a 128-D per-Gaussian local code, primitive-local fusion to a 256-D affine coefficient, and a shared 1280-D RADIO basis.
- The active five-benchmark target uses one 512-D Canonical Capability Feature per Gaussian and must remove historical MPR, semantic-score, descriptor, and topology sidecars before cold query execution.
- DINOv3 and SAM3 capability losses/readouts use the frozen adaptors shipped in the official C-RADIOv4 checkpoint.
- The text readout is custom and predicts a 1280-D RADIO backbone summary token from a physical-scale, surface-connected canonical primitive region. The predicted token is passed through the frozen official SigLIP2 summary head; the custom readout must not be described as an official adaptor.
- It is trained once, without labels, masks, text, or scene IDs, on 24 ScanNet train scenes and selected on 8 disjoint official-val scenes. The older 2-D crop bridge and scene-specific crop-summary MPR caches are diagnostic/oracle baselines only.
- A semantic descriptor cache is disposable: formal text descriptors are deterministic functions of the canonical core, global readout, and official head. Cold per-scene storage contains no semantic field.

The latest controlled experiment record is `paper/artifacts/canonical_compact_capability_iteration_20260714.md`.

## Three-layer organization

### 1. Canonical representation

This layer handles only multi-view lifting, compact storage, affine RADIO reconstruction, and render consistency.

Primary entry points:

- `radio_gs/scripts/build_gaussian_multiview_teacher_cache.py`
- `radio_gs/scripts/train_canonical_radio_field.py`
- `radio_gs/scripts/finetune_canonical_radio_rendering.py`
- `radio_gs/scripts/render_canonical_radio_cache.py`
- `radio_gs/scripts/train_zero_mean_view_residual.py`

### 2. Frozen capability readouts

The same decoded RADIO primitive can be viewed through:

- the official DINOv3 adaptor;
- the official SAM3 adaptor;
- a frozen global region-to-summary bridge followed by the official SigLIP2 summary head.

Primary entry points:

- `radio_gs/scripts/build_canonical_capability_views.py`
- `radio_gs/scripts/build_scannet_surface_region_cache.py`
- `radio_gs/scripts/train_surface_region_summary_readout.py`
- `radio_gs/scripts/build_surface_region_semantic_cache.py`

### 3. Query interfaces

Supported query types compile into primitive-domain evidence:

- text category query;
- pose-free real-image exemplar query;
- registered 2D scribble, point, or reference-mask query;
- world-space 3D point query.

The support solver is optional: category relevance can be read directly, while instance prompts normally use the shared topology for grouping.

Score normalization is also typed and explicit. Independent cosine/relevancy is
the default text scorer, so adding an unrelated query cannot change an existing
score. Scene-category softmax is retained only as a named closed-taxonomy or
LERF paper-compatibility evaluation variant. Every `QueryResult` records the
effective policy; no modality override may use target labels.

Primary entry points:

- `radio_gs/scripts/eval_lerf_grounding.py`
- `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- `radio_gs/scripts/eval_nvos_gaussian_first.py`
- `radio_gs/scripts/eval_scannet_3d_point_query.py`
- `radio_gs/scripts/build_posefree_image_query_cache.py`

The reproducible pose-free ScanNet benchmark builder and its method-independent
mesh-domain evaluators live in `radio_gs/benchmarks/scannet_pfir/`. See
`docs/benchmarks/scannet_pfir_small_v1.md` for the held-out-frame, private-GT,
ranking, and instance-selection contracts.

## Audit ladder

Do not infer field quality from primitive-to-MPR cosine alone. The repository includes separate checks for lifting, coverage, compression, compositing, capability preservation, and query behavior:

- `audit_mpr_view_consistency.py`
- `audit_observation_lifting_contract.py`
- `audit_canonical_render_ceiling.py`
- `audit_feature_compositing.py`
- `audit_canonical_capability_fidelity.py`
- `audit_capability_cache_fidelity.py`
- `eval_canonical_primitive_reconstruction.py`

Every main result should retain provenance for the RADIO checkpoint, geometry rows, field checkpoint, capability adaptor, semantic bridge and region policy, graph topology, query compiler, threshold, and solver.

## Evaluation contract

Benchmark protocols are task-specific, but method internals are allowed when they are frozen without target labels. The current default rules are:

- use the benchmark's official query text/prompt and target data;
- compute its published metrics in the published output domain;
- allow query-independent field training, official frozen adaptors, fixed prompt registration, and a globally frozen support solver;
- record unary, propagated, and connected-component stages separately;
- do not select thresholds, calibrations, bridge checkpoints, or graph policies on benchmark test labels;
- keep `test_calibration=false` for formal results.

Here, `test_calibration=false` means that target labels, masks, or metric feedback are never used. If a declared variant uses query-conditioned scores or unlabeled evaluation-scene statistics, those transductive inputs are recorded separately rather than hidden by the safety flag.

Current NVOS and SPIn-NeRF outputs are protocol diagnostics until all official scenes are present and evaluated. A single-scene result must not be copied into a full benchmark main table.

## Environment

The code targets Python 3.9+, PyTorch/CUDA, `gsplat >= 1.4`, and a local checkout of the official RADIO repository.

```bash
conda create -n radio-gs python=3.9 -y
conda activate radio-gs
pip install -r requirements.txt
pip install -e .
export RADIO_REPO=/path/to/RADIO
```

Geometry, feature, and benchmark paths are supplied through each script's arguments or the legacy YAML configs where needed. Large datasets, checkpoints, and `output/` artifacts are intentionally not versioned.

## Tests

```bash
python -m pytest -q
```

Focused tests cover canonical-field checkpoint contracts, official capability fidelity, contribution compositing, view residuals, query compilation, score calibration, graph channel mixing, deterministic support solving, generic bridge data splitting, and evaluator safety flags.

## Legacy path

The following modules are retained for historical and matched-baseline experiments only:

- `radio_gs/scripts/train_feature_field.py`
- HCD codec / dual-stream hybrid configs;
- screen-space feature refiners;
- benchmark-specific auxiliary heads from earlier iterations.

Legacy checkpoints must be evaluated with their matching legacy renderer. They must not be silently loaded as canonical-field checkpoints or used to support claims about direct 2D/3D query consistency.

2026-09-16 iteration: [native decoder coupling and joint query contracts](docs/experiments/2026-09-16-decoder-coupling-and-joint-query-contracts.md)
records six controlled LERF runs, optional exact teacher/query readout, ScanNet
raw-class point output, and eight NVOS reference scribble query compilations.
Mixed LERF results do not pass joint promotion; new ScanNet/NVOS field metrics
are pending. [Evidence index](paper/artifacts/decoder_joint_iteration_20260916.json).

2026-09-16 follow-up: [signed support and NVOS camera repair](docs/experiments/2026-09-16-signed-support-and-nvos-camera-repair.md)
fixes prompt/render validity mismatch and NVOS camera/frame identity contracts.
The first fern frozen geometry and source contribution assets are built;
rendered validation rejects its joint-scale candidate. Signed image conditioning
also remains experimental after mixed-to-negative cross-view results.
