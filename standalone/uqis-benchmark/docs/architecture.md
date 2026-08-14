# Package architecture

The standalone extraction is organized around four deep modules. Their
interfaces are the supported caller and test surface; helper functions inside
them are implementation details.

## Release Authority module

Interface: `freeze_release(...)` and `audit_release(...)`.

It owns opaque query identity generation, public/private manifest splitting,
schema validation, asset hashing, config commitments and release audit. The
ScanNet constructor and text-profile code feed normalized records into this
interface rather than writing method manifests directly.

Relevant implementation files: `protocol.py`, `construction.py`,
`official_constructor.py`, `construction_authority.py`, and
`scannet_assets.py`.

## Query Runtime module

Interface: `stage_query_workspace(...)`.

It hides modality manifest lookup, mesh copying, image normalization, exact
one-query filtering and receipt construction. A production launcher adds
read-only mounting/process isolation and emits a runtime receipt.

Relevant implementation files: `workspace.py` and
`stage_query_workspace.py`.

## Evaluation Authority module

Interface: `evaluate_release(...)` for sealed file execution and
`evaluate_predictions(...)` for explicitly diagnostic in-memory evaluation.

It owns prediction inventory validation, private target-mask construction,
metric calculation, scene/modality aggregation and report privacy. Formal
deployment should expose only the sealed interface.

Relevant implementation files: `evaluate_predictions.py`, `metrics.py`,
`seal_predictions.py`, and `controlled_evaluation.py`.

## Method Field Inventory module

Interface: `validate_method_field_inventory(...)`.

It validates persistent artifacts, modality dependency sets, shared-byte
accounting and representation class. It does not know how a specific method
builds CLIP, DINO, Gaussian or voxel fields.

Relevant implementation file: `method_fields.py`.

## Internal adapter seam

`scannet_assets.py` is the official-asset adapter used by the constructor. It
is intentionally internal: callers provide asset roots and receive normalized
records through the Release Authority module. The adapter was extracted from
the older RADIO-GS PFIR code and verified byte-for-byte on official mesh XYZ,
instance IDs and metadata.

## Source layout

```text
uqis-benchmark/
  src/uqis_benchmark/   installable benchmark core
  tests/                tests through supported interfaces
  docs/                 protocol, construction, metrics, results and ADRs
  tools/                extraction/synchronization tooling
  pyproject.toml         package and CLI metadata
```

Until repository split, RADIO-GS is canonical and
`tools/sync_from_radio_gs.py` refreshes the standalone snapshot. The tool
rejects any exported module that imports the parent package.
