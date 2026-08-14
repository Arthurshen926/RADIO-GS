# UQIS Benchmark

UQIS (Unified Query Interaction Segmentation) evaluates whether one persistent
3-D scene representation can respond to four independently authorized query
interfaces:

- natural-language text;
- an isolated RGB reference crop;
- a positive 2-D click on a benchmark-rendered, RGB-free interaction raster;
- the paired positive 3-D world point.

Every method returns a finite `float32[V]` score/probability vector on the same
content-bound official ScanNet mesh. UQIS therefore compares query interfaces
without changing the output domain or metric implementation.

The package exposes a deliberately small interface: construct/audit a release,
stage one isolated query workspace, seal a complete prediction batch, and run
the evaluator. Dataset parsing, opaque query IDs, method/evaluator information
firewalls, field accounting, bootstrap aggregation and fail-closed receipts
remain behind those interfaces.

## Status

This directory is an open-source-ready extraction snapshot of the implementation
inside RADIO-GS. The current nine-scene v0.2 result is a non-formal construction
candidate. Publishing a formal leaderboard release still requires a separate
dev calibration authority, an external immutable release commitment and a
one-shot evaluator deployment.

No ScanNet imagery, meshes, labels, Nr3D annotations, LUDVIG weights or model
outputs are redistributed here. Users must obtain each upstream dataset under
its own terms.

## Install and test

```bash
python -m pip install -e '.[test]'
pytest -q
```

Primary commands:

```bash
uqis-construct-scannet --help
uqis-build --help
uqis-audit --help
uqis-stage-workspace --help
uqis-seal-predictions --help
uqis-evaluate --help
```

Read the documentation in this order:

1. [Benchmark design](docs/benchmark-design.md)
2. [Dataset construction](docs/dataset-construction.md)
3. [Protocol and information firewall](docs/protocol.md)
4. [Metrics and reporting](docs/evaluation.md)
5. [Package architecture](docs/architecture.md)
6. [LUDVIG reproduction](docs/ludvig-reproduction.md)
7. [Current results](docs/results.md)
8. [Open-source release checklist](OPEN_SOURCE_CHECKLIST.md)

## Synchronization

RADIO-GS remains the canonical development location until this directory is
split into its own repository. From the RADIO-GS checkout, refresh the source
snapshot with:

```bash
python standalone/uqis-benchmark/tools/sync_from_radio_gs.py
```

The synchronizer refuses any selected module that still imports `radio_gs`, so
the exported package cannot silently depend on the parent repository.
