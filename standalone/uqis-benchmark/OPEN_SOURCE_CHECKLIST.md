# Open-source release checklist

- [ ] Choose and add a project license; the parent repository currently does
  not provide authority to select one automatically.
- [ ] Confirm redistribution language for all documentation excerpts and
  figures.
- [ ] Publish only scripts/manifests/hashes; never redistribute ScanNet,
  ReferIt3D/Nr3D or model checkpoints.
- [ ] Replace local absolute paths in example manifests with placeholders.
- [ ] Mint a signed immutable v0.2 release identity.
- [ ] Freeze independent dev scenes and calibration receipts.
- [ ] Deploy the evaluator-private bundle separately from method workspaces.
- [ ] Run secret-ID, schema-fuzzing and repeated-evaluation-oracle audits.
- [ ] Add CI for Python 3.9–3.12 and build wheel/sdist artifacts.
- [ ] Archive the exact LUDVIG upstream commit, environment lock and field
  inventory without redistributing restricted weights.
