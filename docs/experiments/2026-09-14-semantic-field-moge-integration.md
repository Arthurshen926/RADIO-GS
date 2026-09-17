# Semantic-field primary, MoGe geometry integration

User direction: return to a semantic-field base map and use sparse surface
carriers as auxiliary geometry. ADR 0007 records this change to ADR 0005.

## Implemented

`radio_gs/field/geometry_prior.py` defines bounded depth compatibility weights.
`radio_gs/field/moge_geometry_prior.py` loads calibrated source MoGe observations,
verifies the calibration/source/prediction hashes, rejects changed camera world
frames, and maps teacher pixels through their camera rays into the native depth
intrinsics. Invalid/out-of-image depth and rejected local calibration are
unknown and preserve original writes. The confidence input currently encodes
validity and accepted calibration, not a learned calibrated uncertainty.

`build_gaussian_multiview_teacher_cache.py` now accepts:

- `--moge-geometry-report`
- `--expected-moge-geometry-report-sha256`
- `--moge-geometry-strength` (default 0.5)

It changes teacher write weights before actual contribution-mean feature
accumulation, not final text scores. The option requires dense exact-marginal
contribution-mean construction and the same sealed responsibility authority
used by MoGe's source selection. Use `--observation-contract legacy` to avoid
claiming an immutable canonical contract; the output carries the new named
geometry-prior contract and source hashes. This does not make it a historical
method: `legacy` here means no promoted canonical-lifting declaration.
Existing commands with no geometry option retain their behavior. Full field
retraining and downstream benchmark evaluation were not executed in this step.

## Verification

59 tests passed across geometry prior, multiview teacher accumulation, sparse
marginal authority and observation contracts. Tests include actual teacher
accumulation, missing evidence, disabled identity, invalid policy, report hash
and camera mismatch. `git diff --check` passed.

The real-asset diagnostic uses one accepted source frame per LERF scene. It
checks Gaussian row hashes and original raster intrinsics against the exact
source authority, reads source MoGe predictions, and verifies original weights
remain unchanged and strength zero reproduces them bit-for-bit. It does not
open query strings or labels and does not rebuild a complete semantic field.

Artifacts:

- `paper/artifacts/moge_gaussian_write_diagnostic_20260914.json`
- `paper/artifacts/moge_precompositor_waldo_20260914.json`

Reproduce with `PYTHONPATH=.` and
`/root/miniconda3/envs/cybersim_agent/bin/python`:

```bash
python scripts/audits/audit_moge_gaussian_writes.py \
  --fusion-root /mnt/pool/sqy/results/RADIO-GS/output/v4_fusion_contract_repair_20260914 \
  --primitive-root /mnt/pool/sqy/results/RADIO-GS/output/optimization_20260815/core_method_v1 \
  --output paper/artifacts/moge_gaussian_write_diagnostic_20260914.json

python scripts/audits/audit_moge_precompositor.py \
  --fusion-report /mnt/pool/sqy/results/RADIO-GS/output/v4_fusion_contract_repair_20260914/waldo_kitchen.json \
  --output paper/artifacts/moge_precompositor_waldo_20260914.json
```

## Newly exposed geometric failure

Waldo kitchen source frame 1 has only 19 accepted Gaussian rows covering 2790
pixels in the retained exact compositor authority. Their hit-weighted-by-count
median centre depth is 0.0076 metres, while sampled MoGe depth is about 2.14
metres. The stored verified COLMAP camera pose agrees with the independently
read COLMAP pose within 7.3e-7; the semantic-cache xyz hash matches the exact
Gaussian authority. Fresh GPU rasterization reproduces the 19-row observation.
This is strong evidence of near-camera foreground contamination in this view;
it is not a claim that every historical evaluation frame has this defect.

A source-only GPU causal pilot compares unchanged rasterization, a two-sigma
camera-plane-clearance gate, and clearance plus per-view MoGe free-space centre
culling. The latter two increase participating rows to 47,531 and 51,580, but
weighted depth compatibility remains below 0.00003 (original approximately
zero). More participating rows therefore does not establish better geometry.
Neither diagnostic gate is promoted or persisted into scene checkpoints.

Post-compositor weighting cannot recover real surfaces already omitted by
visibility/transmittance. Gaussian-centre culling also misses the full pixel
footprint of large or off-image-centre Gaussians. The next necessary stage is
footprint/ray-aware geometric registration or a source-depth-constrained
Gaussian reconstruction, with compatible source-heldout checks. MoGe is useful
here precisely because it supplies an independent geometric constraint; a
simple nearest-voxel transfer or centre-only gate is not enough.

No task-metric improvement is claimed. Historical controls are unchanged. The
new map-builder path remains an optional candidate pending geometric support
repair, matched teacher/field rebuilding, and cold identity–extent evaluation.
