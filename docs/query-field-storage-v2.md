# Query-field storage v2 (2026-10-12)

Status: implemented, CPU validation and a real two-variant migration pilot passed.
The separately tracked five-variant teatime batch is in progress. Original
artifacts remain; no deletion or benchmark quality promotion is implied.
ADR0013/0021 mathematical and typed-input contracts are unchanged.

## Supported new chain

Use `python -m radio_gs.query_field.pipeline_v2` from this checkout:
- `train --observations O --config C --output RUN --store-root STORE`
- `scannet-predict --field RUN/trained --store-root STORE --queries Q --encoder E --output P`
- `scannet-score --field P --evaluation EVAL --output RESULT.json`
- `export-v1 --field RUN/trained --store-root STORE --output DEPLOY`
- `migrate-v1 --field OLD --store-root STORE --output NEW`

All destinations must be new. Supply canonical physical paths, not the checkout's
output symlink. Store roots are explicitly provided on every shared read; no
environment variable or provenance path silently locates dependencies.
The new train/migrate entry requires a store root, so untrained/trained variants
share their unchanged geometry and native tensors. Older frozen entry points
continue writing/reading v1. This is not a claim that all future runners have
automatically switched. Use this new versioned entry for the supported chain.

The public ScanNet predictor is the existing implementation, with an explicit
loader injection. Scoring is unchanged. Public QueryFieldRuntime.load_v2 enables
the same typed runtime after a v2 load; legacy load rejects v2. Existing current
full-observation and region/optics/topology state are preserved and tested, but
their frozen compiler/runner scripts have NOT been converted to v2.

## Format and cost

Each state tensor, including geometry, native buffers, all evidence/offset/index
tensors, and learned parameters, has a SHA-256 address over canonical
dtype/shape/byte-count plus exact contiguous little-endian bytes. Equal values
share; differently trained native values and changed/reordered geometry receive
different addresses. No quantization, precision conversion, lossy compression,
or removal of persistent buffers is used. Noncontiguous stride is canonicalized;
values, order, dtype and shape are preserved. Sparse/quantized tensor objects
are explicitly unsupported; dense CSR pointer/index/value components are supported.

A variant manifest binds the entire state mapping, science spec, ordered Gaussian
row contract, provenance, and full content-addressed dependency closure. The
manifest is integrity-checksummed, not cryptographically signed/authenticated.
Row ordering is explicit via geometry hashes plus every tensor's exact ordering;
this does not create new physical element identities or certify scientific data.

Research shared v2 has blob_store=explicit and is NOT standalone. The explicit
blob dependency graph contains only addressed tensors; no training-source
dependencies are allowed. blob_store=local with a local blobs directory is a
complete v2 closure. save_field_v2(..., store_root=None) creates that closure.
Independent export-v1 emits existing self-contained v1 field.pt plus manifest;
v1-unrepresentable constructor dtypes are refused rather than silently cast.
For such fields use a complete v2 closure.

Cost reporting distinguishes:
- tensor_bytes: logical tensor bytes including duplicate values
- unique_blob_bytes: all blobs needed by this model, even when already shared
- new_blob_bytes: newly allocated blobs for this save
- deployment_bytes(v2): complete unique blobs + canonical manifest including newline
- deployment_bytes(v1): historical field.pt-only definition, unchanged

Neither new_blob_bytes nor manifest bytes alone is a complete model-size metric.
Encoder dependencies remain separate, as previously declared.

## Safety and resource behavior

Writers serialize with a POSIX advisory store lock; blobs and manifests use
same-filesystem temporary files, fsync and atomic no-replace hard-link publication.
These are raw single-tensor immutable blobs, NOT hardlinks between different
full model checkpoints. Existing blobs are verified before reuse. A finished
manifest is published last. Failed saves may leave an incomplete reserved
destination or orphan blobs; automatic cleanup never deletes them.
GC only reports candidates, and explicitly cannot prove the supplied inventory
is complete. Do not delete any original or candidate without separate approval.

The preflight estimates missing-blob allocation plus conservative metadata,
largest temporary blob, and additional free-space reserve (default 1 GiB).
Atomic link publication does not duplicate physical file blocks. Budget covers
one save, not an entire future run. Existing shared blobs must still be read to
verify their integrity; saves can be I/O-heavy. No large pilots may overlap the
separately coordinated GPU/NFS timing window. Memory includes all loaded state
and per-tensor buffers; this is not a streaming multi-TB migration implementation.

Static symlinks, traversal/content-address misuse, wrong shape/bytes/dtype,
missing/corrupt blobs and manifest alteration fail closed. This is a cooperative
immutable store, not a security sandbox against an adversary concurrently
replacing parent directories. O_NOFOLLOW protects payload leaves but parent-path
TOCTOU is not claimed solved. Locking assumes the filesystem supports POSIX locks.

## Legacy save-site inventory

The original inventory had 25 direct compiler/writer files plus train.py's two
save sites, accounting for 27 pre-existing research save sites. On 2026-10-13,
compile_joint_observation.py, compile_region_semantic.py and compile_native_modes.py
were migrated with their actual runners; the other 22 listed files remain unmigrated old v1 writers.
The separate copy-based compile_mixture_evidence.py was also migrated for both
mixture and confidence variants; it was outside the original direct-save count. Their current
activity was not fully audited; they are not presumed unused or frozen history.
Direct train.fit now requires an explicit writer or legacy_v1=True; it no longer
silently defaults to v1. The main CLI and versioned pipeline inject the v2 writer.
Their field.pt receipts and frozen scientific provenance cannot be reinterpreted
as v2 without separately versioning the complete reader/report chain.

radio_gs/query_field:
- compile_joint_observation.py — migrated, explicit store default (2026-10-13)
- compile_native_modes.py — migrated, explicit store default (2026-10-13)
- compile_paired_evidence.py
- compile_region_semantic.py — migrated, explicit store default (2026-10-13)
- compile_semantic_views.py
- compile_soft_paired_evidence.py
- coordinate_training.py
- element_relation_training.py
- functional_training.py
- metric_study.py
- relation_training.py
- support_control.py

scripts/research:
- compile_query_field_coverage_factors.py
- compile_query_field_extent_factors.py
- compile_query_field_native_region_appearance.py
- compile_query_field_region_carrier.py
- compile_query_field_scope_constraints.py
- compile_query_field_scope_permissions.py
- compile_query_field_support_footprint.py
- run_competitive_evidence.py
- run_expected_support.py
- run_query_field_extent_preparation.py
- run_query_field_protected.py
- run_query_field_scope_topology.py
- run_query_field_semantic_measure.py

## Validation scope

Tiny CPU tests cover exact state/readout/dtype (including fp16, bf16 and float64),
sharing versus changed native/geometry, corrupt/missing/truncated blobs, manifest
tampering/traversal, static symlink rejection, no-replace, budget rejection,
GC report-only, full-observation evidence/offsets, optics/preparation/topology,
and standalone v1 cold queries after removing the tiny shared source.
A real two-step CPU fit reproduces v1 training history and complete states;
the actual public ScanNet prediction path reproduces native/member outputs
with a deterministic fake input encoder. No GPU training, real encoder inference,
full five-benchmark run, or adversarial NFS race was run. Real existing-artifact
migration evidence is separately recorded in query-field-storage-v2-pilot-results.json.

## Default entry points and remaining compatibility boundaries

The commonly used `python -m radio_gs.query_field fit` now defaults to shared v2.
It requires `--store-root STORE` and rejects a missing root before opening the
training config or starting training. `--legacy-v1` is an explicit historical
reconstruction choice, mutually exclusive with a shared store. There is no
implicit fallback to a full v1 save. Example:

`python -m radio_gs.query_field fit --config C --output RUN --store-root STORE`

The matching `evaluate` stage now uses the versioned loader, accepts v1 unchanged,
and requires `--store-root STORE` for an externally shared v2 package.
`python -m radio_gs.query_field.public_scannet predict` similarly accepts v1/v2
with `--store-root`; its CLI also accepts `--device cpu` for small checks.
The baseline mathematical configuration is unchanged. Direct low-level fit()
requires an injected save_function or explicit legacy_v1=True. The low-level
save_field() remains the deliberately named v1 writer for frozen callers/export.

The current `scripts/research/run_query_field_five_build.py` now accepts explicit
--store-root, --cohort and --summary-root for new builds, forwards the store to
fit, and reports v2 identity/full cost. Missing new-build arguments fail closed.
Historical full-package rebuilds require --legacy-v1. Existing frozen cohort
configuration and construction receipts are not rewritten. See
query-field-storage-support.md for the complete support and replay matrix.

Public LERF and NVOS now read v1/v2 with explicit --store-root; standalone
export remains an optional bridge for frozen configurations. Public UQIS now natively reads v1/v2 with explicit --store-root and binds
schema/state identity/full cost in receipts. The historical five-predict scheduler
has not been made a general v2/export scheduler. The remaining 22 unmigrated old field
writers (activity not exhaustively audited), smoke/diagnose paths and initial observation/compiler caches remain
outside this switch. Native-region and joint-observation runners now require
--root plus --store-root for new runs and preserve explicit --legacy-v1 replay.
See query-field-storage-support.md for exact commands and restrictions. No claim
is made that every historical producer/parent-resume chain has been converted.

### Versioned legacy-config replay

`python -m radio_gs.query_field.replay_v2` prepares a NEW replay config and
standalone v1 export. Required arguments are --field, --store-root,
--legacy-config, --field-pointer (JSON pointer), --expected-legacy-field,
--expected-source-sha256, --deployment and --output-config.
It verifies the explicit legacy-source digest against the v2 migration receipt,
checks every tensor/spec, and runs a file-denied CPU cold query before publishing
the copied config. Only that pointer is changed. Original config, original
manifests, other fields and output locations remain unchanged. No benchmark is
automatically launched. --reuse-deployment requires a verified existing
standalone export of the same v2 state, and still reloads/verifies its payload.

Verified real example: legacy lerf_two_scene_probe.json, pointer /jobs/1/field,
source e926257bb3ee3c924246c779a0f640f98e8b4d7a8ddb77289036469cb54dbe8f.
Prepared config is /tmp/radiogs-lerf-two-scene-replay-v2-bound-20261012.json,
with adjacent .replay.json receipt; the original config SHA remains
862289697644af1a9502445ff3f25c3a7b9ddad655b2a52190638af12ad41be0.
A later explicitly authorized replay can invoke public_lerf predict with this
NEW --config, --job figurines_18 and a NEW --output; no GPU replay was run here.
If original payloads are retired later, bare historical loads will fail until
this explicit export/replay route is used. Never silently rewrite old manifests.

CLI checks use real two-step CPU fits and the real evaluator/public ScanNet
control flow. The tiny evaluator test substitutes a synthetic one-row-per-pixel
rasterizer, and the ScanNet test supplies a deterministic fake input encoder;
these are storage integration tests, not GPU/encoder/benchmark quality claims.
