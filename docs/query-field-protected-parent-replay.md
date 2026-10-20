# SHA-bound protected parent replay, 2026-10-13

## Scope and identity contract

This adds an explicit storage revision of the existing protected-study parent
chain. It does not change historical design/config/result files or silently run
new training. It is not a newly chained three-arm experiment.

- `legacy_field_resolver.create_mapping(bindings, destination, output_root=...)`
  takes explicit (old field directory, v2 directory, store root) triples.
  It hashes the old archive, compares every raw tensor key/dtype/shape/value and
  normalized scientific spec to the v2 load, and checks migration provenance.
  It publishes a new no-replace mapping only after every binding passes.
- Each mapping entry has separate old path + original field.pt SHA, retained
  old-manifest SHA, new manifest-file SHA, new state identity, and explicit store.
  A blob digest or state identity cannot substitute for the original archive SHA.
- `FieldResolver(mapping, expected_mapping_sha256=...)` supports mapped v2 and
  still-present unmigrated v1. A broken mapped dependency is an error, even if its
  old original still exists. Retained original manifests are required.
- The returned parent manifest is the real original v1 manifest, preserving its
  scientific lineage identity. It is never synthesized by putting a blob digest
  in the old field.pt receipt slot.

## Versioned runner

`python -m scripts.research.run_query_field_protected_v2 prepare --source-root OLD_PROTECTED_ROOT --source-design-sha256 ORIGINAL_DESIGN_SHA --mapping MAPPING.json --mapping-sha256 MAPPING_SHA --root NEW_REPLAY_ROOT --store-root SHARED_STORE`

This new overlay anchors the immutable source design and construction ledger,
all old training/manifest receipts, matched source trace and protected-state
assertions, and the current runtime code set. Original code hashes are retained
as historical provenance; they are not claimed to match the revised runtime.
The overlay explicitly records this storage-runtime override. New roots/store
must not be inside the frozen source tree; symlink components are rejected.

`python -m scripts.research.run_query_field_protected_v2 build --root NEW_REPLAY_ROOT --store-root SHARED_STORE --keys SCENE_KEY ...`

Build here means verify/resume existing frozen three-arm outputs through the
resolver, including full payload checks. It does not launch training. Missing
or partial original outputs fail rather than trigger implicit reconstruction.
The optional explicit keys limit the expensive full-load scope; the full
small-metadata construction ledger remains checked. Budget real I/O before use.

Explicit `fit --arm ARM --index INDEX` is available for a separately authorized
future reconstruction into a fresh output slot. It uses unchanged relation or
functional math through loader/writer callbacks and defaults to shared v2.
Functional-native always selects the original frozen protected parent's archive
identity via the resolver. It never silently switches to a newly fitted
protected-training output under the replay root. Such a new experiment needs a
separately specified parent binding and is outside this revision.

`verify-fit` verifies complete new receipts, schema/state identity, both rollback
parents, observation digest, scientific settings/arm, protected-state flags,
functional construction-map hash, and the complete v2 payload. Partial outputs,
wrong mode/store, stale source identity or changed code are rejected.

`export-legacy --old-field OLD_DIRECTORY --old-field-sha256 ORIGINAL_SHA --deployment FRESH_FIXED_SLOT`
uses the same --root and --store-root. It only reports byte-exact old replay after
the actual exported field.pt SHA matches ORIGINAL_SHA. If state-preserving
serialization is not byte-identical it raises and leaves the failed export for
explicit inspection; it does not silently bless or delete it. Tiny CPU float32
archive equality has been tested. No claim is made yet for all 20 real archives.
Exports also require a conservative size cap (--max-export-bytes, default16GiB)
and an additional1GiB free-space reserve; use one explicit fresh fixed slot and
retain it until cleanup is separately authorized.

## Compatibility boundaries

- The old protected runner and old design files are unchanged.
- LERF/NVOS prediction consumers still need explicit independent export plus a
  copied-config field override. Their 14 frozen cohort/config files are not
  rewritten by this resolver.
- Relation/functional fit APIs retain their old default writer for old callers;
  the new protected runner explicitly injects the shared writer. This is not a
  blanket migration of every old writer.
- No real training, GPU experiment, new large migration or deletion was run.
- CPU tests include float32/float64 exact mapped parent states, missing original
  payload by reversible fixture rename, unmigrated v1, changed native rejection,
  wrong SHA namespace/mapping/source, missing store, code/ledger/root checks,
  partial resume, actual runner CLI, and exact standalone micro-export.

## Next batch, prepared only

`docs/query-field-next20-migration-plan.json` lists all 20 exact old paths and
original SHAs plus proposed new paths. Originals total 50,994,366,888 logical bytes
and 54,326,806,528 allocated bytes at the previous stat. The estimated duplicate
native bytes, 24,507,896,832, still require real tensor-hash evidence.

Five migration stages contain four fields each, each bounded to 3400 seconds
(800 seconds per field), followed by separately bounded exact mapping
registration and selected-scene parent replay/export verification. Partial
targets are never overwritten, originals stay until a separate retirement
approval, and no stage starts without RAM/free-space checks and a serialized
I/O window. The plan is not an execution or deletion receipt.
