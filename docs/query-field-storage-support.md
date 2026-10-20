# Storage entry-point support, updated 2026-10-13

No GPU experiment is launched by this document. Storage equivalence is not a
benchmark-quality claim. Original frozen configurations and results are retained.

## Native shared-v2 paths

- Main construction/fit:
  `python -m radio_gs.query_field fit --config C --output NEW_RUN --store-root STORE`
  Shared v2 is the default. Missing store fails before training. Historical
  reconstruction is an explicit separate `--legacy-v1` choice.
- Development evaluation:
  `python -m radio_gs.query_field evaluate --config C --output NEW_RUN --store-root STORE`
  v1 remains readable; shared v2 requires the explicit store.
- Five-cohort build:
  `python -m scripts.research.run_query_field_five_build --cohort NEW_COHORT.json --summary-root NEW_SUMMARY --store-root STORE --gpus GPU_IDS`
  The supplied cohort must name fresh job output paths and the approved unchanged
  scientific recipes. The builder passes the store to fit and reports complete
  model bytes. Existing v1 output cannot masquerade as a new v2 build. Historical
  `--legacy-v1` is explicit and mutually exclusive with a store.
- Public ScanNet prediction:
  `python -m radio_gs.query_field.public_scannet predict --field V2_FIELD --store-root STORE --queries Q --encoder E --output NEW_PREDICTIONS`
  Score stage is unchanged. Both v1 and v2 load through the version-aware reader.
- Public UQIS prediction (all four input types):
  `python -m radio_gs.query_field.public_uqis --field V2_FIELD --store-root STORE --public PUBLIC --scene SCENE --encoder E --output NEW_PREDICTIONS`
  v1 is still readable. A missing v2 store/closure fails before encoder work.
  Receipts bind the field schema, state identity and complete referenced payload cost.
- Direct Python `train.fit`: explicitly inject a writer (for example
  `partial(save_field_v2, store_root=STORE)`), or pass `legacy_v1=True` for
  historical full-package reconstruction. Omitting both now fails before reads.
- Active region-semantic compiler:
  `compile_region_semantic.compile_field(..., store_root=STORE)`.
  Parent and region-construction fields may be v1 or v2; all buffers are retained.
- Active joint-observation compiler:
  `compile_joint_observation.compile_scene(..., store_root=STORE)`.
  All four arms retain the exact observation evidence and supervised forest.
- Their actual research runners:
  `python -m scripts.research.run_query_field_native_region prepare --root NEW_ROOT --store-root STORE`
  then `run --root NEW_ROOT --store-root STORE --gpus GPU_IDS`;
  `python -m scripts.research.run_query_field_joint_observation prepare --root NEW_ROOT --store-root STORE`
  then `all --root NEW_ROOT --store-root STORE --gpus GPU_IDS`.
  These commands are documentation, not authorization to launch experiments.
  New runs require both flags before computation, freeze storage settings in the
  new design, propagate them to compiler/predictor subprocesses, and reject stale
  cached construction or prediction identities. The store serves both shared
  inputs and outputs; different stores must be explicitly consolidated beforehand.
  Historical v1 reconstruction uses explicit `--legacy-v1`, without a store.
  Original result/config/design files are not rewritten for new runs.
  Explicit legacy selects the old serialization, not permission to bypass frozen
  code/input SHA checks. A previously frozen design may require its matching code
  checkout; otherwise prepare a fresh design under a fresh root with the same
  scientific settings. Storage changes do not reinterpret old scientific receipts.
- Mixture/confidence and native-mode compilers and their three actual runners
  now require explicit shared storage and new-run roots. Their common prediction
  and score caches bind method/spec/query/config/storage identities. See
  query-field-function-storage.md for commands, test scope and frozen-runtime
  compatibility boundaries.
- Runtime API: `QueryFieldRuntime.load_v2(field, store_root=store)`.
- Versioned direct tools: pipeline_v2 train, scannet-predict, scannet-score,
  migrate-v1, export-v1.

## Public readers and optional independent legacy export

Public LERF and NVOS now load v1/v2 directly: add `--store-root STORE` to their
`predict --config C --job JOB --output NEW_PREDICTIONS` commands when C names
shared-v2 fields. Missing v2 closure/store fails before encoder work. Their new
receipts bind the full manifest/spec as well as state identity; v1 payload SHA
keeps its original meaning. Alternatively prepare one standalone legacy slot:

`python -m radio_gs.query_field.replay_v2 --field V2_FIELD --store-root STORE --legacy-config ORIGINAL.json --field-pointer POINTER --expected-legacy-field OLD_FIELD --expected-source-sha256 ORIGINAL_FIELD_SHA --deployment /tmp/radiogs-replay-cache/active --output-config NEW_REPLAY.json`

The v2 field must have a matching migration-source digest. The helper verifies
the full standalone state/spec and a file-denied cold query. Original config
stays unchanged; only the chosen pointer in the new config is replaced.

Then, only when a benchmark run is explicitly authorized:
- LERF:
  `python -m radio_gs.query_field.public_lerf predict --config NEW_REPLAY.json --job JOB --output NEW_PREDICTIONS`
  Pointer is the exact selected `/jobs/INDEX/field`.
- NVOS:
  `python -m radio_gs.query_field.public_nvos predict --config NEW_REPLAY.json --job JOB --output NEW_PREDICTIONS`
  Pointer is the exact selected `/jobs/INDEX/field`; all original signed-input,
  camera and source-access contracts remain unchanged.
- UQIS:
  Use the corresponding cohort job's exact field pointer when preparing the
  replay receipt, then explicitly supply the resulting standalone directory:
  `python -m radio_gs.query_field.public_uqis --field /tmp/radiogs-replay-cache/active --public PUBLIC --scene SCENE --encoder E --output NEW_PREDICTIONS`
  Preserve the original query program/options; the generated config and receipt
  record the explicit override even though this reader takes a field argument.

LERF/NVOS/UQIS all accept this optional standalone bridge. The historical
five-predict orchestrator still uses frozen cohort configuration and is not
converted into an automatic v2/export scheduler.

## Bounded standalone-cache lifecycle

Use one explicit fixed slot per sequential replay workflow; never auto-generate
timestamped export directories. Creation refuses an existing destination.
For the same state, `--reuse-deployment` revalidates and reuses that exact slot.
A different state is rejected, rather than creating or overwriting another slot.
New exports have a conservative size cap (`--max-export-bytes`, default 16 GiB)
and require an additional 1 GiB free-space reserve.

After the authorized use, retain the small replay receipt and mark the slot
available for a user-approved cleanup/retirement decision. Keep the immutable v2
manifest+all addressed blobs, so the standalone state can be reconstructed.
Do not automatically permanently delete or rotate the slot. If the next state
cannot reuse it, stop and request explicit cleanup/archive handling; do not
silently accumulate exports. The real pilot's existing
/tmp/radiogs-v2-real-deploy-20261012 is preserved pending that decision.

## Not converted

- Of the original 25 direct historical compiler/writer files listed in
  query-field-storage-v2.md, region-semantic, joint-observation and native-modes are
  now migrated. The other 22 are unmigrated old v1 writers, not supported new
  shared defaults. The additional mixture/confidence copy-based writer (outside
  that original direct-save inventory) is also migrated.
  Their present activity/call graph was not comprehensively audited; this list
  does not establish that they are unused or merely historical.
- The next-20 protected parent chain now has an explicit mapping/overlay resolver
  and versioned replay runner; see query-field-protected-parent-replay.md.
  Its 20 real packages are migrated with exact state/spec and historical cold
  parent-chain evidence, but the 20 originals and standalone export remain retained.
  This compiler/reader change invalidates that old overlay's frozen runtime-code
  check. The complete previous 106-file runtime source is preserved with exact
  ledger hashes under experiments/storage_writer_fix_20261013_source_snapshot/before.
  A new explicitly versioned overlay and full cold parent-chain revalidation are
  required before any later retirement; old designs/overlays are not rewritten.
- Initial observation/native compiler caches are not deduplicated by this change.
- Main smoke/diagnose and other historical direct field.pt consumers require an
  explicit standalone export or a separate versioned adaptation.
- No full five-benchmark GPU run or broad old-provenance rewrite was performed.

## Checks

Real tiny CPU main-CLI fits retain v1/v2 states, main evaluation retains rows with
a synthetic renderer, public ScanNet CLI retains predictions with a deterministic
test encoder, and five-build mock jobs verify compile/fit argv propagation for
both storage modes. A real frozen LERF config has been copied with a single
explicit field override and cold-loaded from a real standalone export; no GPU
benchmark was launched. Refer to the test/report receipts for precise counts.

2026-10-13 additions: real tiny region compilation and all four joint-observation
exports match v1/v2 tensor dtype/value and spec exactly. UQIS four modalities match
predictions/selections with synthetic encoders and CPU ray fixtures for both query
programs. Runner command tests cover shared and explicit-legacy propagation,
missing-argument early rejection, construction closure and prediction identity
checks. This is storage validation, not a new scientific benchmark result.
