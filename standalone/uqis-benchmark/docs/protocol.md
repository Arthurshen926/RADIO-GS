# Protocol and information firewall

## Three authority roles

1. **Construction Authority** reads official annotations and derives the cohort.
2. **Method Runtime** receives one query workspace and declared persistent field
   dependencies, never private pairing or labels.
3. **Evaluation Authority** receives a sealed complete prediction batch and
   only then opens private instance arrays.

Keeping these roles in separate filesystem bundles is a protocol requirement,
not just a JSON visibility label.

## Opaque identities and pairing

Four queries for one target share no method-visible target ID. Their pairing
exists only in the evaluator manifest. Query IDs are salted commitments over
benchmark version, scene, instance and modality. Image assets are re-encoded
under opaque names so source frame and bbox conventions do not become a side
channel.

Public reports contain aggregate scene/modality metrics. Per-query metric rows
are evaluator-private because ordering and metric fingerprints can reconstruct
cross-modality pairing.

## Independent query workspace

Each method process receives exactly:

- one query row;
- the public mesh domain for its scene;
- its declared query asset, if any;
- a content-bound workspace receipt.

The process starts fresh, the workspace is read-only, and cross-query state is
forbidden. Persistent scene fields are allowed only through the declared field
inventory. Runtime receipts record fresh-process status, physical device,
command, workspace hash, output hash and privacy claims.

## Multiple fields

The v0.2 inventory models modality dependency sets rather than attaching each
field to exactly one modality. For example, a LUDVIG scene may declare:

```text
text      -> [clip_field, dino_field]
image     -> [dino_field]
point_2d  -> [dino_field]
point_3d  -> [dino_field]
```

The CLIP and DINO bytes are charged once each. A persisted DINO k-NN topology
must also be hashed and charged. Rebuildable query-independent caches must be
identified separately from learned field state.

## Prediction sealing

Before private data opens, the evaluator-owned sealer:

1. inventories every expected query ID;
2. requires exact `<query_id>.npy` naming;
3. loads and validates finite `float32[V]` arrays;
4. copies them into an immutable snapshot;
5. records bytes, shape and SHA-256;
6. binds the method identity, field inventory and runtime receipts.

Scoring consumes the in-memory/snapshot arrays directly. It must not hash one
path and later reopen a different filename. A production deployment also needs
an evaluator-owned once-only ledger to prevent adaptive metric-oracle attacks.

## Formal versus candidate

Candidate mode is useful for engineering and controlled validation but always
reports `formal_benchmark_eligible=false`. Formal release additionally requires
fixed official scene/config identities, complete construction provenance,
external immutable commitments, dev-only calibration receipts, sandbox runtime
receipts and one-shot evaluation service enforcement.
