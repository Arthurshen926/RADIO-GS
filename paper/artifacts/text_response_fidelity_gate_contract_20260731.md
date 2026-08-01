# Target-blind text-response fidelity gate

## Contract audit

The schema-v3 SurfaceRegion cache already contains every teacher-side tensor
needed by this gate:

- `official_crop_summaries [R,V,1536]` and `teacher_mask [R,V]` define the
  normalized mean official SigLIP2-g teacher descriptor;
- `radio_features`, `geometry`, `token_mask`, `reliability`, and `anchor_index`
  are the exact readout inputs;
- `metadata.region_records` supplies row-aligned scene and region identities;
- `metadata.radio_checkpoint_sha256` and `region_contract_sha256` bind the
  official summary head and SurfaceRegion contract.

The schema-v3 readout checkpoint does **not** store student descriptors.  It
stores only the 1280-dimensional token-space readout state.  Therefore a
separate CPU materialization step must apply the same frozen
`SigLIP2SummaryHead` used during training.  The materializer refuses CUDA,
benchmark scenes/vocabulary, mismatched RADIO checkpoints, mismatched region
contracts, and caches without row identities.

The response SmoothL1 metric is exactly the same independent-query objective
as
`compute_independent_normalized_cosine_response_smooth_l1_loss`:

```text
S = normalize(student_descriptor) @ normalize(text_embedding).T
T = normalize(teacher_descriptor) @ normalize(text_embedding).T
SmoothL1 = mean(smooth_l1(S, T, beta=1))
```

No softmax is applied across queries.

## Frozen generic text bank

The vocabulary source is
`target_blind_imagenet1k_primary_text_bank_v1.json`, with `fit/dev/audit`
splits.  Promotion accepts only `dev` or `audit`, never `fit`.  The evaluator
requires one frozen CPU embedding artifact and its explicit sidecar for the
requested held-out split.  It accepts only algorithm
`siglip2-target-blind-split-v1`; the earlier all-997 synthetic layout is not a
fallback.

```text
schema_version: 1
artifact_type: target_blind_text_embedding_cache
algorithm_version: siglip2-target-blind-split-v1
benchmark_vocabulary_opened: false
uses_benchmark_vocabulary_for_construction: false
split: dev | audit
split_synset_tab_query_lf_sha256
prompt_templates: ["{query}"]
text_canonicalization: official_c_radio_siglip2_g
records / queries / synsets: selected split only
vocabulary_path / vocabulary_sha256
vocabulary_manifest_path / vocabulary_manifest_sha256
ordered_records_sha256
embeddings: CPU float32 [Q,1536], L2 normalized
embedding_semantic_sha256 / embedding_tensor_sha256
text_encoder:
  model_id
  revision
  snapshot_path / snapshot_files_sha256
  tokenizer_sha256
  config_sha256
  model_index_sha256 / weight_shards_sha256
  output_dimension: 1536
  dtype: float32
  normalization: l2
  device: cpu
```

The evaluator reopens only this target-blind generic vocabulary, verifies the
canonical and manifest hashes, verifies each split hash, verifies exact
embedding-row alignment, both embedding hashes, all frozen encoder snapshot
file hashes, the sidecar's artifact SHA, and its builder SHA.  The requested
split must equal the artifact split.  It does not open a LERF, NVOS, or
ScanNet benchmark vocabulary.

The vocabulary and model manifests are not trusted as self-signing
authorities.  Production code independently pins the canonical vocabulary
SHA, the `1000/997/806/101/90` source/deduplicated/split counts, both timm
source SHAs, every fit/dev/audit split SHA, and every file SHA in SigLIP2
revision `a713301b...` (including both safetensor shards).  A coherently edited
vocabulary/manifest or a same-named snapshot directory is rejected.  Tiny
fixtures can inject a private in-process test contract; those arguments are
not exposed by the production CLI.

## Descriptor artifact

`materialize_surface_text_response_descriptors.py` emits:

```text
schema_version: 1
artifact_type: surface_text_response_descriptor_pair
method_id / seed / split_role=validation
student_descriptors / teacher_descriptors: CPU floating [R,1536]
scene_ids / region_ids
student_descriptors_sha256 / teacher_descriptors_sha256
descriptor_rows_sha256
descriptor_space:
  name: official_siglip2_g_summary
  dimension: 1536
  normalization: l2
provenance:
  all benchmark/annotation/text access flags: false
  cache, readout, RADIO, and region-contract hashes
```

Control and candidate reports are pairable only when the selected generic
query hash, embedding tensor hash, ordered descriptor rows, and complete
teacher descriptor tensor all match exactly.

Loading a descriptor also reopens and hashes its readout checkpoint and
report, RADIO checkpoint, binding-authority manifest/completion, and every
validation cache.  Cache metadata is checked against the split, RADIO,
SurfaceRegion, and teacher-target hashes, and cache `region_records` must
exactly replay the descriptor's ordered `(scene_id, region_id)` rows.  Boolean
`false` provenance declarations alone are insufficient.

## Metrics and promotion

For each descriptor artifact the evaluator reports:

- mean response SmoothL1 and MAE over every `region x query` cell, with the
  corresponding per-region query means retained for reverse validation;
- mean and p05 cosine between each student's and teacher's full response
  profile;
- scene-query Spearman rank correlation across regions (average ranks for
  ties; constant teacher queries excluded from rank only);
- scene-query top-decile overlap at `ceil(0.10 * scene_regions)`;
- mean and p05 rank/top-decile metrics, plus per-scene and per-scene-query
  audit rows.

The gate recomputes every aggregate, scene statistic, and declared count from
the persisted per-region/per-scene-query details (allowing only float32
reduction-order tolerance of `1e-7`).  A changed aggregate or scene row is
rejected before bootstrap or promotion checks.

Before that statistical validation, the gate reopens every report-bound
descriptor, embedding artifact, embedding manifest, vocabulary, snapshot and
provenance file, regenerates the complete report, and requires byte-semantic
equality with the supplied JSON.  Thus a coherent rewrite of detailed metrics
and aggregates is also rejected.  `--phase dev` is selection-only and
`--phase audit` is confirmation-only; all control/candidate reports must use
that same split.  Every report must cover the complete preregistered eight
validation scenes from
`scannet_surface_region_query_free_validation_scenes_20260731.txt`; one-scene
runs and favorable scene subsets are invalid.

The three-seed gate requires seeds `0,1,2` by default.  SmoothL1 and MAE must
improve in at least two seeds and their paired scene-clustered bootstrap 95%
CI lower bounds for `control - candidate` must be strictly positive.  Response
profile, ranking, top-decile, and their p05 tails use
`candidate - control`; each scene-clustered CI lower bound must be no worse
than the frozen non-inferiority tolerance (zero by default).

## CPU-only invocation

```bash
bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/materialize_surface_text_response_descriptors.py \
  --validation-cache /path/to/validation_shard0.pt \
  --validation-cache /path/to/validation_shard1.pt \
  --readout-checkpoint /path/to/readout_seed0.pt \
  --radio-checkpoint /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar \
  --method-id control --device cpu \
  --output /path/to/control_seed0.descriptors.pt

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_text_response_fidelity_gate.py evaluate \
  --descriptors /path/to/control_seed0.descriptors.pt \
  --text-bank /path/to/target_blind_text_embeddings.pt \
  --text-bank-manifest /path/to/target_blind_text_embeddings.manifest.json \
  --query-split dev --output /path/to/control_seed0.response.json

bash radio_gs/scripts/run_repo_python.sh \
  radio_gs/scripts/eval_text_response_fidelity_gate.py gate \
  --control-report /path/to/control_seed0.response.json \
  --control-report /path/to/control_seed1.response.json \
  --control-report /path/to/control_seed2.response.json \
  --candidate-report /path/to/candidate_seed0.response.json \
  --candidate-report /path/to/candidate_seed1.response.json \
  --candidate-report /path/to/candidate_seed2.response.json \
  --phase dev \
  --output /path/to/paired_seed_gate.json
```

## Remaining external artifacts

The CPU evaluator and gate are complete.  A real promotion decision remains
blocked until the active Surface screen has produced all paired validation
caches and readout seeds.  Missing artifacts must not be guessed or
substituted with a benchmark-derived query cache.
