# LERF3D implementation and intervention audit — 2026-08-07

## Outcome

The sealed Figurines D0--D5 run isolates the primary structural bottleneck.  A
single selected SurfaceRegion has high purity but inadequate coverage, while a
union of at most eight regions has a very high development oracle ceiling:

| diagnostic | fixed IoU | AP | oracle IoU | purity | coverage |
|---|---:|---:|---:|---:|---:|
| D0 AcceptedV2 | 0.475389 | 0.742445 | 0.648193 | -- | -- |
| D1 O4 through current readout | 0.525433 | 0.799956 | 0.674679 | -- | -- |
| D2 best single region | 0.376714 | 0.732869 | 0.644094 | 0.918888 | 0.397277 |
| D3 up-to-eight-region union | 0.771298 | 0.961138 | 0.863364 | 0.961768 | 0.801454 |
| D5 direct thresholded teacher | 0.142299 | 0.540459 | 0.485668 | 0.708842 | 0.145516 |

D3 selects 6.19 regions on average, has median eight, and reaches the fixed
eight-region limit for 11 of 21 queries.  Therefore the current region basis is
not too weak, K=2 descriptor modes are not the priority, and changing the
renderer or graph/radius parameters is not justified.  The missing capability
is a target-blind query-to-set readout that can select complementary regions.

The D0--D5 numbers are development oracles, not deployable benchmark results.
Their sealed result is
`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/lerf3d_support_readout_oracle_d0_d5_figurines_v2/sealed_run/result.json`
with SHA-256
`775df3adff6ac95aba1dd3061e6202fa8501405bdc768d1a1e2cf2b951c2aba1`.

## P0 corrections completed before V2.1 training

### O4 teacher temperature alignment

The O4 oracle implementation uses normalized LSE with `BETA=10`, equivalent to
temperature `0.1`.  V2.1 originally inherited V2 temperature `0.05`, equivalent
to `BETA=20`.  This made its teacher more max-view-like than the O4 target that
motivated the loss.  V2 remains byte-identical; V2.1 now uses temperature 0.1
for positive and every canonical-negative response before the fixed relevance
logit scale 10.

- correction addendum: `paper/artifacts/source_global_response_listwise_loss_v21_o4_temperature_alignment_addendum_20260807.json`
- V2.1 implementation SHA-256 after typed-relation integration: `64c9eb2226aba193b510122b55f73e124ff497c85f54e9073c2c78803da0b370`
- frozen V2 SHA-256 retained: `552e7bf0e4d83e9346af731e6ce9eaf891968b14f32b49f728b5188c5e012ae7`

### Explicit compositional-stratum weighting

There were two incompatible interpretations of the 5,808-row compositional
bank.  One combined component would give approximately 69% of the row mass to
counterfactual attributes and below 1% to parts.  Five equal components would
instead give the 30-row fit part bank 20% of the objective.  V2.1 now requires
an explicit finite positive weight per optional component, keeps a mean inside
each component, and normalizes the following pre-metric fixed weights across
components:

- primary object nouns: 0.25
- synonym relation queries: 0.20
- lexical-sibling queries: 0.20
- counterfactual attributes collectively: 0.30
- high-precision part-of queries: 0.05

The four optional fit embedding components have now been independently
materialized and SHA-bound.  Uniform row duplication cannot alter a component's
weight.

- weight addendum: `paper/artifacts/source_global_response_listwise_loss_v21_stratified_weight_addendum_20260807.json`
- materialization receipt: `paper/artifacts/target_blind_compositional_siglip2_fit_gpu1_result_20260807.json`

### Minimal deployable multi-region candidate

`radio_gs/querying/multi_region_union_readout.py` implements the D3-motivated
target-blind candidate.  It consumes final canonical-negative region
probabilities and registered semantic-core memberships only.  Per query it:

1. applies the existing global probability gate 0.6;
2. selects at most eight regions;
3. greedily maximizes `probability * uncovered_core_fraction`;
4. breaks exact ties by smaller immutable candidate index;
5. emits the binary union of selected semantic-core primitive rows.

It has no GT, scene identity, connected-component rule, radius setting, graph
parameter, or tunable novelty coefficient.  The candidate is sealed before its
first metric in
`paper/artifacts/lerf3d_target_blind_multi_region_union_readout_preregistration_20260807.json`.
Its implementation SHA-256 is
`65bfa4c4b1726ea8cf3ac4f1e18c95e1a6372d7ee15d53501b57cc2ed6aca2bb`.

## Remaining implementation and theory gaps

### Typed relation records are now consumed

The original V2.1 embedding loader consumed query strings and embeddings but
not the compositional source JSON's `relation_records`.  This has now been
closed before training by a source-only authority with 657 synonym pairs and
167 sibling pairs.  Runtime training consumes only SHA-bound indices:

- synonym: teacher-calibrated student response-gap and relevance-gap
  distillation across primary/alias pairs;
- sibling: continuous-gap weighted same-region relevance-gap distillation,
  balanced between left-dominant and right-dominant teacher directions.

The authority is
`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260807/target_blind_typed_text_relation_authority_v1/fit_relation_indices.pt`
with SHA-256
`482e363bf31884e190b255cc0cf0996461400bcbb3cb3f8785fbc236da2702a9`.
The engineering receipt is
`paper/artifacts/source_global_response_listwise_v21_typed_relation_engineering_receipt_20260807.json`.

### P1: raw relevance is not the frozen post-readout score

V2.1 supervises raw descriptor relevance.  The current LERF path subsequently
performs KNN10 smoothing, per-query/per-scale scene min--max normalization, and
`clip(2*x-1)`.  The final fixed 0.6 threshold therefore corresponds to 0.8 in
the normalized pre-clip range; raw absolute offset and scale are removed by the
readout.  Thus V2.1 absolute relevance can improve response fidelity, but cannot
by itself guarantee fixed-threshold calibration after the current remap.

The D4/D5 result says not to remove KNN/remap immediately: direct thresholded
teacher probability is weak.  If the union sentinel improves AP/coverage but
misses the fixed operating point, the minimum next intervention is a source-only
post-readout loss or one global calibration head, evaluated through the frozen
renderer.  It must be shared across scenes and never selected on LERF labels.

### P1: trainable-domain weighting

The V2.1 complete-scene loss includes inactive/OOD fallback rows and pairs in
its reductions even though those descriptors are bitwise immutable.  The
scene0001 adaptive context has full typed-context validity, so this is not a
demonstrated scene0001 failure.  Before multi-scene training, record the active
fraction and the fraction of hard-negative units with at least one trainable
endpoint.  If inactive/OOD rows exist, exclude fully immutable units from the
optimization denominator while retaining them in full-scene validation.

### P2: D4/D5 interpretation and protocol integrity

D5 writes exact zero on teacher-unavailable rows, whereas D1 uses the registered
O0 fallback.  Therefore D5 minus D1 is not a pure renderer/readout estimate.
The sealed report correctly separates D4 available-row, D4 end-to-end abstain,
and D5 results.  D0/D1 runtime file/SHA binding was added before the sealed run.

No further D4 scale experiment is warranted: s0, s1, s2, and per-primitive max
are exactly identical.

## Result-driven minimal decision policy

| observed diagnostic branch | minimum general intervention |
|---|---|
| D3 far above D2, as observed | target-blind sparse multi-region union; keep the registered region basis and renderer fixed |
| D2 and D3 both low | replace the support basis with overlapping adaptive/query-independent supports; do not tune scene radii |
| D4 available high but D1 low | replace destructive single-peak/KNN/remap readout with a source-trained reliability-aware continuous readout |
| D4 low, as observed | retain response/listwise and compositional teacher improvement; do not treat direct teacher probability as a deployment oracle |
| primitive D4 high but D5 low | add source-only frozen-render consistency/inverse-compositing supervision; do not modify the renderer |
| AP/oracle high but fixed score low | add one source-only global calibration objective after the exact deployed readout |

## Immediate execution order

1. Materialize the target-blind union from O0/AcceptedV2 region probabilities,
   not D3 GT choices, and run only the preregistered Figurines sentinel.
2. Promote the union only if fixed IoU improves and AP/oracle IoU do not
   materially regress against an exact same-run O0 control.
3. In parallel, prepare the corrected V2.1 pilot with temperature 0.1 and the
   explicit five-stratum weights and the typed relation authority.
4. If the union sentinel passes, combine it with the source-validation-selected
   V2.1 checkpoint, then run the frozen LERF3D scene/macro protocol.

CPU validation for the modified V2.1 loss and union readout: 22 tests passed.
Frozen evaluator, metric, renderer, V2 loss, and V1 trainer were not modified.
