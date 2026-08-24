# Evaluate Query-Native Universal Gaussian Memory as a v2 candidate

Status: Proposed

## Context

Universal Field v1 stores one RADIO-anchored D512/L512 feature and five
query-independent reliability scalars per Gaussian.  Native capability
teachers may supervise typed readouts, but the deployed path still frequently
decodes or reconstructs a foundation-model descriptor before answering a
query.  ScanNet score-level distillation shows that preserving the variable
consumed by a query is more effective than preserving a high-dimensional
descriptor.  LERF diagnostics additionally show that accurate identity
similarity does not imply complete object membership.

The proposed change is therefore not to add another task field.  It is to test
whether one Gaussian memory can be trained and queried directly in posterior
space:

```text
pretrained query encoder -> small modality adapter -> QueryPacket
frozen/shared Gaussian memory + QueryPacket -> Gaussian Query Posterior
```

RADIO, DINO, SAM and SigLIP remain legal mapping-time teachers and
initializers, but no longer define the required deployment coordinate system.

## Proposed decision

Introduce a Query-Native Universal Gaussian Memory candidate with these
invariants:

- one persistent Gaussian latent and query-independent reliability state;
- frozen pretrained query encoders with small replaceable modality adapters;
- one model-independent `QueryPacket` boundary;
- class-set-equivariant categorical decoding with no class-indexed parameter;
- direct Gaussian posterior output rather than decoded DINO/SAM/SigLIP
  descriptors;
- identity retrieval separated from an identity-preserving extent residual;
- registered prompt seeds are strong evidence and unknown rows remain unknown;
- source-native teachers supervise scores, rankings, masks, calibration and
  cross-modal posterior consistency at the variable actually consumed;
- scene adaptation, if required, is constant-size and query-independent.
- the default query-facing memory is the post-fusion coefficient; raw local
  codes and fixed projections of decoded RADIO are matched ablations;
- identity similarity uses a bounded learned temperature, and extent uses
  multiple identity anchors plus finite Gaussian geometry;
- binary object supervision is positive/negative/unknown, with unmatched and
  part/whole-conflicted support excluded from the loss;
- train, validation and final source-heldout views are disjoint, and test
  views never select a checkpoint or threshold.

The first categorical sentinel is intentionally transitional: it consumes an
existing scalar baseline response as an identity prior, but never emits a
1536D descriptor.  This is sufficient to test class-set equivariance and
posterior-level supervision.  It is not sufficient to claim feature-decode-
free arbitrary-query deployment.  Final promotion additionally requires a
cross-scene direct identity scorer that reproduces this scalar for unseen
queries without decoding a foundation visual descriptor.

The candidate does not supersede Universal Field v1 yet.  Field latents remain
frozen during the first validation stage.  A latent update is authorized only
after a decoder-only source-heldout gate demonstrates a residual capacity
limit.

## Promotion gates

Universal Field v2 is accepted only if all of the following hold:

1. LERF2D and LERF3D consume the same Gaussian posterior and both improve over
   their retained development rows without reducing LocAcc.
2. ScanNet 19/15/10 paper8 is noninferior to the compact score student, and
   query order/subset tests are exactly equivariant.
3. A scene-query bipartite holdout demonstrates that a query seen in other
   scenes transfers without per-query fitting.
4. The scalar identity prior can be produced directly from memory and query;
   no decoded foundation visual descriptor or fixed-vocabulary score cache is
   needed for an unseen query.
5. Image and prompt QueryPackets pass source-heldout mask, Brier, null and
   unknown gates; cross-modal consistency is enabled only for independently
   trusted same-object pairs.
6. No second Gaussian-indexed feature table is added, and storage/query-time
   accounting beats decoding multiple high-dimensional capability fields.
7. Exact/D512/D256 matched-readout comparisons identify whether remaining
   loss comes from memory capacity or posterior supervision.

## Consequences

The fixed ScanNet 44-channel score head remains a valid v1 development result
until the query-set-equivariant candidate passes all numerical gates.  NVOS's
protocol-authorized SAM3 query workspace remains frozen and is not attributed
to improved persistent-field fidelity.

LERF source SAM masks are not automatically considered cross-view object
authority.  A direct posterior decoder trained on independent same-view masks
must pass disjoint source-heldout extent reconstruction before any benchmark
execution.  Failure blocks model scaling and requires better cross-view
authority or supervision, not additional threshold search.
