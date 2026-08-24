# Lift native capabilities at the level consumed by the query

## Context

Universal Field v1 reconstructed a compact RADIO representation and recovered
several downstream capabilities through frozen heads.  Exact marginal lifting
is linear, while native DINO, SAM and SigLIP operators are not.  In general,
applying a capability head after multi-view averaging is not equivalent to
running it on each legal source observation and lifting the resulting
statistic.  ScanNet capability-before-MPR experiments demonstrate this gap.
LERF and registered-prompt diagnostics additionally show that feature
similarity, object identity and object extent are different variables.

## Decision

RADIO remains the shared visual anchor and Universal Field v1 remains the
current persistent D512/L512 representation.  RADIO is no longer assumed to be
the exclusive sufficient intermediary for typed readouts.

Native teachers are permitted when all of the following hold:

- they run on Evaluation-Contract-authorized source or query RGB;
- their output is lifted with exact front-to-back marginal responsibility at
  the semantic level consumed by the query;
- they add no second Gaussian-indexed semantic table;
- any learned decoder is scene-global and source-gated;
- text identity, object extent, categorical identity and registered-prompt
  extent keep separate ownership.

Dense native teachers first undergo a capacity-matched A/B/C sentinel:
frozen latent/global decoder, RADIO-anchored latent update and native-only
latent update.  A full field rebuild is allowed only when updating the shared
latent improves disjoint source-heldout capability.  Object-level SAM and
SigLIP statistics instead undergo source-heldout membership or a fixed
multi-scene categorical transfer gate.

The fixed ScanNet native SAM3-extent plus native SigLIP2-region residual passes
that transfer gate on top of the already promoted L512 direct-capability
decoder and is part of the typed categorical compiler.  The DINO-only A/B/C
sentinels select the frozen-latent arm and therefore do not authorize field
retraining.  LERF native-DINO proposal gating and the native multimodal
membership decoder remain gated diagnostics unless their source-heldout object
reconstruction gate passes.

## Consequences

The method remains one compact persistent Gaussian latent with typed readouts,
not a bank of DINO, SAM, SigLIP and RADIO fields.  Source-native capability
statistics can improve a readout without redefining Universal Field v1.
Rejected native teachers remain causal diagnostics and cannot be selected by
benchmark metrics.  Registered-prompt RGB-assisted results must continue to be
reported separately from source-only evidence.  A future Universal Field v2
requires a new ADR and a full capacity/storage/five-benchmark validation.
