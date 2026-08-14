# Keep unified-query pairing evaluator-private

Status: accepted for ScanNet-UQIS-9 v0.1.

## Context

ScanNet-UQIS derives text, pose-free image, Registered 2D Point, and World 3D
Point queries from the same Unified-Query Target. Shared physical targets make
the modality comparison controlled, but a shared method-visible identifier
would let one query disclose the answer to another.

## Decision

Expose each modality with an unrelated opaque query identity and execute it in
an independent Query Workspace. Cross-modality pairing, Query Camera target
identity, private depth, official instance labels, and derivation evidence
remain evaluator-only. The 2-D and 3-D prompts identify the same physical
surface point without sharing a method-visible key.

## Consequences

The evaluator can compare four responses to exactly the same target while the
method cannot join them. Runtime orchestration must prevent cross-query state,
and public reports cannot emit the private common target key. Debugging that
reveals pairing consumes the cohort for that method lineage.
