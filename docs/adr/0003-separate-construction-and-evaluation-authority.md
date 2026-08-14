# Separate construction authority from evaluation authority

Status: accepted for ScanNet-UQIS-9 v0.1.

## Context

Official dataset construction can be complete before method execution is
sufficiently isolated for a public formal evaluation. Treating those states as
one boolean would either block legal mapping work or let a construction pilot
be laundered into a formal benchmark row.

## Decision

UQIS seals a content-addressed `Construction Authority` after independently
verifying the frozen cohort ledger, Nr3D annotations, ScanNet inputs, scene
derivation receipts, query exclusions, target records, and the evaluator-only
construction candidate release. This authority permits method mapping against
the frozen field-frame inventory. It does not expose formal query bundles,
open evaluator labels, or authorize metric release.

Evaluation authority remains a later fail-closed gate. It additionally
requires an evaluator-private query-ID refreeze, physically separated method
bundles, per-query sandbox receipts, a complete method field inventory, sealed
predictions, and an evaluator-owned one-shot ledger.

## Consequences

The nine-scene cohort and all construction inputs can no longer drift while
LUDVIG fields are built. A sealed construction may truthfully be called
construction-formal, but no metric or benchmark row may be called formal until
the separate evaluation authority is enabled.

