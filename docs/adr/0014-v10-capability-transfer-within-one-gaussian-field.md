# Recover v10 capabilities within one Gaussian field

Status: research direction authorized by the user on 2026-10-08; mechanisms below
are requirements or hypotheses, not a claim of recovered accuracy.
Supersedes overly restrictive implementation choices in ADR0013, not its core.

## Decision

The user explicitly permits adjusting the ideal architecture to absorb v10's
successful experience, while retaining its core and recovering full performance.
The design remains one scene-specific Gaussian capability field, three teacher
roles (SigLIP semantics, DINO local appearance, SAM grouping), shared conditional
membership, and standalone standard inference from the field plus current input.

Unification means shared scene state, evidence semantics and conditional spatial
relations. It does not require every task to discard its native evidence and
argmax the same scalar affinity. Typed query adapters and small task-semantic
readouts are permitted; they cannot own separate scene maps or silently call old
solvers. Category competition must consume native category evidence as well as
common structure. Instance scope must consume identity uncertainty as well as
positive/negative spatial evidence. One Gaussian posterior feeds 2D and 3D output.

The initial full-native, one-mean-vector-per-channel implementation is not a
capacity upper reference: averaging may already have removed decisive evidence.
If functional retention requires multiple appearance modes, reliability, or
scene-specific decoding capacity, these can belong to the same Gaussian codes
and declared shared decoder. Their utility and storage must be measured. Do not
choose a slot count or compress before identifying what information must survive.

## What is transferred

1. **Native query functionality.** Preserve response ordering and local matching,
   not only reconstruction cosine. v10's bidirectional patch matching and
   separate native-channel ranks are explicit capability references. A mean
   Gaussian vector with maximum query-patch cosine is not an equivalent transfer.
2. **Conditional membership evidence.** Learn view-balanced co-membership,
   joint visibility versus missing evidence, nested-mask disagreement and metric
   relations from legal construction observations. SAM masks are partial and
   hierarchical; unseen or unmatched support is not negative ground truth.
3. **Identity and extent.** Retain alternatives and native identity evidence
   through conditional scope selection. Known-positive point competence is
   measured separately before uncertain text/image conditions. Adding query
   statistics or blending wrong anchors is not an already successful solution.
4. **Typed decisions.** Category partition, category union and single reference
   instance are distinct set questions over the same field. A category logit,
   ordinal identity rank and member affinity are not interchangeable quantities.
5. **Training and rendering.** Match deployed observation/query conditions;
   cover all training regions and record exposure. Retain protected writes,
   valid affine query algebra and actual subset transmittance recomputation.

Source RGB, source masks, tracks and full per-view patch arrays remain construction
artifacts in standard mode. Source-derived relation targets may supervise the
field offline. Using the 22-room labelled v10 member model, including distilling
its predictions, would inherit that extra supervision and requires a separately
declared experimental condition. It is not part of the source-only track.
The attachment's explicitly described NVOS source-assisted extension can be
studied separately through the same core, with full dependencies and costs; the
current standard-mode comparison must not silently acquire those inputs.

## Evidence and experiment order

The [transfer plan](../method/v10-capability-transfer.md) maps each v10 mechanism
to its evidence, current deficiency, proposed realization and retention test.
The five-task baseline and full attribution are complete and immutable. A local
mechanism test may follow that coverage, but cannot replace joint validation.

First implement and verify the native identity primitives against the old exact
operators. Run a fixed, full-67-image UQIS intervention on existing Gaussian
fields to test ordinal fusion and bidirectional field-set correspondence.
This isolates query-side alternatives without retraining or changing resources;
it does not reproduce v10's source patch bank and cannot certify representation
adequacy. Existing same-eight-candidate oracle IoU22.13% already prevents claiming
that reranking those candidates alone can recover v10 image IoU39.90%.

Then establish a matched legal-source high-capacity reference and functional
construction losses, followed by known-identity relation learning and typed
decoding. Keep source budget, teacher/label conditions, representation and readout
interventions separate. Each accepted common method runs all five cohorts and
all four UQIS modalities; no per-query, scene or benchmark winner bundle.

Full v10 performance remains a measured goal. Changed contracts may limit what
is recoverable; no guarantee or top-conference novelty follows from this design.
Compression and speed optimization follow capability recovery, not precede it.
