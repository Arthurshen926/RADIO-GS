# RADIO-GS

RADIO-GS reconstructs a query-independent RADIO representation into a compact Gaussian scene field for reusable scene understanding. This glossary defines the small set of domain terms that active method and benchmark work should share.

## Representation

**Canonical Capability Feature**:
The sole persistent semantic feature owned by each Gaussian and used as the source for every downstream query.
_Avoid_: task feature, capability bank, semantic sidecar

**Single Compact Feature Field**:
A scene representation in which every Gaussian owns exactly one Canonical Capability Feature; task-specific semantic fields are excluded.
_Avoid_: multi-head field, task field, hybrid feature bank

**Universal Field v1**:
The versioned Single Compact Feature Field containing one factorized D512/L512 RADIO feature and its five query-independent reliability scalars per Gaussian; query readouts are not part of its identity.
_Avoid_: Method-v1, final method, task field

**Deployment Scene State**:
The persistent per-scene state required to execute a query from a cold start, including Gaussian rendering state and the Canonical Capability Feature.
_Avoid_: deployment cache, training checkpoint

**Training Artifact**:
State used to construct or optimize the field but unavailable to cold-start query execution.
_Avoid_: deployment dependency, hidden scene state

**Capability View**:
A query-independent working representation derived from the Canonical Capability Feature for semantic, appearance, or boundary reasoning.
_Avoid_: capability bank, task descriptor

**Query-Native Universal Gaussian Memory Candidate**:
The proposed v2 representation contract in which one persistent Gaussian
latent is supervised at query-response/posterior level and is not required to
decode into a named foundation-model feature space.  It remains a candidate
until ADR 0003 promotion gates pass.
_Avoid_: Universal Field v2, promoted field, multi-feature field

**Structured Universal Gaussian Memory v3 Candidate**:
The proposed single-field representation in which one persistent D512 latent
and five reliability scalars per Gaussian jointly retain visual semantics,
scale-conditioned instance membership, boundary structure, and reliability.
It is trained through source-heldout 3D-membership rendering and remains a
candidate until ADR 0004 promotion gates pass.
_Avoid_: v2 readout extension, instance sidecar, benchmark memory

**Shared 3D Instance Membership**:
The one per-Gaussian object posterior selected by text, image, or prompt
identity anchors and consumed unchanged by both 3D evaluation and 2D rendering.
_Avoid_: 2D proposal, projection repair, separate 2D/3D posterior

## Querying

**Authorized Query Input**:
The text, scribble, reference mask, reference image, or point that an evaluation protocol explicitly gives to the query procedure.
_Avoid_: query hint, target assistance

**Query Workspace**:
Ephemeral state derived from one Authorized Query Input; it cannot update the scene field or carry information into another query.
_Avoid_: online scene memory, persistent prompt cache

**Canonical Query Interface**:
The shared boundary that turns Authorized Query Input and the field into one Gaussian-domain result before output conversion.
_Avoid_: benchmark query path, task head

**QueryPacket**:
A model-independent, query-transient packet containing adapter-aligned tokens,
confidence, optional registered seed probability and spatial metadata.  It is
the replaceable query-encoder boundary for the Query-Native candidate.
_Avoid_: raw VFM feature, persistent query cache, zero-shot arbitrary encoder

**Primitive Readout-v0**:
The frozen minimal causal baseline that converts primitive similarity into output with fixed thresholds and no learned extent, boundary completion, or category rejection.
_Avoid_: Method-v1, final query interface

**Typed Posterior Interface**:
A globally shared query interface specialized by query family that produces a calibrated Gaussian Query Posterior without introducing another persistent scene field.
_Avoid_: task field, benchmark-specific sidecar

**Identity--Extent Posterior**:
A typed posterior that preserves a query-to-field identity unary for localization
and derives object extent from query-independent instance/region membership.  The
extent branch may complete support but cannot move or replace the identity peak.
_Avoid_: SAM semantic classifier, unrestricted region averaging

**Bounded Categorical Region Residual**:
A low-margin-only topology residual for mutually exclusive category posteriors.
It may smooth uncertain rows using query-independent SAM affinity, but cannot
create a class identity or alter high-margin predictions.
_Avoid_: SAM category logits, global proposal propagation

**Independent Prompt Consensus**:
For registered image prompts, box and signed-point SAM hypotheses are decoded
independently and combined after mask decoding.  Prompt types are not assumed
to be additive inside the frozen SAM prompt encoder.
_Avoid_: joint box-point posterior, prompt concatenation as Bayesian fusion

**Gaussian Query Posterior**:
The calibrated per-Gaussian output of the Canonical Query Interface before raster or world-space conversion.
_Avoid_: benchmark mask, task score map

**Signed-Evidence SAM Selector**:
A query-transient selector that ranks frozen SAM proposals by inclusion of
sealed positive field points and exclusion of sealed negative field points;
field-mask overlap and SAM confidence are tie-breaks only. It is not eligible
for contracts that forbid opening the target RGB at query time.
_Avoid_: field repair, strict-unseen readout, target-mask calibration

## Evaluation

**Five-Benchmark Program**:
The joint RADIO-GS objective covering LERF-2D, LERF-3D, ScanNet OVS, NVOS, and Available-Nine SPIn-NeRF with one method family.
_Avoid_: six-task program, UQIS program

**Available-Nine SPIn-NeRF Cohort**:
The orchids, leaves, fern, room, horns, fortress, pinecone, truck, and lego scenes; the unavailable Fork scene is excluded.
_Avoid_: full-ten SPIn-NeRF

**Evaluation Contract**:
A frozen definition of a benchmark cohort, legal information, output domain, metrics, aggregation, and comparator.
_Avoid_: evaluation setting, benchmark setup

**SOTA Target**:
A dated numerical target tied to one Evaluation Contract and comparator identity.
_Avoid_: competitive result, SOTA-level

**Development Evidence**:
A result used to choose or revise the method before the final frozen evaluation.
_Avoid_: final result, blind confirmation

**Joint Development Baseline**:
One complete method identity evaluated across all five benchmarks and used as the common comparison point for subsequent candidates.
_Avoid_: per-task best bundle, virtual incumbent
