# RADIO-GS

RADIO-GS reconstructs a query-independent RADIO representation into a compact Gaussian scene field for reusable scene understanding. This glossary defines the small set of domain terms that active method and benchmark work should share.

## Representation

**Canonical Capability Feature**:
The sole persistent semantic feature owned by each Gaussian and used as the source for every downstream query.
_Avoid_: task feature, capability bank, semantic sidecar

**Single Compact Feature Field**:
A scene representation in which every Gaussian owns exactly one Canonical Capability Feature; task-specific semantic fields are excluded.
_Avoid_: multi-head field, task field, hybrid feature bank

**Deployment Scene State**:
The persistent per-scene state required to execute a query from a cold start, including Gaussian rendering state and the Canonical Capability Feature.
_Avoid_: deployment cache, training checkpoint

**Training Artifact**:
State used to construct or optimize the field but unavailable to cold-start query execution.
_Avoid_: deployment dependency, hidden scene state

**Capability View**:
A query-independent working representation derived from the Canonical Capability Feature for semantic, appearance, or boundary reasoning.
_Avoid_: capability bank, task descriptor

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

**Gaussian Query Posterior**:
The calibrated per-Gaussian output of the Canonical Query Interface before raster or world-space conversion.
_Avoid_: benchmark mask, task score map

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
