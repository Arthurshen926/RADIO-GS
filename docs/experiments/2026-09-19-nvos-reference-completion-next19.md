# NVOS next19: reference-object evidence bridges spatial fragments

The previous spatial-component candidate reaches79.3864% IoU but removes valid
extent in fern/orchids. This bounded control asks whether authorized reference
object evidence can join otherwise disconnected 3D pieces without treating
SAM-complement pixels as negatives or using target labels to choose a policy.

## Why reference evidence, not unbound multiview pseudo-labels

The scene stores multiview teacher/episode caches, but the frozen online readout
forbids training-supervision reads. More importantly, those regions do not carry
verified identity for this actual query. This experiment therefore uses the
already authorized reference SAM binary mask and original positive scribble.
It does not invent query identities for unrelated source proposals.

## Registered two-arm intervention

Compute 8-connected components of the reference SAM mask, retaining only those
that intersect original positive scribble pixels. Lift that support using the
actual reference renderer's adjoint, normalized by each element's full reference
contribution. Retain the prior spatial graph and explicit positive anchors.

- `reference_component`: additional anchors have >=0.5 reference contribution
  from the prompt-connected SAM components.
- `reference_core_component`: first erode the selected 2D components using a
  fixed5x5 kernel, then require >=0.9 contribution purity.

These extra anchors can preserve spatial components that sparse scribbles alone
missed. Candidate nodes remain original above-threshold nodes plus admitted
anchors. No SAM-complement negative labels are created. The original membership
values are preserved on retained nodes, followed by the same positive-only
reference repair. Neither teacher connection nor a high contribution fraction
proves real object identity: SAM can merge instances or parts, and a reference
view cannot verify hidden target parts.

Both schemes use the same k16 spatial graph and distance cap3x median nearest
neighbor distance. There is no per-scene parameter selection, risk-threshold
sweep, or target-based gating. All eight tasks run both registered schemes.
Each task shares a single base query optimization across arms. Element states
are now saved as `*_elements.pt`, bound to geometry, field and query receipt,
so subsequent experiments can reuse them without repeating GPU optimization.

Authority: `output/nvos_reference_completion_next19/registration.json`.
Evaluator: `scripts/audits/nvos_reference_completion_next19_evaluate.py`.

## Evaluation contract

Same positive-only inputs, frozen geometry/field, >=0.5 threshold and continuous
bilinear scoring-grid resize. Negative scribble, target RGB and training
supervision reads are forbidden. All variant predictions and temporary element
states are saved before target GT becomes readable. Old79.3864% spatial masks
must match at both prediction and scoring grids; continuous cold numeric error
is explicitly reported with the existing1e-6 bound. Frozen defaults and earlier
seals remain untouched.

## Complete eight-task result

| Task | Spatial baseline | Reference component | Eroded reference core |
|---|---:|---:|---:|
| fern | 65.62% | 75.33% | 75.00% |
| flower | 82.71% | 82.77% | 82.71% |
| fortress | 93.22% | 93.77% | 93.77% |
| horns_center | 91.77% | 93.44% | 93.42% |
| horns_left | 89.07% | 89.07% | 89.07% |
| leaves | 69.38% | 69.38% | 69.38% |
| orchids | 65.12% | 66.47% | 66.45% |
| trex | 78.20% | 78.33% | 79.49% |
| **Macro** | **79.39%** | **81.07%** | **81.16%** |

Macro AP: 90.11% / 91.44% / 91.51%.

The reference-component arm gains1.6822IoU points with seven increases and one
tie. The eroded-core arm gains1.7744points with six increases, one tie and a
flower decrease of0.0001848points. These are two fixed global methods, not a
per-scene winner selection. Neither frozen default nor earlier seals change.

Fern improves65.62→75.33/75.00%, recovering the component-cut loss and exceeding
the old unpruned prompt-SAM73.98%. Horns and leaves retain their large spatial
exclusion gains. Trex improves78.20→78.33/79.48%, rather than reverting to the
old SAM-only over-extension.

Orchids improves only65.12→66.47/66.45%, still below the old unpruned82.14%.
Its reference SAM has166connected components, only two intersecting explicit
positive scribbles. This shows a remaining ambiguity: additional disconnected
teacher pieces may be legitimate parts or unrelated regions. Reference connection
alone does not resolve their identity. No unverified component is added merely
because a benchmark label would reward it.

All eight spatial-baseline IoU and pixel-accuracy values reproduce the prior
authority exactly. Binary masks match both grids; maximum continuous cold error
is 2.98e-07.

All three arms on all eight tasks took 444.5seconds end-to-end including cold loading and evaluation.

Seal: `paper/artifacts/nvos_reference_completion_next19_eight.json`.

## Runtime state ownership

The audit runs one query per process and shares a base optimization within that
process. That audit cache must not be reused in a multi-query server. The new
`nvos_reference_completion_next19_runtime.py` instead accepts explicit pre-component
`prompt_sam` scores, masks and camera/field state and has no global query cache.
An explicit `base_stage` check rejects pruned outputs supplied as an upstream base.

Saved cohort element outputs each have a sidecar binding the field, geometry,
positive prompt, SAM mask, query receipt, parameter registration and stage. The
loader rejects a horns-center state requested under horns-left identity even
though both share the exact same field and geometry. These cohort files are
post-component outputs, not reusable pre-component base scores. Input identity
checking alone is not claimed to prove query-batch isolation; a real A/B/A runtime
execution is recorded separately.

## Actual runtime isolation check

The pure interface executes real horns_center, horns_left, horns_center queries
on the same frozen map. Repeated A outputs are bitwise identical after B; the
input base scores and persistent latent tensor remain unchanged. Both runtime
outputs agree with sealed cohort element scores within2.98e-7/1.19e-7, with zero
selected-element differences. No target RGB, GT or negative prompt is read.

Reusable pre-component prompt_sam states for these two queries are separately
persisted with full identity bindings. Combined with the wrong-query load
rejection, this verifies the specific query-state isolation concern. It is not
a claim of complete production-service certification.

Runtime/state seal: `paper/artifacts/nvos_reference_completion_next19_runtime.json`.
