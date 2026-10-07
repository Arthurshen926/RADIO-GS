# Physical-scale member energy: fixed development protocol

Fixed before new fits or scores on 2026-10-07. Starting point: commit 990d2c9,
training-supported inferred anchors and the complete observed output carrier.
This is a bounded mechanism experiment, not an independent final benchmark.

The existing member logits depend on a scale-gated group cosine but not spatial
distance. Equal group codes at distant positions can receive equal membership.
Test one common energy for text, image, 2D rays and registered 3D supports:

`logit(i | S,s) = temperature * (cos(group_s(i), prototype_s(S)) - bias)
                  - ||x_i - weighted_mean(x_S)||^2 / (2 * (s * units_per_metre)^2)`.

The spatial term is a Gaussian log prior with physical scale as its standard
deviation. Its coefficient is fixed at 1/2; no threshold/strength/radius sweep.
Geometry units are taken from the construction manifest and stored in the scene
export, not inferred at query time. The term is applied inside the same member
kernel, including negative supports. Exact registered hard constraints retain
their existing semantics. Native channels, candidate policy and output carrier
are unchanged. This Euclidean prior is deliberately simple and may truncate
long, curved or uncertain-depth objects; its suitability must be measured.

Two arms on figurines and ramen: (1) add the prior to the existing frozen model,
(2) train from the same seeded initialization for the same 2,000 source-only
subset-alpha updates with the prior present. Both use the existing eighteen
feature-training views, their exact cached native coordinates and source masks.
No extra source views, teachers, labels, per-query scales or benchmark-specific
readout changes. Frozen controls separate an inference restriction from changes
learned through the source objective. Previous supported-anchor exports are the
paired baseline.

All four text prediction packages must precede LERF scoring. Use exact previously
encoded native query vectors and verify native response/geometry parity. Keep
the retained scorer (including 8-bit selected alpha > 10). Report actual subset
IoU and precision/recall, native and member semantic IoU separately. Evaluate all
development teacher regions, retaining unsupported cases as zero; report image,
2D and 3D inputs separately, including privileged scope only as a diagnostic.

Promotion gate for a common candidate: actual text subset IoU improves on both
scenes and no default-scale image/2D/3D teacher-region mean regresses on either
scene. If one fixed arm passes, confirm that same arm on teatime without retuning.
If both pass, prefer fitted metric only if its mean actual text subset IoU is
higher; otherwise prefer the frozen control. No outcome-driven parameter sweep.
If neither passes, retain the previous supported-anchor candidate and record
the failure as evidence about spatial locality, scale and training/readout.

The existing v10 contract and resource differences remain explicit. Report the
same-scene v10 gap, not the five-task macro as a matched comparison. Standalone
four-input export/reload and existing CUDA hidden-element gradient checks remain
required before treating an arm as a usable opt-in implementation.
