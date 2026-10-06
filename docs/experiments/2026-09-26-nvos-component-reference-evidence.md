# NVOS: component retention from measured reference evidence

This is one registered, positive-only development control on the same eight
tasks and frozen maps. It is not a new independent test, an input-budget-matched
LUDVIG reproduction, or a jointly promoted four-query method.

## Method problem and correction

The retained reference-core runtime admits extra 3D anchors only from SAM
fragments connected to the original positive scribble in the reference image.
It then sets every other candidate spatial component to zero. This is a
topological identity assumption: a disconnected image fragment cannot belong
to the requested scope. Occlusion and separated parts can violate it even when
the authorized reference SAM mask already supplies positive evidence there.

The new control preserves the old anchored components and additionally retains
a whole positive-candidate spatial component when its *aggregate actual
renderer-adjoint contribution* lies at least 90% inside the full eroded
reference SAM mask and provides at least one effective pixel of foreground
mass. The existing 5x5 erosion, k16 spatial graph, radius, query field, positive
repair, threshold and rendering are unchanged. The 0.9 purity comes from the
existing reference-core admission; there is one registered rule, no sweep.

Aggregation precedes normalization. A tiny visible tail with foreground ratio
one cannot establish identity; an invisible component cannot pass. This avoids
restoring every disconnected semantic-positive component. SAM remains a
fallible teacher hypothesis: passing this gate does not verify physical identity.
Pixels outside SAM create no new negative label, and this additive correction
does not otherwise remove the prior runtime's topological exclusion assumption.

Implementation: `radio_gs/querying/component_reference_evidence.py`, with the
explicit FP32 renderer-adjoint boundary in
`radio_gs/querying/component_reference_evidence_numeric.py`. The pure runtime
receives query-bound pre-component scores and never caches another query's
state. Neither field weights nor geometry are updated.

## Numeric correction and interrupted execution

The first strict subset-mass check failed before target scoring in flower.
Separate CUDA adjoint reductions produced foreground-over-total errors at four
elements, at most 0.000244 effective pixels on totals 345–601 pixels; maximum
relative excess was 8.55e-7. The retained diagnostic is
`output/nvos_component_evidence_next31/flower/evidence_numeric_diagnostic.json`.

The v2 boundary allows and clamps only errors within 64 FP32 epsilon times
`max(total_mass, 1)`. Larger inconsistencies still fail closed. No method
threshold changed. Seven focused tests cover disconnected positive support,
absolute evidence mass, aggregation-before-normalization, candidate support,
input immutability, numerical roundoff, and real partition violations.

An external conversation interruption stopped v2 after flower and horns_left.
The resume runner verifies and copies those two completed artifacts unchanged,
keeps original partial directories, and executes the remaining six tasks in a
new output root. It records both the original registration and resume source
hashes. Partial attempts are not counted as finished scenes.

## Evaluation contract

Authority: `output/nvos_component_evidence_next31_v2_resume/registration.json`.
Summary, when complete:
`output/nvos_component_evidence_next31_v2_resume/summary.json`.

Each task must exactly reproduce the old 81.1608% reference-core baseline mask
at the prediction grid and at the scoring grid. Continuous differences have
the inherited 1e-6 tolerance. Negative scribbles, target RGB, and source teacher
caches are denied. Predictions for both arms are saved before target GT becomes
readable. Continuous bilinear resizing before the fixed >=0.5 decision remains
unchanged. Results and conclusions are appended only after all eight finish.
The final check passes nine tests: seven component-evidence/numeric tests and
two existing query-input identity tests.

## Complete eight-task result: rejected

| Task | Reference-core baseline IoU | Component-reference candidate IoU |
|---|---:|---:|
| flower |82.7127%|82.7127%|
| fortress |93.7727%|93.7727%|
| fern |74.9983%|74.9983%|
| horns_center |93.4228%|93.4228%|
| horns_left |89.0657%|89.0657%|
| leaves |69.3769%|69.3769%|
| orchids |66.4520%|66.4520%|
| trex |79.4852%|77.9059%|
| **Macro** |**81.1608%**|**80.9634%**|

Macro AP is91.5057→91.4251%; pixel accuracy96.4790→96.4432%.
All eight old baseline prediction/scoring masks replay exactly. The candidate
is not elementwise identical: fern adds one selected element, invisible in the
target raster, and trex adds169 selected elements across103 components. The
other six element outputs are unchanged. Trex changes2,240 pixels on the
prediction raster and34,406 pixels on the larger scoring raster. No per-task
winner selection is performed; the rule is rejected for promotion.

This directly shows that even a high-purity positive reference footprint is
insufficient to certify useful cross-view extent. It does not separately prove
whether the trex error originates in SAM scope, geometric correspondence, or
previous semantic false positives: that would require additional independently
verified evidence. Do not lower purity thresholds to chase the orchids score.

## Reference-only failure localization

The additional orchids diagnostic reads neither target RGB nor target labels.
Of157 positive-candidate spatial components,94 are retained and63 are removed.
Of those63,38 have less than one effective reference pixel of visibility; all
remaining25 have eroded-SAM foreground purity below0.5. There are no removed
individual elements with both purity>=0.9 and at least one foreground pixel.
The dominant mixed component contains301 elements but only8.80% of its visible
mass falls inside the eroded reference foreground.

The removed components contain only2.127% of candidate reference-core mass.
Thus the source reference can still look nearly satisfied while another view
loses valid extent. The diagnostic distinguishes absent reference information
from confidently foreground-only components: there is no pool of high-purity
disconnected components to recover by this rule. Object identity under weak
visibility and teacher/geometry disagreement remains unresolved.

Detailed output:
`output/nvos_component_evidence_next31_v2_resume/orchids_reference_component_diagnosis.json`.
The diagnostic explicitly restricts counts to candidate components. The earlier
runtime fields named `invisible_candidate_components` and
`subpixel_candidate_components` count all labels in the full observed-node
graph, including isolated noncandidate nodes; do not interpret those fields as
counts of candidate objects. This reporting limitation does not enter admission
or prediction, and the dedicated diagnostic reports the correct population.

Final artifact binding:
`paper/artifacts/nvos_component_reference_next31.json`.

## Separate selected-output supervision issue found in this audit

`train_selected_scope_final_next28.py` penalizes an isolated selected-opacity
silhouette against a visible source SAM mask over every pixel. Removing
nonselected occluders exposes hidden selected geometry, so those pixels cannot
automatically supply negative physical-identity supervision. Earlier frozen
diagnostics attribute 48–55% of selected-only false-positive pixels in the two
diagnosed scenes to removing nonselected occluders. That is a projection
diagnostic, not proof all exposed elements belong to the target.

This motivates visibility-conditioned negative supervision while retaining the
official selected-render metric for reporting. The NVOS control above neither
changes that LERF metric nor claims to solve hidden 3D object identity.
