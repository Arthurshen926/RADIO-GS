# PFIR and AGILE3D Qualitative Visualization Design

## Goal

Produce reproducible static PNGs and a Markdown report that show the actual
inputs, predictions, ground truth, and error masks for the frozen
ScanNet-PFIR-Small v1 and AGILE3D ScanNet40 evaluations.

## Scope

The report is evidence for the existing formal runs. It must not change
training, feature caches, query settings, thresholds, click policy, or test
metrics. Ground-truth instance IDs are opened only by the offline
visualization/evaluator path and are clearly labelled as such.

## Inputs and Outputs

PFIR reads the frozen held-out query crop, the saved mesh-aligned ranking and
selection predictions, and the benchmark evaluator manifest/official mesh.
Each PFIR row shows the crop, target GT, predicted mask, and TP/FP/FN overlay.
It also reports the frozen Track-A rank and Track-B IoU.

AGILE3D reads the formal field-capability cache and official PLY/object list.
It deterministically replays a small fixed set of official click trajectories
with the exact formal predictor configuration. For click counts 1, 5, and 15,
each row shows the GT target, prediction, TP/FP/FN overlay, and the signed
clicks. The replay must match the stored formal per-object IoU trajectory.

## Case Selection

PFIR cases are selected from the saved formal per-query records by three
fixed predicates: successful ranking and mask, rank-1 with low mask IoU, and
same-category hard-query confusion. Ties are resolved by `query_id`.

AGILE3D cases are selected by fixed formal-result strata: a high-coverage
success, a median-coverage middle result, and a low-coverage failure. Ties
are resolved by `(scene_id, object_id)`. The report records the exact selected
cases and their metrics in an audit JSON, so no manual cherry-picking occurs.

## Rendering Contract

Meshes/point clouds use a deterministic camera based only on scene geometry.
The first RGB panel in every row keeps the full-scene view. GT, prediction,
and error panels use a target-centered crop of that same fixed projection to
make small objects readable. The crop is anchored by the evaluator-only GT
mask and is explicitly labelled as such; it cannot change the view direction,
method input, click policy, prediction, or metric. Green means true positive,
red false positive, gold false negative, cyan prediction-only support, and
blue/red markers are positive/negative clicks. Every panel contains its
scene/query identity and the corresponding formal metric values. Rendering
never influences prediction.

## Validation

The generator verifies required files and vector lengths, asserts that PFIR
saved vectors align to the official mesh, and asserts AGILE replay trajectories
equal the selected rows in `formal_v1/results.json` within numerical tolerance.
Unit tests cover deterministic case selection, error-color assignment, and
trajectory equivalence checking. Generated images are inspected visually and
the report links only files that exist.
