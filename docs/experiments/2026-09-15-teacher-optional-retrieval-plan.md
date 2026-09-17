# Teacher-optional element retrieval: next implementation sequence

User requirement: no particular VFM is mandatory, including RADIO. Removing
RADIO must leave a runnable construction/training/query/cold-load path with
other configured teachers. This is an implementation requirement, not a claim
that removing any teacher preserves all performance, or that zero teachers
and zero labels provide semantic supervision.

## Current gaps

The Gaussian teacher builder still requires a RADIO feature manifest and
uses positive raw RADIO amplitude to gate semantic support. Several field
loaders and training losses assume a RADIO-shaped target and affine decoder.
Disabling a RADIO loss is therefore insufficient. These paths must be made
teacher-optional under a new explicit contract; historical replay contracts
remain intact.

## P0: close geometry and seal retrieval interfaces

Use the existing independent-scale geometry development bundles, including
their documented coverage/reference limitations. Seal full Gaussian state,
camera intrinsics/poses, clipping planes and source splits in every training
artifact and cold-load path. Connect the new geometry loader to the actual
field finetuner, replacing xyz-only acceptance for the new contract.

Persistent state: one learned per-element latent plus separately defined
evidence/reliability state. Queries produce element membership scores through
a query encoder, optional modality adapter and compatible map projection.
Geometry visibility is independent of query/feature confidence. Keep
independent membership/unknown support; do not normalize arbitrary overlapping
queries into a categorical partition.

## P1: remove mandatory teacher dependencies

Define explicit teacher records: feature space, encoder checkpoint, native
resolution, normalization, valid support, dimensionality and supervision type.
Geometric visibility comes only from geometry. Each teacher owns its own
availability/validity; missing evidence is unknown, not a negative.

Support initialization from a selected configured teacher or random latent
initialization. Freeze pretrained query encoders by default and train small
matching adapters. Learned map space need not equal a native teacher space;
each prediction head/adapter must define which spaces it connects.

Acceptance: run with RADIO checkpoints and caches unavailable, not merely
weight zero. Build source episodes, train, cold-load and execute text/image/
registered-prompt queries using configured alternatives. Repeat with a
different teacher omitted. Failure to retain performance is reported
separately from structural dependency failure.

## P2: build query episodes and retrieval-first losses

Construct source-only object/region episodes: query view, target views,
query modality, positive/negative/unknown support, visibility, identity and
teacher provenance. Use cross-view verified membership; non-observation is
not evidence of absence. Separate unseen-view and unseen-query/instance
validation; keep a final untouched test partition.

Primary losses consume the actual inference scores:
- identity ranking / positive-vs-negative retrieval;
- element membership supervision on known support;
- rendered cross-view mask loss using the same element posterior;
- positive/negative prompt constraints and supported null/rejection cases;
- cross-modal consistency only for queries proven to denote the same target.

Teacher features, local relations and boundaries are auxiliary constraints.
RADIO backbone is one optional teacher, independent of RADIO adaptor outputs.
Do not average embeddings from incompatible teacher spaces. Weight/detach
teacher branches explicitly so a teacher cannot change physical opacity.

## P3: establish an uncompressed retrieval control

Train field latents and matching adapters on frozen geometry before evaluating
compression. Compare feature-only supervision, retrieval-only supervision and
retrieval plus auxiliary teachers under identical geometry/splits/budgets.
Track training, validation and final test separately; prohibit selecting
hyperparameters on already inspected final holdouts.

Report text/image/prompt retrieval ranking and membership IoU, shared-posterior
2D/3D results, coverage, false positives, null calibration and boundary metrics.
Report conditional and full-support results so empty or missing regions
cannot artificially improve scores. Add perturbation tests for query order,
query subsets and geometry immutability.

## P4: compression and optional instance refinement

Compress only after uncompressed heldout retrieval is stable. Compare compact
against uncompressed results for each modality and teacher ablation. Add an
instance codebook only if residual errors demonstrate that instance
separation/extent is the bottleneck; preserve soft overlap and unknown
membership, and evaluate whether it improves heldout retrieval.

## Required experiment matrix

Same geometry and fixed train/validation/test partitions:
1. available non-RADIO teachers, no RADIO files;
2. RADIO only;
3. RADIO plus selected native teachers;
4. selected-teacher initialization versus random initialization;
5. feature-only versus retrieval-first with auxiliary supervision.

This matrix tests both teacher independence and the empirical value of each
teacher. It does not assume native teachers always outperform RADIO.

Visual-review addendum: matched renders show reduced small-object depth relief
in figurines as well as holes. P0 therefore includes a stagewise geometry
detail check (calibrated depth -> points -> fusion -> raster), alongside the
training-loader integration. Current bundles remain development carriers;
mean depth gains do not establish fine-detail acceptance.
