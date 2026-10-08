# Protected membership and native functional construction

Fixed before this study's benchmark predictions/scores, on the existing research
branch. Prior completed implementation/results remain reproducible at `dfd107b`;
its source hash lock refers to that commit, not the evolving working tree.

## Question and three complete method arms

The figurines factorial diagnosis separated learned Gaussian relations from the
shared acceptance parameters. Across the full cohort, does protecting the latter
retain new relations without damaging actual selected sets? Independently, does
native query-function training repair some of the loss from averaging teachers?

All 30 fields / 31 scene cases and all five benchmark families are mandatory.
All arms use `native_evidence_shared_members_v2`, unchanged from the completed
study. LERF union and registered NVOS inputs delegate exactly to the same member
core. ScanNet evaluates native/member controls plus the typed category readout.
UQIS re-encodes the public inputs and retains all 268 predictions, including every
text, image, 2D-point and 3D-point query. No per-scene or per-task winner bundle.

1. `decoder_rollback`: previous cross-view Gaussian grouping codes, original
   parent's scale cutoffs, affinity bias and log temperature. No further fitting.
2. `protected_training`: original parent, previous cross-view relation recipe
   (two complete region epochs, seed 20261008, Adam .001), but only Gaussian
   grouping codes are optimized. Shared acceptance parameters remain exact.
   Region exposures and every paired source/target/anchor trace must match the
   previous joint fit. This tests co-adaptation versus posthoc block rollback.
3. `functional_native`: protected training followed by full-native functional
   construction, defined below. This is the fixed combined proposal, not an
   automatically promoted default. All controls and failures will be reported.

## Native construction

Use exactly the existing legal source training observations and original teacher
channels; no benchmark query, target label, additional teacher or v10 member head.
Gaussian geometry, grouping codes, shared member parameters, observed support and
anchor eligibility are protected. Semantic and local channels each learn a full
dimensional affine residual, initialized to identity. This is not a compressed
or multimodal capacity upper bound: an affine map cannot unmix arbitrary local
appearance modes. It is the first functional construction control.

Each eligible training SAM region generates a semantic descriptor query and a
local teacher-pixel query. Sample 64 pixel pairs from its known positive interior
and disjoint comparable-scale negative interiors; unknown/nested support remains
unknown. Require at least .5 observed projection mass and normalize the observed
operator within each pixel. The semantic and DINO teachers provide their actual
cosine-score differences on those pixels; a positive SAM pixel is not forced to
be more similar under DINO. Teacher soft preference is sigmoid(delta / .1).

Student native directions are normalized on each Gaussian, scored against the
same query, then projected. Minimize paired Bernoulli KL plus mean directional
drift (1 - cosine) from original Gaussian codes, coefficient 1. Equal .5 channel
weights; AdamW .0001, weight decay .01, gradient norm clip 1, seed 20261009, two
complete eligible-region epochs. No parameter sweep or early stopping. Source
development views are opened only after the final update to report parent versus
refined preference KL, with fixed seed 20261010 and 64 pairs per region. They do
not choose checkpoints or methods.

Bake normalized refined coordinates into the existing full-native buffers in
their original dtype, only on observed rows. Construction maps are retained for
audit but never referenced by deployed exports. Runtime remains the one Gaussian
field plus current input/shared encoders, with no external scene dependencies.
Every field in this cohort has local_metric=none, so native-only updates must
preserve registered point membership exactly; this is checked on real outputs.

## Attribution and reporting

Freeze code, configs, original parents, previous cross-view fields and protocol.
All 90 field exports must complete before predictions; all three complete
prediction sets must complete before new benchmark scoring. Preserve attempts.
Record protected tensors, source exposure, held-out source metrics, export sizes,
and construction cost. Validate independent copied-export query replay.

Compare rollback to previous common method, protected training to rollback, and
functional native to protected training. Report per-scene and aggregate response
IoU/AP, actual-set IoU/precision/recall, ScanNet 19/15/10 and four UQIS modalities;
paired scene bootstrap with the existing seed/cluster procedure. LERF actual
subset projection is not independent 3D ground truth. v10 remains a frozen
capability reference with different source budgets/extra-labelled member head
and NVOS source-assisted inputs, not a same-supervision causal comparison.
Do not claim full recovery or top-conference novelty from implementation alone.
