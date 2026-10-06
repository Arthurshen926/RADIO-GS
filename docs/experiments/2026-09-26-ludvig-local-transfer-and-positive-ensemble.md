# Pinned LUDVIG audit and same-input positive-reference ensemble

The transferable lesson is to preserve the query's object evidence through the
map, rather than to add another generic similarity smoother. The local LUDVIG
91.2577% NVOS result and our frozen reference-only 81.1608% result have different
online image rights and different geometry. Their difference cannot be assigned
to graph propagation alone.

## Exact source provenance

The repository retains the reproduction launcher, protocol records and patch
under `reproductions/ludvig/`, but its original `/root/baselines/LUDVIG` checkout
is absent. We restored the exact locked official commit
`4461fc515439bb498a75d71738a1e73cf7a452ed` into
`output/ludvig_source_audit_next32/upstream`; its tracked-file hashes and the
temporary checkout location are recorded in the adjacent `source_manifest.json`.
This is restored pinned upstream, not a claim that the historical patched
checkout remained available. The local patch and launcher were inspected
separately. No LUDVIG solver or target-image inference was run in this audit.

## What the release actually does

| Stage | Pinned LUDVIG implementation | Current frozen field/readout | Implication |
|---|---|---|---|
| Feature uplift | Accumulate actual Gaussian renderer contribution times dense 2D features, divide by total renderer contribution | Native regional evidence is fused into a protected64-coordinate prefix, with32 learned structural coordinates | LUDVIG does not remove mixing by magic; inputs and query-dependent mask construction matter |
| NVOS-SAM query | Lift the positive scribble, project it into **every** camera, sample10 triplets, run SAM per camera, then uplift scalar query masks | One reference SAM mask becomes a masked SigLIP crop; field membership is repaired on reference positives and spatially filtered | Generic appearance is asked to replace multiview query-specific object masks |
| Graph topology | Euclidean XYZ neighbors; NVOS DINO configuration uses200 | Current retained NVOS spatial graph uses16 with a distance cap | Released graph is not a feature-space kNN graph or a superpoint system |
| Graph semantics | Dense native DINO features, scene PCA40, normalized feature distance and a query-dependent regularizer | The retained NVOS core-component rule uses spatial connectivity; structural similarity is computed but does not determine its edges | “Multiteacher field” does not mean native DINO graph behavior is deployed |
| Positive evidence | Exact renderer adjoint; positive-only similarity averages per-anchor cosine using evidence weights | Exact observed-alpha-normalized reference adjoint already implemented | Replacing this with nearest-XYZ assignment would be a regression |
| Selection/calibration | DINO graph bandwidth search uses reference SAM; SAM arm uses its own normalization and threshold0.25 | Frozen field probabilities, continuous resize, threshold0.5 | Thresholds and source fitting cannot be silently exchanged |

The primary code is
[uplifting](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/utils/solver.py),
[query-specific SAM masks](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/predictors/sam.py),
[graph topology](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/utils/graph.py),
[DINO extraction/PCA](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/predictors/dino.py),
and [query-conditioned graph](https://github.com/naver/ludvig/blob/4461fc515439bb498a75d71738a1e73cf7a452ed/diffusion/segmentation.py).

The ScanNet adapter reads precomputed segmentation indices and512D feature rows,
normalizes each row and expands it into a dense feature image. Those 2D region
indices are not an implementation of 3D superpoints. Dense image features then
enter the same renderer-weighted uplift. Its LERF CLIP adapter instead uses
multiscale sliding-window crops and normalizes the accumulated dense features.
There is no one universal “LUDVIG region representation” across these adapters.

Its text graph also goes beyond smoothing text scores: `diffusion/clip.py`
extracts high-confidence CLIP seeds, forms a native-DINO prototype or logistic
unary, and uses the supplied query list's per-node argmax to assign feature
distance scales. Consequently the released text graph can depend on what
other queries are supplied alongside a query. Copying that behavior without
an explicit interface change would conflict with our independent-query
isolation contract. That dependency is separate from graph quality.

## What can and cannot transfer

1. **Exact contribution-weighted uplift** is compatible and largely already
   present. Inverse-render evidence must use the same compositor/normalizer as
   the output, with visibility retained separately. Next31 corrected an audit
   that used a different normalized quantity; it did not discover that every
   current runtime lacked renderer weights.
2. **The released graph algorithm** is already ported in
   `radio_gs/querying/query_conditioned_diffusion.py`, including its unusual
   neighbor-slot normalization. Earlier signed NVOS controls used201 entries
   including self and the released kernel. Increasing the current graph's
   neighbor count alone is not a new method. Dense-native DINO versus learned
   structure also has prior mixed controls; it is not justified to replace the
   latter globally just because it is smaller.
3. **Query-conditioned masks** are the most important representational
   difference for NVOS. Query-specific multiview SAM uplifting cannot be called
   a same-input repair of our frozen reference-only runtime: it opens additional
   RGBs, including the target. It is a distinct, potentially legitimate future
   all-view track with matched competitors and separately declared compute.
4. **Reference-only repeated prompt segmentation** is transferable with current
   rights. It tests whether a single prompt grouping committed to a wrong
   object/part scope before any map lookup. It adds query compute, not images,
   latent coordinates, target labels or stored task-specific fields.
5. **Native capability access needs an explicit contract.**
   `train_three_query_field.py` exports `field.state_dict()` but not its separate
   DINO/RADIO reconstruction heads. The field owns a native SigLIP decoder and
   structural/affinity heads, not a deployable native DINO decoder. This is a
   design limitation, not proof of an accidental serialization failure: the
   present interface never promised those separate teacher heads at query time.
   DINO supervision can still affect the learned suffix; missing its decoder
   does not mean that every learned DINO relation was erased.
   If native DINO matching is desired, the head/metric must be exported or refit
   with source-only validation; arbitrary suffix cosine is not that capability.
6. **Positive-only is a real constraint.** LUDVIG's logistic option can treat
   reference-visible unmarked rows as negative. Copying that option would
   violate our retained policy that outside a positive SAM mask is unknown.
   Its positive-only prototype regularizer is a different operation, but
   prior learned/native graph controls prevent claiming a guaranteed benefit.

## Registered bounded control

Authority: `output/nvos_positive_ensemble_next32/registration.json`.
One fixed rule is evaluated on all eight tasks; no per-scene policy selection.

- Same SAM3 checkpoint, reference RGB and official positive scribble.
- Ten deterministic triplets using the existing salted pixel-index sampler;
  `multimask_output=False` for each call.
- Admit a trial only if its integer coverage of the **entire** raw positive
  scribble is at least the frozen parent's coverage. This avoids accepting a
  plausible subpart that ignores other authorized positive strokes.
- Take strict majority (`>0.5`) of admitted binary trials, retaining all raw
  positive pixels. If none are admitted, fall back to the exact parent mask.
- Recompute the masked image crop and native image encoding, then execute the
  entire unchanged query pipeline: field retrieval,80-step positive/SAM repair,
  eroded-core spatial selection and80-step positive-only repair.
- Preserve all eight paired element and pixel predictions **before any new
  target GT scoring**. Parent binary replay must be exact on both prediction
  and scoring grids. The threshold and interpolation remain frozen.

Agreement with a positive scribble does not establish object identity elsewhere.
The ensemble complement remains unknown; it supplies no new negative labels.
The method changes the upstream query scope, not the previous rejected
high-purity-component or one-node-bridge rule.

The cost is ten3-point decoder calls instead of one8-point multimask call, with
one image embedding in either case. It is not equal-query-compute evidence.
An older signed/canonical-hash branch already used10x3 SAM3 completion. This
experiment is a positive-only control on the current64+32 field, not a claim
that repeated SAM prompts are novel.

Six CPU tests verify full-scribble admission, exact parent fallback, identical
trial reduction, strict-majority ties/raw-positive retention, permutation
invariance, input immutability and rejection of malformed inputs.

## Outcome

All eight paired evaluations finished. The registered joint gate fails: macro
IoU improves by only0.1232points while AP falls0.2561points. **Do not promote**
the ensemble and do not select it only for the tasks it improves.

| Task | Parent IoU | Ensemble IoU | Change |
|---|---:|---:|---:|
| flower |82.7127%|84.7734%|+2.0607pp|
| fortress |93.7727%|93.8107%|+0.0380pp|
| fern |74.9983%|75.0497%|+0.0514pp|
| horns_center |93.4228%|91.5048%|−1.9180pp|
| horns_left |89.0657%|89.8264%|+0.7607pp|
| leaves |69.3769%|69.3697%|−0.0072pp|
| orchids |66.4520%|66.4520%|0|
| trex |79.4852%|79.4852%|0|
| **Macro** |**81.1608%**|**81.2840%**|**+0.1232pp**|

Macro AP:91.5057→91.2496%; pixel accuracy:96.4790→96.5087%.
Raw-positive agreement is unchanged for every task. All baseline binary masks
match at both the prediction and native scoring grids. The separately executed
scorer verified all16 paired pixel files and16 element files before opening any
target GT. Target RGB, negative scribbles and training feature caches were not
used. The1773 frozen source hashes remain unchanged.

The source-only diagnostic is especially informative. Unfiltered majority
voting over all10 orchids trials covers only25.92% of the known positive
scribble, compared with100% for the parent. Individual trial coverage ranges
from20.80% to99.89%. None qualifies under the registered full-positive rule, so
the parent is retained exactly. Trex likewise has no admitted trial. This is
direct evidence of incompatible whole/part predictions under small point
subsets, without requiring any target labels. Merely averaging repeated SAM
masks can erase authorized evidence.

Among admitted tasks, reference agreement still does not predict target scope:
flower improves2.06points while horns_center loses1.92points. The latter changes
3472 selected elements despite only3334 changed reference-mask pixels. These
counts have different units and should not be read as an amplification ratio.
The intervention changes both the crop query and reference constraints, so this
experiment does not assign separate causal fractions to those two paths.

The actual-output audit also keeps the two outputs separate: leaves changes no
selected elements but changes164 thresholded target pixels through continuous
score rendering. Orchids and trex retain the parent masks and selected sets;
their fresh cold computations are not claimed to be byte-identical float
serialization.

The remaining high-impact problem is **query identity and complete scope after
cross-view transfer**. A positive reference mask can identify some surface, but
neither its complement nor an unobserved component establishes background or
same-object identity. LUDVIG's best SAM arm obtains more query-specific evidence
by reading additional RGB views. Under our frozen input rights, that missing
evidence has to be learned/preserved in the map with independently checked
scope correspondence; a different prompt sampler or another spatial cutoff
does not supply it.

Independent code review found no blocking method or input-right issue. Two
provenance limitations are explicit: the builder/predictor validate their data
hashes but the registration's manifest hash is directly rechecked at scoring;
field/geometry identities are validated at runtime and enumerated in the final
seal rather than fully enumerated in the initial registration.

Results: `output/nvos_positive_ensemble_next32/summary.json`.
Source diagnostic: `output/nvos_positive_ensemble_next32/source_scope_diagnostic.json`.
Seal: `paper/artifacts/nvos_positive_ensemble_next32.json`.
No default, sealed source, geometry, field or historical result was changed.
