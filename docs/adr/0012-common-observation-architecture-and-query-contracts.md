# Common observation architecture with explicit query contracts

Status: accepted, current v10 development mainline, 2026-10-06. Extends ADR0001/0010;
does not assert that all query types share one trained vector or inverse solver.

The maintained architecture is:

1. Freeze geometry under the declared input track. Compile legal observations
   into native teacher descriptors, element association and visibility.
2. Encode each query in its teacher's native space or obtain a geometric point
   anchor. Keep identity evidence distinct from element membership evidence.
3. Decode the requested set type. A complete category partition competes across
   classes; a category set can contain multiple instances; a reference instance
   requires physical correspondence. A partial region is not a complete object.
4. Evaluate the consumed output: semantic rendering, actual subset projection,
   category labels, or scores on the official mesh. These are not interchangeable.

`querying/observation_readout.py` is shared by registered mesh retrieval, dense
native field queries and compiled category observations. Typed render solvers
remain shared by source-assisted text sets and NVOS reference instances. Runtime
routing depends on explicit scene capabilities and input access, not target names
or evaluator labels. `pipeline.cohort` now maintains all five evaluation tasks in
one recipe and finishes all predictions before starting their scorers.

UQIS remains strictly map-only: image queries carry their reference crop; point
queries carry only their coordinate/camera inputs. The method cannot access
cross-modality target pairings. Legal reference RGB/SAM in NVOS or declared source
evidence in another track does not authorize it for UQIS. Shared architecture
does not mean identical input budgets or silently upgrading an access contract.

The bounded UQIS candidate separates native region identity from physical scope:
choose the highest-native-score supported region, then match one region per
other view using co-visible IoU >= .5 and bilateral visible fractions >= .5.
The seed-only control isolates localization; cross-view scope isolates range
recovery. Both use bounded positive support s+s(1-s)p; a missing partial mask does
not zero the native response s. These are ranking scores, not calibrated
probabilities. Both preserve the original native semantic coordinates. This candidate does
not execute spatial language relations and cannot correct an initially wrong
instance identity. The complete comparison and a fresh 31-job execution are now
finished. It is enabled uniformly for UQIS by `configs/pipeline/system_v7.json`;
the original native and seed-only readers remain explicit controls. The public
tasks retain their validated v6 rules. This is a development selection, not proof
of generalization to independent UQIS scenes.

No benchmark target, query phrase or GT mask is compiled into the map. Diagnostic
best-region IoU and identity purity are computed only after all predictions.
The existing 67 targets and four individual AP scores stay primary. Additional
same-target minimum/geometric-mean AP exposes a failure of any modality without
changing queries, removing difficult cases or calibrating masks on this cohort.

UQIS currently supports the four positive-query instance inputs, not all possible
query behaviors: independent part labels, category unions, signed/multiround
interaction, independent calibration/final scenes and compact student retention
remain incomplete. Same-interface execution is not proof of unified high accuracy.

The UQIS provider now also emits an independent explicit boolean element set:
the seed-region control uses its own membership; the correspondence candidate
uses a strict majority of confidence-weighted visible observations; point queries
use anchor co-membership > .5. All rules are fixed before this selection study.
The evaluator optionally scores these complete sets separately from the native
ranking and its uncalibrated fixed-cutoff diagnostic. No GT threshold, oracle
choice or per-query selector is permitted. This follows ADR0011's response/set
distinction on the mesh domain; an emitted set is a prediction, not a claim that
its unselected complement is verified negative evidence.

2026-10-05 development extension: v8 retains native local appearance before
region pooling. All original legal source views compile DINO patches and sparse
region-to-patch area weights alongside the original native field. An image query
matches its own patch features bidirectionally to each regional observation;
equal SigLIP/local-appearance identity ranks choose a seed. Original semantic
coordinates, physical scope, and point/text rules remain unchanged. These are
identity scores, not foreground probabilities or verified 3D correspondences.
The full three-arm UQIS comparison and all31 common jobs are complete; v8 is the
sole development default, with v7 retained. This changes the high-capacity map
budget to2852.81MiB over nine rooms, plus shared encoders. No source photos are
read by UQIS queries. Independent generalization, complete scope and compact
student retention remain open; local nearest-patch matching alone is not claimed
as a novel paper contribution. See the
[complete experiment](../experiments/2026-10-05-uqis-local-correspondence.md).

2026-10-05 scope follow-up: four complete UQIS conditions (36 jobs/1072 query
executions) tested third-view containment and two offline entity priors while
retaining native identity. No candidate is promoted. The optional shared scope
interface compiles teacher masks offline and accepts only map associations at
query time; it grants no source-image access. The panoptic prior improved some
coverage but caused more false membership, including changes to the dominant
selected instance despite unchanged identity seeds. Additional 3D instance/part
training, if introduced, must declare a separate supervision condition and use
disjoint training scenes. See the [scope study](../experiments/2026-10-05-uqis-entity-range.md).

2026-10-05 supervised follow-up: preserve the exact query condition through the
membership decoder. An anchor-containing SAM region can contain several objects;
replacing a point by that region loses identity information. The research element
head now retains the anchor position and local native appearance. It is trained
only for positive points and cannot silently claim text/image membership training.
Disjoint rendered-room confirmation improves, but actual UQIS selections regress
despite better AP; fixed-size readout does not repair this. The heads remain
disabled in v8. Additional3D labels, rendered observation conditions and extra map
capacity remain explicit. The [full result](../experiments/2026-10-05-independent-instance-membership-learning.md)
includes all five tasks; identical public-task predictions from the current replay
are shared as common controls before scoring, not selected from historical scores.

2026-10-05 query-conditioning correction: every supervised stage must refer to the
actual query identity. Seed-majority instance labels are not interchangeable with
the instance containing a positive point; a seed's purity cannot be forced to one.
The research region head now conditions on the exact anchor and predicts seed
quality. The element head retains view-balanced anchor co-membership, granularity
ambiguity and missing joint visibility. Room-disjoint upstream predictions avoid
using in-sample label-informed confidence as downstream training evidence.
Three full UQIS conditions and all public tasks complete49 scene jobs. Small real
set gains remain unstable and all-four success decreases; none is promoted.
Additional3D supervision and extra surface arrays remain explicit conditions.
See the [mechanism comparison](../experiments/2026-10-05-query-conditioned-membership-evidence.md).

2026-10-05 structured-set extension: the optional surface relation graph is built
offline from frozen geometry, legal source RGB and native DINO; all four registered
whole-instance inputs use the same graph and global Potts set solver with typed
unaries. Additional disjoint instance labels train only the learned relation
condition; fixed relations provide the matched control. Explicit set decisions
remain distinct from native semantic coordinates. MAP conditional scores are not
calibrated posterior marginals and cannot inherit a different decoder's AP.
The40-job study confirms point-set improvement over identical unaries, but no
consistent learned-edge advantage, no stable all-query promotion. v8 is retained.
Neutral unknown-node completion was rejected on eight development rooms: local
adjacency cannot certify target identity. Category sets and parts must not inherit
whole-instance exclusion assumptions. See the [complete study](../experiments/2026-10-05-surface-consistent-instance-selection.md).

2026-10-05 natural-language extension: an optional scalar identity head consumes
the frozen full-expression token features, regional native descriptors and public
geometry. It must declare independent NR3D/3D supervision and preserve the native
semantic output. Candidate-scope IoU supervision mixes identity and completeness;
it cannot be described as a separately verified identity classifier. Matched
local/contextual heads complete40 jobs with about2pp UQIS text development gains,
but independent rendered-room regression and scene uncertainty prevent promotion.
Other modalities remain exact; v8 stays the sole default. Real RGB-D compilation
on two fixed independent rooms improves training observations without changing
query permissions; remaining room rebuilds and retraining are not complete.
See the [grounding and observation study](../experiments/2026-10-05-referent-learning-and-observation-domain.md).

2026-10-05 real-observation completion: all29 independent NR3D rooms now use the
deployment RGB-D compiler,2643 stride20 frames without a total cap. Three fixed
objectives complete49 scene jobs and804 UQIS queries. Local scope supervision
raises text actual IoU16.387->20.212%, but scene uncertainty and identity reversals
prevent promotion. Attention does not establish a stable benefit; purity-only
supervision reduces IoU to8.229% and is rejected. All native semantic/image/point
outputs remain exact. Protecting stored coordinates is distinct from protecting
the final identity decision. Keep the real training observation repair and v8;
additional NR3D/3D supervision, native map cost and typed task permissions remain
explicit. The full31-job candidate is the negative purity control, not the local
9-scene result. See the [complete real-observation study](../experiments/2026-10-05-real-observation-grounding-and-identity-supervision.md).

2026-10-06 query-supervision follow-up: having a text-conditioned code path is not
evidence that the learned correction binds the requested instance. Query-mean and
wrong-target interventions on independent training-domain development rooms show
that the retained local candidate mainly changes common regional preferences.
These interventions remain diagnostic only; runtime queries stay independent.
The67-job matched study finds only an uncertain+.816pp training-eligibility gain;
bidirectional and same-class controls do not improve. Native/other modality
outputs remain exact. A reverse loss that changes under a per-query constant can
optimize offsets irrelevant to the consumed single-query seed ranking. An explicit
query-normalized correction removes that freedom but fails its development gate,
so it is not deployed or reported on benchmarks. v8 remains default. NR3D training
eligibility must be declared separately from frozen benchmark denominators; the
local head cannot replace missing reference-object evidence by a loss change.
See the [query-specific study](../experiments/2026-10-05-query-specific-supervision-and-transfer.md).

2026-10-06 v9 development promotion: positive geometric anchors may supervise a
direct complete-carrier membership decoder. It consumes exact per-view mask
co-membership, ambiguity, visibility, metric displacement and native appearance;
no intermediate seed is declared a pure instance. Extra3D labels come from22
independent training rooms; real observations use the deployment compiler. This
supervision condition is explicit, distinct from a zero-shot VFM comparison.
The matched rendered/real learner study finds real observations improve both
UQIS point IoUs by9.62pp; v9 improves7.65/7.52pp over v8. Text/image sets and all
native arrays are exact; four-query success remains the same7/67. Scene0030
regresses and stays in the result. All five tasks have completed validation,
including full affected UQIS regeneration after isolating a shared-array mutation.
The default common recipe is now system_v9; v8 remains a numeric comparator.

Published ScanEnts3D phrase correspondence supervision is retained as research
data, but its new language head fails the independent development gate and is
not combined with v9. Native pooled phrase retrieval, contextual token states,
and learned phrase-to-region matching are different capabilities. An arbitrary
contextual token cannot inherit the teacher's pooled-text alignment claim.
Independent final evaluation, native phrase relation binding, compact student
retention and unified high four-query accuracy remain open. See the
[complete study](../experiments/2026-10-06-scene-binding-and-real-membership.md).

2026-10-06 native phrase follow-up: learned query-only phrase roles may select
strings for independent frozen-teacher encoding; contextual-token projections
must not stand in for native pooled semantic descriptors. Candidate identities
and landmark positions remain uncapped, and multimodal spatial evidence should
not be replaced by a potentially nonexistent mean landmark. Two matched rounds
find development IoU11.110% versus prior local9.786%, but spatial intervention
gain0.297pp and uncertain scene transfer fail the first gate. Explicit same-class
instance supervision raises spatial dependence to0.919pp but lowers total IoU
to10.305%, failing the second gate. Thus no new text reader is promoted and no
benchmark results are claimed for it. v9 remains the default. The next mechanism
must preserve target identity while resolving relations, measured by both error
types and actual selected sets. See the [native phrase study](../experiments/2026-10-06-native-phrase-identity-and-relational-competition.md).

2026-10-06 typed-identity follow-up: separate semantic/relation parameters may
isolate training gradients, but cannot certify preservation of final identity
decisions. Same-class recognition also cannot replace reference-instance
supervision. Three fixed fits and49 scene jobs complete the matched comparison;
the new lexical/relational heads fail promotion and remain optional research.
Reusing the v9 complete-carrier member model with inferred regional anchors
improves native-identity UQIS text/image sets16.39/34.88->20.10/39.90%, but text
scene uncertainty crosses zero. Actual-point and all native outputs are exact.
An estimated anchor has uncertain identity; forcing its probability to one is
an assumption of the plug-in reader, not a verified positive user input. The
largest text regression lands on a different instance inside a mixed region.
One query-supported anchor alternative fails external development validation.
No per-query fallback or default change follows. The next mechanism must carry
query identity uncertainty into membership, measured separately from true-point
decoding; v9 remains sole default. See the
[complete typed study](../experiments/2026-10-06-typed-identity-and-inferred-membership.md).

2026-10-06 v10 component convergence: the existing complete-carrier member
classifier may decode image-estimated anchors while retaining the original
SigLIP/local-DINO identity. This is distinct from claiming the anchor is a
verified positive prompt. Image setIoU34.88->39.90%, scene interval positive;
text/actual-point ranks and sets and all native arrays stay exact. Four-query
success stays7/67, with one target gained and one lost. All31 common tasks were
freshly predicted before scoring. Public job definitions are unchanged; their
small numerical replay differences are recorded. The image component is chosen
from prior opened development evidence, not presented as an independent new gain.
v10 is sole default; v9 remains the unchanged numerical comparator. Text retains
its previous physical scope; one common architecture need not silently impose
the same uncertain-identity decoder on every input type.

Three matched query-ground-truth member fits are rejected after full-range exact
calibration. Keeping full-query scalar evidence or eliminating the inferred point
does not resolve reference identity, and harms correct-anchor conditional scope
competence. These extra NR3D-supervised heads are not part of v10. Preserve the
conditional member model and investigate independent reference identity before
mixing identity uncertainty into its labels again. See the
[full convergence study](../experiments/2026-10-06-query-conditioned-membership-and-system-convergence.md).
