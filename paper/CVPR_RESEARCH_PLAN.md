# CVPR research plan — current system, 2026-10-04

Current executable recipe: `configs/pipeline/system_v6.json`.
All22 identical executable jobs completed: `output/system_dual_native_candidate/summary.json`.
See [implementation](../docs/SYSTEM_IMPLEMENTATION.md) and
[complete source/inference controls](../docs/experiments/2026-10-03-complete-source-cold-system.md).
The June compact-RADIO manuscripts are historical and do not describe this system.

Latest completed [individual region / geometry study](../docs/experiments/2026-10-04-region-observation-identity-transport.md):
78 full scene evaluations; original10 and three other official-validation rooms
0670/0678/0685, all453 RGB and34828 native regions. No fitting/parameter revision.
Single regions outperform matched class unions by0.52/0.80pp19, but candidates
60.14/58.68 and60.19/58.93 lose to v6 61.01/59.39. No promotion or graph-depth sweep.
The name-product gate leaves almost all node identities fixed; new-room native
classification changes215 times under links but final verified top1 changes0.
1174 high-purity endpoint pairs all share a GT instance, conditionally; mixed
scopes and locked semantic identities remain. This supports rejecting this
mechanism, not claiming correct general correspondence or a CVPR contribution.
Next distinguish region identity alternatives/unknown from local membership;
region classification must not masquerade as dense semantic supervision. Original
compact/four-query/public-SOTA claims remain open; LERF/NVOS unchanged this round.

Previous completed [local element / dual-native study](../docs/experiments/2026-10-04-local-element-and-dual-native-fusion.md):
Promote only the label-free equal-native-probability prior to v6. Original10
19/15/10=61.011491/62.527589/72.486061;9 up/1 down. Dev6 19=60.812414 versus
59.314710. Another3 reserved rooms,515 views/36587 regions, give55.260616→57.608254,
2 up/1 down0.463783pp. All22 integrated benchmark jobs completed; LERF/NVOS retain
accuracy, with no claimed new gain. Extra native storage1.104306GiB for original10.
This is a useful system ensemble, not a proven new CVPR contribution.
Local shared heads trained on9 official-training rooms give58.180987%19 on new3,
+2.920371pp over original and+4.432704pp over equal-capacity global control, but
only+0.572733pp over the untrained average. They lose to the average on dev6 and
new3 15/10; no learned-head promotion, no parameter sweep. Whole-view versus local
is no longer the only issue: aggregate score statistics still lack independent
region identity/correspondence and extent information. Preserve these before
mixing in the next bounded mechanism. Current validation rooms are now opened.
Auxiliary native weighting mismatch was fixed and affected dev runs repeated;
new-room predictions used the correction from the start. No SOTA/compact/four-query
claim closure. UQIS stays paused. Results do not validate unseen category transfer.

Previous completed [consumed-element reliability study](../docs/experiments/2026-10-03-consumed-element-reliability.md):
Native region/query compatibility and exact full-view differentiation are now
implemented. Linear and546-parameter capacity controls fit9 official-training
rooms. Another3 reserved rooms0575/0591/0599,388 views/24817 regions, completed
30 public-entry evaluations plus3 separately labelled diagnostics. Same-map19
61.839701→62.801336 with MLP retrieval, but original SigLIP vote62.840361 and
same-MLP matched-mass shared62.843615. Two rooms improve, one regresses; no
promotion or signed-head contribution claim. Map mean gains4.72pp but0575 loses
8.59pp;0599 window loses10.91pp as FN rises2333→3940 despite FP806→516.
True-quality64.806771 is a label-using diagnosis, not a result/upper bound.
Stop extending global view/category scalar heads. Next test local identity and
extent at native-region/visible-element granularity on training rooms before a
new fixed transfer experiment. New supervision regime, no zero-label SOTA claim.
Current validation is now opened. LERF/NVOS and compact/four-query claims unchanged.
Learned fusion itself is established prior work (GLFS, DeepViewAgg); novelty and
substantial independent gains remain unproven. Keep default system_v5.

Earlier [quality-only independent-room study](../docs/experiments/2026-10-03-supervised-scope-quality-independent-rooms.md)
completed15 evaluations on0050/0231/0412, all554 sources/36940 regions. Map19
42.72→50.88 but full55.60→55.76, typed nearly identical to matched-mass shared.
These scenes are development data now. The above study implements the proposed
native identity/consumed-objective follow-up; it has not closed the remaining gap.

Earlier completed [source likelihood/identity/resolution study](../docs/experiments/2026-10-03-source-likelihood-and-resolution.md):
70 full scene evaluations. Same improved map, cross-view-excluded detector error
models only60.35%19 versus60.27 vote. Fewer wrong overrides are offset by fewer
useful corrections; source identity/scope agreement does not improve the result.
Paired same-forward rasters improve60.273548→60.653474%19,9/10 scenes, with
identity/projection storage6.958→20.563GiB. Against v5 the complete candidate gains
1.060509pp19 with two scene regressions, not the proposed substantial-gain/new-scene
gate. Keep v5 default and candidate explicit. No SOTA, novelty, or compactness claim.

Stop reweighting the same pseudo-evidence as the next mechanism. Independently
supervised identity correctness and extent completeness must be separated, with
training and untuned validation scenes disjoint from this repeatedly used cohort.
The earlier claim of no readable additional scenes was a search error; official
validation scenes exist in ScanNet-PFIR-Small/AGILE3D. The separate supervised
scope-quality branch uses only nine official training rooms and validates on
three different official-validation room prefixes. Its nine fit scenes are not
independent evaluations, and the supervised branch is not the original zero-label
track. Do not use this cohort's point GT to train an override gate and call its
own scores generalization. A higher
source raster is a useful system control, not a resolution of the semantic bottleneck.

Earlier complete [scope/identity study](../docs/experiments/2026-10-03-scope-granularity-and-source-identity.md):
70 scene experiments,1158 source images,5392 new native union encodings. Best
candidate60.064263%19 (+0.471298pp), not promoted. Under the same categorical
decoder, emitting-query labels58.386028 beat native union probabilities51.666103
and native Top-1 labels49.696308. Correct native identity cannot be assumed on
actual scope crops. Prioritize task-granularity/context/scope purity over further
weight sweeps. No independent new-scene or SOTA claim follows from this study.

Completed follow-up: [positive identity grounding](../docs/experiments/2026-10-03-positive-region-grounding.md),
two rules × ten scenes. Region grounding final59.903771 versus v559.592965 and
simple pixel grounding59.003233. The intermediate50.15→58.64 gain introduces
source information already consumed downstream; it is not a new offline-field
claim. No default promotion or mixing-coefficient search. This result does not
pass the substantial-gain/new-scene gate. Identity/scope correction must reduce
wrong overrides as well as improve intermediate predictions.

## Current mechanism and bottleneck evidence

Same v3 map/source assets: information-weighted native regions and detector-relative
view confidence with matched per-query budget give59.592965/60.869057/71.430577%,
+1.143720/+1.216239/+0.403534pp over v3.9/10 scenes improve19class. LERF/NVOS
retain70.400676/66.960933 semantic/selected and86.513417 reference IoU. All22 fresh
public-entry jobs completed. These are development results, not public SOTA.

Whole-view-excluded support failed. Two fixed controls and the complete combined
system are recorded in [the report](../docs/experiments/2026-10-03-observation-identity-and-reliability.md).

LABEL-USING diagnosis: unchanged geometry/W with GT regional assignments reaches
80.292984%19class; fitting regional logits reaches81.590436%. Do not put these
numbers in the automatic method table or present them as generalization/upper
bounds. They locate a semantic assignment/selection bottleneck, not defective
carrier capacity alone. Next prioritize grounded region identity supervision,
protect native query geometry and unknown evidence, and separate raw teacher
recognition from matching/mixed-scope selection. More generic consistency is not
supported by the controls. An untuned-scene gain is still required before novelty
claims, followed by full multi-query and independent3D evaluation.

The following consolidation and v2 findings are frozen attribution history.

## Latest system consolidation

ScanNet now uses a precompiled native region dictionary/sparse element transfer.
It preserves response-before-pooling gains, with1.210025GiB semantic files versus
4.493211GiB in v2. Complete22-job v3: LERF70.400676/66.963237 semantic/selected,
NVOS86.513417, ScanNet58.449245/59.652818/71.027043%. LERF/NVOS are retained.
Compilation adds no accuracy; it preserves the earlier0.53pp category gain while
removing its deployment-cost objection. Five scenes still regress versus v2.
This is operator factorization/storage, not independently established novelty.
Natural-background input failed the complete balanced comparison; masked stays.
[Full report](../docs/experiments/2026-10-03-compiled-observation-system.md).

The following v2 results are frozen attribution controls, not the active recipe.

## Established results and their attribution

Frozen geometry, native semantic coordinates and typed source-assisted readouts
are integrated. Complete ScanNet10 now uses1158 declared source frames, with no
historical scene calibration parameters. LERF4 retains separate semantic and
selected-set outputs; NVOS8 retains positive-reference tracking.

Fresh v2: LERF semantic70.400676%, selected projection66.961588%,
NVOS86.513417%, ScanNet10 point19/15/10 mIoU57.921506/58.972766/70.905986%.
LERF/NVOS did not substantively improve this stage. LERF selected projection is
not independent3D GT. NVOS's geometry/tracking adapter does not establish the
value of the semantic feature field. All numbers use local frozen protocols;
public SOTA/input equivalence is not established.

Matched8, same newly computed teacher observations: increasing source evidence
from176 to886 frames improves55.7015/56.5157/68.4335 to57.2945/58.1129/70.2912.
This is useful input coverage, not a new mechanism contribution.

Full10, identical evidence and prior: one-pass visibility-weighted fusion reaches
57.9215/58.9728/70.9060 versus iterative inverse fitting plus soft consistency at
56.9082/58.0746/69.5986.19class improves in all10 scenes; scene0140's15class loses
0.524pp. The simpler method replaces the inferior category default, and public
replay exactly matches the independent baseline's metrics.

**Do not claim iterative robust scope inversion as the source of category
performance. The direct weighted average is now the baseline to beat.** This
comparison removes iteration and consistency together, so it does not establish
which subcomponent caused the loss. A weighted average is not a novel contribution.

## Native-field contribution now measured, with important limits

A complete10-scene uniform-prior ablation with identical geometry, support,
source identity teacher and0.1 regularization lowers19/15/10 mIoU by
2.090728/2.784880/2.182746pp. This supports the native prior's contribution in the
source-assisted category adapter, not compact multi-teacher learning or reference
query benefit.19class still worsens in3/10 scenes when native information is added.

Spatial proposal-owner confidence and partial categorical observations both fail
full-cohort promotion. Evaluation-only correction analysis finds52,853 corrected
but36,455 spoiled points, including spatial/class confusion and generic-category
substitution. Compare original native region evidence with the written field
before blaming geometry or only scope completion. Do not train/select a gate on
these evaluation labels. [Full results](../docs/experiments/2026-10-03-native-prior-and-evidence-conflicts.md).

## Pooling localization completed

Same observations and geometry: early region category responses improve native
prior19class by2.147149pp but full source fusion only0.527739pp, with5/10 scenes
regressing. Public-entry reproduction is exact. Additional assets11.029344GiB
and per-query lifting prevent treating this as compact-field success. The uncompiled version remains
a comparator; the equivalent compiled representation is now active for its cost
advantage. Do not treat compilation as a separate accuracy contribution. [Full experiment](../docs/experiments/2026-10-03-native-observation-pooling.md).
Teacher recognition/context loss and source scope/identity failures remain
unseparated. Any new crop input is an information ablation, not same-evidence
algorithmic gain. No additional temperature/weight sweep is warranted.

## Remaining method question

Can native identity and uncertain, possibly partial multiview extent evidence
produce a more reliable element set than this stronger simple baseline, without
losing whole/part/instance distinctions? Source agreement is not truth. Absence
outside a partial detection is not proven exclusion. The region-mean writer
can still mix identity and scope despite retaining native encoder coordinates.

First separate source recognition, extent coverage and lifting losses in the
remaining weak classes. The class diagnostic in
`output/scannet_source_vote/class_diagnosis.json` identifies failures but does
not identify their cause. Do not immediately add a new teacher, token bank or
more fitting steps, or reuse a failed finest-mask/dual-scope recipe.

Use at most two core candidates and two parameter revisions. Compare with the
new one-pass fusion and retained set/reference readouts using identical images,
teacher outputs, geometry and budget. A method must keep its benefit on scenes
not used to design it. Do not select winners per query or per scene.

## Claims requiring additional evidence

1. Same-map annotated text/image/2D/3D retrieval, including the intended signed
   prompts. Current execution probes are not four-query benchmark accuracy.
2. A feature-field contribution to reference tasks, not only successful source
   tracking and geometric propagation.
3. Independent3D extraction truth and aligned public LERF/NVOS/ScanNet protocols,
   with input tracks and source-image/model/cache costs stated.
4. A useful globally fixed mechanism beyond a simple baseline, validated on
   untuned scenes before compact multi-teacher distillation.
5. Compact deployment that preserves all four query abilities, not just native
   coordinate int8 with near-equal text metrics. UQIS remains paused by user.

Earlier rejected ownership, scope transfer, source calibration and sparse RGB
conflict rules remain documented in the experiment reports. No confidence/area/
verifier sweep based on isolated hard queries is scheduled. Current progress
supports a stronger complete research system, not yet a CVPR-ready novelty or
SOTA claim. Historical plans: [research history](CVPR_RESEARCH_HISTORY_2026-10-03.md).


## Regional identity result and narrowed next mechanism (2026-10-03)

[120 complete evaluations](../docs/experiments/2026-10-03-task-trained-region-readout.md)
confirm that a pretrained task-aware external region reader improves map-only19
from50.15 to57.74 under unchanged W/geometry. Plain pooling on the same backbone
reaches49.60. Full source fusion reaches60.27 versus v5 59.59, with two regressions.
This is meaningful upstream capability retention, not a new RADIO-GS contribution
or a7.59pp final/SOTA gain. Keep v5 as sole default; do not build a best-of-scene system.

Whole-scope negative removal fails even with the improved map (59.60 final).
The next mechanism must decide when independently grounded identity and scope
justify replacing the map's response; neither blanket negative admission nor
blanket abstention is sufficient. Use this fixed map to isolate that decision,
not more teacher/crop/weight sweeps. Keep external-head attribution and costs explicit.
