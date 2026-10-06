# Typed source evidence and transient set inference

Status: experimental, user-authorized method work, 2026-09-28. Not a default promotion.

This extends ADR 0001's typed readouts and explicitly relaxes ADR 0007's
"Local field evidence owns identity" **only for the online-source-evidence
candidate**. It does not replace the fast field-only query contract or rewrite
its geometry, latent coordinates, teacher caches, or calibrated prior.

The method consists of a shared frozen map, typed identity-evidence providers,
and a shared element-set solver. The map already includes dictionaries,
calibration and scope state; neither the retained map nor this extended method
should be described as only a 96D vector and one cosine operation.

Two implemented query types:

- `text_union`: the verbatim runtime text is grounded in authorized source RGB.
  Detected masks remain per-view, per-instance observations. Only the declared
  text-set task permits taking their within-view union. A common phrase does
  not establish a common physical instance across views. Native foreground and
  same-window context crop features supply **identity reliability**, never
  per-element membership labels. Soft identity confidence combines detector
  confidence with a sigmoid foreground/context margin, scaled by that query's
  source-proposal RMS margin. It is not a calibrated correctness probability.
- `reference_instance`: the existing authorized positive-only reference mask
  initializes video propagation. It identifies the requested instance. No new
  negative prompts or target RGB are used. Missing tracked objects are unknown.

Membership is inferred with the frozen, observed-support-normalized renderer.
Source masks provide eroded positive/negative regions with an unknown boundary
band. Independent-view consistency continuously weights the source evidence;
there is no permanent veto based on one positive prediction. The parent is a
prior, not the target used to train the identity provider. Unobserved elements
keep their prior. This is transient inference, not per-scene latent training.

The public numerical core is `radio_gs/querying/typed_evidence.py`. It consumes
view operators, membership targets, evidence validity and confidence. Teacher
names, text strings, benchmark labels and scene routing are absent from the
solver. Same-view/other-view teacher agreement does not certify physical truth.

The hard foreground/context rejection and strongest-contribution write gate
are retained as experiments, not imposed on the soft-identity candidate. In
particular, a context-bearing region is not automatically a false object, and
weak rendering contributions can still carry useful completion information.
When write gating is tested, forward predictions still use the full renderer;
the gated adjoint is a visibility-restricted update, not a claim to optimize
an exact ordinary least-squares objective.

Additional online budget must be reported: source RGB access, SAM3 grounding or
video inference, and native crop encoding. Query text is an inference input,
not a map-training vocabulary. Query artifacts are transient caches. Target RGB
and target labels cannot generate or select predictions. The larger NVOS source
variant visits all manifest-authorized training frames, excluding evaluation
frames, and uses frozen cameras/geometry for additional projection operators.
The bridge control uses these frames only for tracking and retains the old
fusion views. Neither is an equal-input-budget field-only comparison.

Not implemented by this decision: an explicit whole/part relation parser,
independently verified cross-view instance components, independent surfel
geometry/rendering, robust multi-prototype dense language reconstruction, or
complete four-query UQIS deployment. Text-union handling does not claim to solve
these tasks. Image category queries and 3D prompts retain their old paths.

See the [next44 experiment](../experiments/2026-09-28-typed-identity-and-set-evidence-next44.md)
for full-cohort evidence and regressions. The historical selected-only LERF
metric remains a 2D projection score, and ScanNet remains the frozen
volume-weighted Gaussian-label protocol; no public SOTA equivalence is implied.


2026-10-03 category-partition decision (system_v2): complete ScanNet10 uses the
shared native map and source evidence, but **one-pass visibility-weighted
category fusion** replaces iterative set fitting for this task. Same-evidence
full10 point19/15/10 improves56.9082/58.0746/69.5986 to57.9215/58.9728/70.9060;
public replay matches the independent implementation exactly. All1158 declared
source frames are used, with the full-vocabulary native cold prior computed
before query batching. No old scene calibration parameters are required.

This refines the earlier assumption of one numerical solver for every task.
Category partitions, whole/part sets and reference instances share geometry and
explicit evidence contracts but need not share an inversion objective. LERF/NVOS
retain their validated set/reference readers. A weighted average is a stronger
baseline, not a novel mechanism; the comparison does not individually isolate
iteration from the removed soft consistency term. See
[complete controls](../experiments/2026-10-03-complete-source-cold-system.md).


2026-10-03 native-observation refinement (system_v5): current ScanNet uses a
query-independent native region dictionary and sparse element transfer, with
complete-menu categorical information weights. Source view weights use detector
relative confidence while preserving the original per-query native-confidence
known-pixel mass. This does not remove native identity from the total budget.
No per-scene routing or new teacher. v3/v4 remain controls; all22 v5 jobs complete.
Current ScanNet reporting is frozen equal-point scene-macro, not the historical
volume-weighted pseudo-label protocol mentioned above. Whole-view-excluded
semantic corroboration did not improve the complete result and remains inactive.
GT representability probes are diagnosis only and never supply runtime weights.
[Full evidence](../experiments/2026-10-03-observation-identity-and-reliability.md).

2026-10-03 supervised category-reliability experiment: optional linear /32-hidden
heads consume scope statistics plus frozen full-menu region/query compatibility.
Fitting differentiates the deployed visibility-weighted vote through all source
views to equal-point element classification. Geometry, native query coordinates
and teachers remain frozen. Quality-only outputs estimate mask precision/recall;
after downstream training they are evidence weights, not calibrated quality.
Nine official training rooms provide labels; this is a separate supervised track,
not an additional zero-label SOTA result. Different reserved rooms0575/0591/0599
completed30 public-entry evaluations. Full candidate61.839701→62.801336%19 on the
same map, two rooms up/one down; matched-mass shared weights62.843615. Neither
linear nor nonlinear signed weights establish a primary-metric advantage over
shared weights. No default promotion or extension to whole/part/reference tasks.
See [complete methods, controls and limits](../experiments/2026-10-03-consumed-element-reliability.md).


2026-10-04 dual-native probability prior (system_v6): retain separate frozen
SigLIP and pretrained Mask-Adapter category dictionaries/encoders; average their
probabilities within the existing visibility vote, preserving the original
SigLIP fallback where source evidence is absent. No per-scene routing, no new
label fitting and no source-budget or geometry change. Full original10 improves
59.592965→61.011491%19 (9 up/1 down); opened development6 improves59.314710→60.812414.
Reserved0606/0629/0653, all515 source views and36587 regions, improves55.260616→57.608254
(2 up/1 down0.463783pp). All18 reserved predictions preceded label scoring.
Identical22 executable jobs completed under the dual-native candidate recipe;
v6 names that same validated job list. LERF/NVOS retain their respective readers.
Native files add1.104306GiB on original10; source/model/RGB/geometry cost is extra.

Supervised577/705-parameter local and705-parameter pooled-global controls were
trained only on9 official-training rooms. Local dual gets58.180987%19 on new3,
but only0.572733pp above the untrained average, loses to it on15/10 and opened
dev6, and still exchanges missing targets for fewer false positives. Do not
promote these learned heads or infer calibrated identity/extent from their logits.
A real auxiliary entropy-policy mismatch was corrected before reserved prediction;
affected36 development runs were repeated. Retain individual region identity and
correspondence before mixing as the next research question, rather than expanding
aggregate confidence heads. Ensemble improvement is a system gain, not proof of
novelty, compactness, annotated four-query accuracy or public SOTA equivalence.
[Full comparison](../experiments/2026-10-04-local-element-and-dual-native-fusion.md).


2026-10-04 individual region identity/correspondence experiment: keep single
source observations before class unions and one-hop reciprocal geometric links
as explicit research assets, not an active default. Six configurations x13
scenes completed78 public-entry evaluations, including453 images from three
additional rooms. Single-vs-union gains0.52/0.80pp19 on old/new cohorts, but full
linked candidate60.188221/58.934898 trails v6 61.011491/59.387189. No promotion.
The emitting-label product locks almost every node's category: new-room links
change215 native top1 predictions but0 final verified labels. GT instance
agreement holds for1174 high-purity endpoint pairs, not necessarily all1897 links.
Whole-scope category compatibility is not dense element membership. Stop tuning
this multiplicative gate/graph depth; future identity alternatives and local
membership must be independently justified with matched end-output controls.
These new validation rooms are now opened. No new LERF/NVOS/compact/four-query or
SOTA claim. [Complete study](../experiments/2026-10-04-region-observation-identity-transport.md).
