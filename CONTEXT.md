# RADIO-GS

RADIO-GS reconstructs a query-independent RADIO representation into a compact Gaussian scene field for multiple downstream perception tasks. This glossary fixes the language used to discuss representation, information legality, and benchmark claims.

## Scene representation

**Canonical Capability Feature**:
The single persistent semantic feature owned by each Gaussian and used as the source for every downstream task.
_Avoid_: task feature, capability bank, sidecar feature

**Single Compact Feature Field**:
A per-scene semantic field in which every Gaussian persists exactly one Canonical Capability Feature; conventional Gaussian rendering state and non-semantic support state are not additional semantic features.
_Avoid_: multi-head field, task-specific field, hybrid feature bank

**Compact**:
A property measured over the authoritative Deployment Scene State required for cold-start query execution. Rebuildable Warm Cache storage and query-time storage are excluded from that cold footprint but must be disclosed separately.
_Avoid_: low-dimensional at every runtime step, undisclosed operational storage

**Deployment Scene State**:
The complete persistent per-scene state that, together with Global Method Parameters and Authorized Query Input, is sufficient for cold-start query execution. It includes the sole Canonical Capability Feature plus conventional rendering state and explicitly accepted Deployment Support State.
_Avoid_: deployment cache, scene package

**Deployment Support State**:
Bounded, schema-fixed, query-independent, and task-independent non-semantic per-scene values produced during mapping and counted in persistent scene storage. At most one validity bit and five preregistered quality scalars may be owned by each Gaussian; they must not substitute for the Canonical Capability Feature.
_Avoid_: semantic sidecar, free metadata

**Persistent Scene-Storage Increment**:
The part of Deployment Scene State added above a byte-identical conventional Gaussian rendering baseline. It includes every per-scene semantic, support, basis, fusion, hash, decoder, and readout value regardless of tensor shape, filename, or “scene-global” presentation.
_Avoid_: feature tensor size, model weights only, checkpoint headline size

**Cold-Start Storage Gate**:
A preregistered promotion boundary that accounts for the Persistent Scene-Storage Increment at deployed precision and verifies query execution after Training Artifacts and caches have been removed. Accuracy does not compensate for a failed gate; changing its budget creates a new method or claim decision rather than silently moving the boundary.
_Avoid_: post-hoc compactness check, compressed-file comparison

**Storage Soft Target**:
A preferred footprint below the Cold-Start Storage Gate that guides comparison without disqualifying an otherwise eligible method. Missing it must be disclosed and storage remains a secondary discriminator after joint task quality.
_Avoid_: hidden hard limit, optional reporting

**Storage Hard Limit**:
The maximum footprint eligible for the Compact claim. A method that exceeds it is ineligible regardless of task quality unless a new explicit claim decision replaces the limit.
_Avoid_: adjustable ceiling, accuracy-offset budget

**Global Method Parameters**:
Frozen parameters whose values and identities are identical across scenes and query executions. Parameters trained, selected, or changed per scene are Deployment Scene State even when shared by every Gaussian in that scene.
_Avoid_: scene-global parameters, shared-within-scene weights

**Method-Specific Global Parameters**:
The subset of Global Method Parameters introduced by this method beyond its designated frozen foundation dependencies. They have their own storage budget; excluded foundation dependencies remain part of the disclosed runtime footprint rather than becoming free or hidden state.
_Avoid_: free decoder, uncounted shared weights

**Rebuildable Warm Cache**:
Non-authoritative per-scene runtime state compiled solely from Deployment Scene State and Global Method Parameters, which may be persisted but can be deleted without preventing cold-start query execution. Its storage is disclosed separately, and only bitwise reproduction under its frozen compiler contract preserves the same evidence identity.
_Avoid_: semantic sidecar, optional scene state

**Field-Derived Support Topology**:
Query-independent connectivity compiled solely from Deployment Scene State and Global Method Parameters. It is a Rebuildable Warm Cache; topology constructed directly from teacher or MPR state is a Training Artifact instead.
_Avoid_: teacher-induced deployment graph, stored semantic topology

## Information stages

**Training Artifact**:
State produced or retained for mapping, training, audit, or reconstruction that is unavailable to cold-start query execution and may not be dereferenced while building a Rebuildable Warm Cache.
_Avoid_: deployment dependency, rebuild prerequisite

**Query Workspace**:
State derived from Authorized Query Input during query execution, including exact memoized or audit copies keyed by the complete scene, method, and query identities. It must not become scene state or accumulate information across distinct queries, and its storage is disclosed separately from the Compact cold footprint.
_Avoid_: query-conditioned scene cache, online scene memory

**Mapping Observation**:
An RGB view, camera description, or protocol-authorized derivative that may be used while constructing or training a scene field.
_Avoid_: source RGB

**Scored Sample**:
A frame, object, or query whose prediction is consumed by an evaluator. Being scored does not by itself make every channel of the sample an Evaluation View or Forbidden Evaluation Evidence.
_Avoid_: test frame, target frame

**Evaluation View**:
An image channel that an Evaluation Contract holds out from mapping, training, model selection, and calibration. It is distinct from a Scored Sample whose non-ground-truth channels the contract may authorize for another stage.
_Avoid_: every scored frame, target RGB

**Information Grant**:
A task-specific authorization binding one sample channel and provenance to a stage and purpose. Permission for one channel never declassifies sibling channels or the container that carries them.
_Avoid_: allowed frame, safe file

**Authorized Query Input**:
The text, point, scribble, reference mask, reference image, or other input that a benchmark explicitly gives to the query procedure.
_Avoid_: query hint, source prompt

**Output Request Metadata**:
Non-semantic camera, coordinate-domain, or resolution metadata required to locate or rasterize a requested prediction. It authorizes output placement only and does not declassify any image, label, or sibling channel from the requested sample.
_Avoid_: target view, query evidence

**Evaluator Oracle Action**:
The minimal typed action that an Evaluation Contract authorizes an evaluator to derive from private ground truth and emit as Authorized Query Input. The action does not declassify the ground truth, error region, metric, or selection process that produced it.
_Avoid_: ground-truth access, oracle mask

**Forbidden Evaluation Evidence**:
Any Evaluation View, ground truth, label statistic, or derivative that the benchmark does not authorize as query input.
_Avoid_: target assistance, evaluation guidance

**Evidence Lineage**:
The transitive provenance of observations, artifacts, caches, parameters, predictions, and evaluator outputs. A missing, unknown, mixed, or forbidden ancestor fails closed rather than being overridden by a self-declared compliance flag.
_Avoid_: provenance boolean, trusted filename

**Stage Receipt**:
An immutable, content-addressed record that binds one information stage to its frozen contract, execution identity, observed inputs, emitted outputs, and Runtime Compliance Proof evidence. Only outputs sealed by a successful Stage Receipt may cross into a later stage.
_Avoid_: run log, completion flag, mutable manifest

**Runtime Compliance Proof**:
An immutable, content-addressed proof that one authority row and all of its scene/query executions are closed over declared legal inputs and matching runtime observations under their frozen method, evaluation, implementation, and environment identities. Any missing, failed, undeclared, unobserved, or identity-mismatched child makes the row fail closed.
_Avoid_: compliance flag, self-attestation, best-effort audit

## Query execution

**Canonical Query Interface**:
The sole task-agnostic execution boundary that compiles Authorized Query Input into Canonical Query Evidence, produces a Gaussian Query Posterior, and optionally applies an Output-Domain Operator. Component selection may depend only on the frozen method version, input modality, Query Intent, and output domain.
_Avoid_: unified task head, benchmark query path

**Query Modality Compiler**:
A globally frozen, task-agnostic transform from one authorized input modality into the shared query representation. It may vary by input modality but not by benchmark or task identity, and its products belong to the current Query Workspace.
_Avoid_: task query head, benchmark adapter

**Canonical Query Evidence**:
The common Gaussian-domain representation emitted by every Query Modality Compiler. It carries typed query intent, one or more hypotheses, signed or categorical evidence, optional registered unary evidence, and complete query identity and Evidence Lineage without embedding benchmark identity.
_Avoid_: task logits, prompt cache

**Gaussian Query Posterior**:
The shared calibrated per-Gaussian result produced from Canonical Query Evidence before any output-domain transformation. Binary posteriors use a globally fixed 0.5 decision boundary and categorical posteriors use argmax.
_Avoid_: task mask, benchmark score map

**Capability View**:
A query-independent working representation obtained from the Canonical Capability Feature through the shared decoder and globally frozen semantic, appearance, or boundary projection. It is transient or a Rebuildable Warm Cache, never authoritative scene state or an additional persistent capability bank.
_Avoid_: capability bank, task descriptor

**Query Intent**:
The task-independent distinction between category retrieval and instance selection that may choose a globally frozen solver or calibration preset. It is derived from the Authorized Query Input contract rather than from benchmark identity.
_Avoid_: task mode, benchmark switch

**Query-Local Fitting**:
Deterministic, gradient-free fitting performed solely from the frozen field and the current Authorized Query Input under a frozen algorithm. Its fitted values remain in Query Workspace and may neither update scene state nor influence a distinct query.
_Avoid_: online adaptation, scene fine-tuning, query memory

**Query-Time Vision Model**:
A globally identified vision model that an Evaluation Contract authorizes to consume specific captured-RGB channels during the current query. Its weights, preprocessing, templates, and randomness belong to the Method Contract, while every query-derived output remains in Query Workspace rather than persistent scene state.
_Avoid_: free foundation model, scene feature generator, implicit visual helper

**Boundary-Calibrated Field Region Hierarchy**:
A query-independent hierarchy of Gaussian regions compiled only from the frozen field and Global Method Parameters, with boundaries calibrated from mapping-authorized evidence. It is a Field-Derived Support Topology and never an additional semantic field.
_Avoid_: teacher segmentation graph, prompt graph, persistent proposal bank

**Semantic Region-Hypothesis Marginal**:
A category-retrieval posterior that marginalizes the complete legal set of regions or antichains in the Boundary-Calibrated Field Region Hierarchy instead of selecting a benchmark-specific level or component. Its query-conditioned values belong to Query Workspace.
_Avoid_: LERF head, best component, task hierarchy selector

**Output-Domain Operator**:
A globally frozen, task-agnostic transform from shared query results into a requested spatial domain. It may vary between camera-raster and world-space outputs but not by benchmark or task identity.
_Avoid_: task output head, benchmark renderer

**Camera-Raster Output**:
An Output-Domain Operator that uses only frozen Gaussian geometry, opacity, a Gaussian Query Posterior, and Output Request Metadata to produce an alpha-normalized continuous posterior and visibility in an image raster.
_Avoid_: task mask renderer, RGB refinement

**World-Sample Output**:
An Output-Domain Operator that uses only frozen Gaussian geometry, opacity, and a Gaussian Query Posterior to produce a normalized continuous posterior and support at requested world-space coordinates. Mesh vertices are world-space request points rather than a distinct semantic readout.
_Avoid_: task mesh head, ScanNet interpolator

**Evaluation Adapter**:
A benchmark-specific, semantically inert boundary that validates Information Grants, sanitizes Authorized Query Input and Output Request Metadata, calls the Canonical Query Interface, and converts its result into the evaluator's required format. It may not select semantic parameters, calibrate scores, or alter a Gaussian Query Posterior.
_Avoid_: task head, benchmark postprocessor

**Query Abstention**:
A typed query result declaring that legal evidence or field coverage is insufficient to produce a prediction. It remains distinct from a valid all-background prediction and may not trigger a fallback to undeclared evidence or a legacy path.
_Avoid_: empty mask fallback, silent failure

## Evaluation

**Validated Protocol Artifact**:
A repository artifact whose cohort, evaluator, aggregation, and reproduced comparator have already been audited and may be reused after its provenance, version, and protocol identity are verified.
_Avoid_: unverified historical result, rerun-by-default baseline

**Evaluation Contract**:
A frozen definition of a benchmark cohort, information boundary, query input, output domain, metrics, aggregation, comparator, and artifact identity.
_Avoid_: evaluation setting, benchmark setup

**Benchmark-Consumed Cohort**:
An evaluation cohort whose scores, labels, or evaluator feedback have informed method selection anywhere in the project lineage. A later freeze can make another run prospective, but cannot restore blind-test status to the cohort.
_Avoid_: validation set, newly blind split, unseen benchmark

**Development Evidence**:
Pre-freeze results or evaluator feedback used to select or revise a method. It remains Context Evidence, consumes the affected cohort, and cannot be relabelled later as independent holdout evidence.
_Avoid_: retrospective validation, recovered holdout, provisional test evidence

**Prospective Freeze**:
An immutable, content-addressed boundary that jointly fixes the complete method, model-selection rule, evaluation contracts, evaluator identities, seed policy, cohort manifests, implementation, environment, and compliance identities before a Prospective Evaluation Batch begins.
_Avoid_: checkpoint freeze, configuration snapshot, paper deadline

**Prospective Evaluation Batch**:
The atomic execution of every required task and preregistered seed under one Prospective Freeze, with all predictions sealed before any batch metric is released. Missing executions or identity drift fail the batch closed rather than permitting selective completion.
_Avoid_: final run, task-wise rerun, best-seed evaluation

**Prospectively Locked Confirmation**:
A result on a Benchmark-Consumed Cohort produced only after the method and evaluation batch have been frozen and all predictions sealed before metric release. It may support contract-scoped comparative or SOTA evidence when every applicable gate passes, but never a blind-test, unseen-data, or generalization claim.
_Avoid_: blind rerun, held-out confirmation, prospective blind test

**Available-Nine SPIn-NeRF Cohort**:
The fixed SPIn-NeRF cohort of orchids, leaves, fern, room, horns, fortress, pinecone, truck, and lego. It excludes the unavailable Fork scene and therefore cannot substantiate or be compared directly with a full-ten result.
_Avoid_: full SPIn-NeRF benchmark, nine-of-ten approximation

**SOTA Target**:
A dated tuple of benchmark, cohort, protocol, metric, aggregation, comparator, threshold, and statistical tolerance that a result must satisfy.
_Avoid_: SOTA-level, competitive result, close to SOTA

**Conditional SOTA Target**:
A dated numerical candidate whose comparator has not yet been proven to match its Evaluation Contract. It cannot support a SOTA claim until the same Exact Row Authority satisfies the numerical target, the comparator-identity gate, and the Runtime Compliance Proof; a task with no eligible comparator has no numerical target rather than borrowing one across contracts.
_Avoid_: provisional SOTA, best available number, cross-protocol target

**Evidence Authority Hierarchy**:
A scope-aware ordering in which evaluation contracts define protocol meaning, method contracts define method identity, Exact Row Authorities define results, and registries select current evidence; conflicts fail closed instead of being settled by recency or score.
_Avoid_: newest artifact wins, highest score wins, canonical filename wins

**Exact Row Authority**:
An immutable, content-addressed result record that binds one row's metric values to its evaluation contract, method identity, cohort, evaluator, aggregation, and evidence chain.
_Avoid_: result summary, copied metric, self-declared authority

**Canonical Current Row**:
The at-most-one row selected for a task as the current planning baseline under the Evidence Authority Hierarchy; it does not by itself imply SOTA or unified-claim eligibility, and may be absent when no row meets the minimum authority threshold.
_Avoid_: best number, paper row, latest experiment

**Development Advancement**:
A pre-freeze decision that a single five-benchmark method candidate has earned a more expensive Development Evidence stage under the frozen joint selection rule. It supports continued method selection only and cannot support a benchmark or SOTA claim.
_Avoid_: task promotion, winning checkpoint, provisional claim

**Joint Development Baseline**:
The one complete destination-compliant method identity against which a five-benchmark candidate is paired on the same development manifests and seeds. Per-benchmark best rows and SOTA Targets may diagnose gaps but may not be stitched into this baseline.
_Avoid_: virtual incumbent, per-task baseline bundle, best-of-task baseline

**Development Non-Regression Floor**:
A preregistered task-level lower bound that a candidate must meet against the Joint Development Baseline before cross-task improvement is considered. Improvement elsewhere cannot compensate for a failed floor.
_Avoid_: weighted regression penalty, average safety score, post-hoc tolerance

**Mapping-Only Checkpoint Rule**:
A globally frozen rule that stops per-scene mapping and selects its checkpoint using only protocol-authorized Mapping Observations and a candidate-bound mapping objective. Its constants and deterministic tie-break are identical across tasks and scenes, and no benchmark metric or evaluator output may influence it.
_Avoid_: best benchmark checkpoint, per-task early stopping, evaluator-guided checkpoint

**Claim Promotion**:
The status earned only when one frozen method and one eligible Prospective Evaluation Batch jointly pass every current public-benchmark target and all applicable eligibility gates. A missing target or failed benchmark blocks the status rather than permitting a partial or benchmark-wise claim.
_Avoid_: partial promotion, per-task promotion, best observed batch

**Five-Benchmark Claim Outcome**:
The later engineering result in which one preregistered Single Compact Feature Field method satisfies the SOTA Target for LERF-2D, LERF-3D, LUDVIG-online NVOS, LUDVIG-online Available-Nine SPIn-NeRF, and ScanNet OVS.
_Avoid_: wayfinder destination

**Joint Mapping Objective**:
The single content-bound, task-agnostic two-stage objective that first preserves raw and capability evidence in the Canonical Capability Feature and then adds source-render-consistent region and boundary supervision. Its schedule and constants are identical across tasks and scenes, and it may consume only protocol-authorized Mapping Observations and mapping-stage Training Artifacts.
_Avoid_: task loss bundle, benchmark objective, per-task training recipe

**Joint-Contract Negative Transfer**:
A causally attributed downstream regression caused by one objective family within a destination-compliant method, established through a matched complete-contract comparison and a recovering objective-family control under the frozen seed and non-regression policy. Gradient conflict, proxy-loss movement, or an isolated benchmark result is diagnostic evidence rather than this verdict.
_Avoid_: any task regression, gradient conflict, proxy trade-off

**Objective Intervention Ladder**:
The preregistered bounded sequence of single-variable candidate changes permitted after Joint-Contract Negative Transfer has been established. Each rung creates a new complete candidate identity and is never an execution retry or a benchmark-specific repair.
_Avoid_: hyperparameter sweep, adaptive rescue, retry policy

**Objective Family Control**:
A complete matched mapping candidate that differs from another candidate only by the inclusion of one named Joint Mapping Objective family. It is diagnostic evidence for causal attribution and cannot be stitched by task into a promoted method.
_Avoid_: ablation checkpoint, per-task variant, best arm

**Failure Attribution Cascade**:
The fail-closed ordering that classifies protocol and implementation failures before mapping or representation failures, then query or readout failures, and only then Joint-Contract Negative Transfer. A downstream category may be assigned only after every earlier category has been excluded by its frozen evidence gate.
_Avoid_: failure label, root-cause guess, metric diagnosis

## Unified-query evaluation

**Unified-Query Target**:
One official 3-D instance in one scene, paired by the evaluator with one Query Camera and exactly one authorized query of each frozen modality; the pairing is evaluator-only.
_Avoid_: query record, category query, public target ID

**Query Camera**:
The evaluator-frozen held-out camera from which a Unified-Query Target's pose-free image and Registered 2D Point Query are constructed; it and its exclusion neighborhood are unavailable to field construction.
_Avoid_: source view, target RGB, method camera

**Registered 2D Point Query**:
One positive raster coordinate plus camera geometry, derived by an Evaluator Oracle Action from a private target mask; the interaction UI image is not Authorized Query Input.
_Avoid_: image prompt, rendered-RGB prompt, SAM click

**World 3D Point Query**:
The world-space surface point obtained by back-projecting the paired Registered 2D Point Query with evaluator-private depth, so both prompt modalities identify the same physical point.
_Avoid_: independent 3-D seed, random mesh click

**Query-Pairing Firewall**:
The information rule that keeps Unified-Query Target identities and cross-modality pairing out of every method-facing manifest and gives each query a fresh Query Workspace.
_Avoid_: shared target key, paired method batch

**Method Field Inventory**:
A content-bound per-scene assignment from every authorized query modality to one Method Field Dependency Set, including mapping-receipt identity, artifact hashes, field multiplicity, and the union of all persistent bytes.
_Avoid_: checkpoint list, self-reported model size, largest-field-only cost

**Method Field Dependency Set**:
The non-empty set of persistent feature fields that one modality must open for a cold-start query; shared dependencies are charged once per scene while undeclared or query-selected dependencies fail closed.
_Avoid_: primary field, free auxiliary field, hidden cross-field read

**Modality-Specific Multi-Field Method System**:
One frozen method identity whose complete query response uses more than one modality-scoped persistent field per scene; it may receive a complete-system aggregate but cannot support a Single Universal Field claim and must pay the summed field storage cost.
_Avoid_: universal feature field, multimodal field, free ensemble

**Formal Cohort Derivation Ledger**:
The immutable pre-evaluation record explaining every inclusion, exclusion, and cohort-size decision using only dataset and construction facts; it prevents later method scores from becoming scene-selection inputs.
_Avoid_: replacement notes, convenient scene list, post-result filtering

**Construction Authority**:
The immutable content-addressed binding that proves a benchmark cohort, annotations, source assets, target derivations, and field-exclusion inventory were constructed under the frozen protocol without opening method predictions. It authorizes downstream mapping inputs but does not by itself authorize formal query execution, evaluator access, metric release, or a benchmark row.
_Avoid_: formal release, evaluation approval, valid benchmark row

**Unified-Query Core Cohort**:
The evaluator-private subset of Unified-Query Targets whose frozen text expression identifies the instance without spatial, comparative, ordinal, or negative relational reasoning; all modalities are scored on exactly this same subset.
_Avoid_: easy split, text-only subset, category-query cohort

**Relational Text Challenge**:
The separately reported text-query cohort whose frozen expression requires spatial, comparative, ordinal, or negative relations to identify an instance; it measures system-level relational grounding and does not enter the common-modality core aggregate.
_Avoid_: hard examples, failed core queries, UQ-Mean text penalty

**UQ-Rank**:
The equal-modality mean of scene-macro average precision on a frozen unified-query cohort; it measures threshold-free target-vertex ranking under one shared output domain.
_Avoid_: localization accuracy, peak-in-box, uncalibrated mask quality

**UQ-Mask**:
The equal-modality mean of scene-macro fixed-boundary IoU on a frozen unified-query cohort after a prospectively bound dev-only calibration receipt.
_Avoid_: oracle IoU, test-fitted threshold, raw-score UQ-Mean
