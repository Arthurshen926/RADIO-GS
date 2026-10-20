# ADR0022: TMM paper on query-conditioned partial-evidence response refinement

Date: 2026-10-16. Status: accepted paper scope; common cross-task gains unresolved.

## Context

The user narrows this paper to one query-response problem and asks to start
implementation and writing for TMM using the strongest existing evidence.
Teacher features, compact field dimension and broad representation redesign
are no longer the paper's research variables. Earlier architecture notes that
require compact-field reconstruction do not determine this paper's scope.

## Decision

Freeze effective teacher/native matching, geometry and typed output providers.
Study query-conditioned admission of partial evidence: permission to support
does not automatically grant permission to exclude. Fixed geometry transports
admitted response updates; unsupported responses retain their prior, and valid
counterevidence may revise high prior scores.

Preserve the full v10 recipe as the performance control. Bind each historical
gain to its real intervention. Extract compatible behavior before changing it.
The new directional residual core remains a candidate. Three matched all-five post-refinement
studies are complete: active-only normalization regresses; available-domain
normalization restores LERF IoU above v10 but leaves ScanNet/UQIS losses.
Observation-level agreement with frozen extent loses to matched-budget uniform
attenuation on UQIS. A lineage audit confirms reuse of evidence consumed upstream;
the fourth study now intervenes inside response generation while preserving all
strong providers. This dependence does not by itself prove the cause of all losses.
The completed 31-job stage study preserves original solvers numerically, with
unchanged threshold/category predictions, and compares diagonal, rendered squared
and rendered Huber reconstruction on identical admitted evidence. Squared source
objectives decrease for all 22 source-assisted jobs, while LERF gains threshold
IoU but loses AP and ScanNet loses mIoU. Huber partly recovers ScanNet accuracy
but remains below its original fusion. UQIS stays a complete, unchanged
compatibility control, not evidence of solver transfer. Do not promote a universal
inverse solver or assemble task-wise winners. Next isolate query-conditioned
admission at response generation under matched evidence/update budgets; lower
fitting loss alone is not an acceptance criterion. Identity, extent, geometry
and view-consistency causes remain unresolved by the solver-only intervention.
Common source files or APIs do not prove a common scientific mechanism has
succeeded. The retained v10 route remains the performance reference.

### Method condensation, 2026-10-16

Name the method **query-conditioned partial-evidence fusion**. One mechanism
contains three successive decisions: query applicability, spatial evidence
authority, and geometric response fusion. Keep the distinction between this
method abstraction and a finished transferable admission algorithm. The native
one-versus-two candidate-score rule has local text evidence; it is not yet a
validated common identity rule for reference, category and point inputs.

The main manuscript now centers on this mechanism and its actual positive
controls. Directional post-refinement, available-domain normalization, scope
agreement and inverse solvers remain diagnostic alternatives, documented in
`paper/tmm_response/diagnostics.tex`; they are not four extra modules in the
method. Positive-evidence monotonicity is not a required property of the
retained consensus rule. Do not rebrand provider gains as shared-core gains.
`method_manifest.json` binds claims to source results and concrete code symbols.
This condensation changes paper organization and the research contract, not
inference defaults or experimental results.

This scope explicitly permits **declared** source-assisted response experiments;
it changes the former source-free compact-system target, not its access ledger.
ADR0021's prohibition on hiding source access, scene-specific winner selection,
or proxy output metrics still applies. UQIS remains map-only under its retained
input contracts; inherited membership supervision is disclosed.

## Consequences

### Repository normalization, 2026-10-16

Use `configs/research/tmm_response.json` as the hand-maintained experiment
registry and `python -m radio_gs.response_study` as the navigation, verification,
export, bounded replay and paper-build entry point. Preserve numerical kernels
at their audited paths. E01–E04 are scientific studies; F01–F03 are interrupted
stage attempts; A01–A05 are compatibility/attribution audits. Generated metrics
must distinguish actual interventions from frozen-set or UQIS compatibility
outputs. Never replace missing historical code with current code silently.

The nine effective v10 UQIS maps use `DirectAnchorMembership`, including
`infer_region` for images. The optional `QueryConditionedMembership` candidate
is not active there. The inherited `infer_region` calls `infer`, so even an
estimated image anchor is forced positive. Earlier prose asserting that the
retained path already avoids this behavior was incorrect. Correct the method
attribution; do not silently modify the measured inference to fit the theory.
Estimated-anchor authority remains a concrete research gap.

The subsequent A06 sparse causal control narrows this gap. Removing only the
positive override leaves all actual sets unchanged: raw self-member scores for
all 67 image anchors already exceed the frozen threshold. Image AP changes by
only +0.000144 pp. Fresh scoring covers all 268 UQIS queries; other tasks are
retained compatibility references. This is not a fifth shared-method study.
After sealing, 23 anchors are incorrect, including 16 whose selected region
contains target elements. Their conditional statistics do not establish full
causality. Prioritize query applicability of the identity condition, rather than
the single positive assignment or confidence of membership conditional on that
identity. Keep the original complete-member capability and actual-point contract.

Q01's query-weighted anchor recentering regresses and nearly matches its shuffled
control. Q02 therefore preserves complete responses explicitly: the original
joint-teacher answer plus the two existing teachers' identity proposals, all
decoded by the original classifier. Query-rank verification raises image AP
slightly but lowers actual IoU; it is not promoted. The finite-pool label oracle
shows 5.20 pp available image-IoU gain, while automatic selection incurs 6.88 pp
regret. This is diagnosis, not an achieved gain or general upper bound. Retain
the interface as research infrastructure; investigate evidence for replacing
the original answer, with repair and damage measured together. Do not tune
verification margins on the opened cohort. A learned/calibrated selector would
need separately declared data and untouched validation. These targeted studies
do not constitute fresh all-31 common-method trials.

Root README, CONTEXT and the paper guide now contain current instructions only;
their exact previous text and hashes reside in `docs/history/`. Historical
continuation instructions no longer govern this branch. Detailed daily evidence
belongs in experiment reports, not repeated appends to the current context.

Active worktree `/root/RADIO-GS-query-field`, branch `codex/tmm-query-response`.
Paper source and generated evidence reside in `paper/tmm_response/`.
Evaluate full LERF2D/LERF projected sets/NVOS/ScanNet OVS/UQIS feedback for each
scientific candidate; concentrate ablations on admission, partial evidence,
matched filtering and transfer. Do not spend paper experiments on latent size
or compression. Existing storage budgets and preserved artifacts remain binding
operational constraints. No acceptance or universal performance claim is made.
