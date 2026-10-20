# RADIO-GS — current context

Updated: 2026-10-17. Active worktree `/root/RADIO-GS-query-field`, branch
`codex/tmm-query-response`. Read [ADR0022](docs/adr/0022-query-response-paper-on-frozen-providers.md)
and [the study guide](paper/tmm_response/README.md) before continuing. The active
research decision is now [ADR0023](docs/adr/0023-learned-complete-response-replacement.md).

## User scope and scientific contract

The user targets TMM with ONE query-response problem and ONE mechanism over
frozen effective providers. Canonical name: **learned complete-response
replacement**. Its partial-evidence background distinguishes query applicability, spatial authority
(support / exclude / unknown), and geometric response fusion. These are not
three new networks. The latest user explicitly rejects further small patches.
The active concrete algorithm is **learned complete-response replacement**:
preserve the incumbent and frozen decoders, learn actual relative answer utility
from separate supervision, and select one intact answer with a globally
calibrated retain/replace decision. The partial-evidence abstraction alone is
not the proposed finished algorithm. Teacher representations, latent dimensions, compression and
broad compact-field retraining are outside this paper's research variables.

Preserve v10's effective abilities; architecture may change for verified ability
transfer. Do not substitute feature retention or training convergence for actual
functional preservation. Evaluate the entire cohort together: LERF 4 scenes with
semantic 2D and selected-set projection outputs, NVOS 8 requests (7 scene clusters),
ScanNet 10 scenes, UQIS 9 rooms / 268 queries. LERF projection is NOT true 3D GT IoU.
LERF/NVOS/ScanNet use declared source-assisted evidence; UQIS is map-only at query
time. Disclose the inherited 22-room membership supervision and 3-room calibration.
Never silently change input access or assemble benchmark/query-wise winners.

## Canonical sources and entry point

- Hand-maintained registry: `configs/research/tmm_response.json`.
- Scientific reference: `configs/pipeline/system_v10.json` and its recorded summary.
- Executable study entry: `python -m radio_gs.response_study` (see `--help`).
- Actual kernels remain in `radio_gs/querying/`; precise symbol/status bindings
  are generated in `paper/tmm_response/method_manifest.json`.
- Active method equations: `paper/tmm_response/sections/replacement_method.tex`; main paper:
  `paper/tmm_response/main.tex`; negative controls: `diagnostics.tex`.
- Generated evidence: `paper/tmm_response/evidence.json`, `*_feedback.json`,
  `artifacts/study_catalog.json`, `artifacts/metrics.csv`, `tables/`, `figures/`.
  Edit source definitions/builders, then `refresh`; do not hand-patch table values.
- Detailed retained reports: `docs/experiments/2026-10-16-tmm-*.md`.

## Paper experiment organization (2026-10-16)

The main manuscript now follows introduction / related work / method /
experimental setup / five task comparisons / mechanism ablations / limitations.
`configs/research/tmm_paper_experiments.json` defines published comparison rows,
protocol groups and a correction ledger. `scripts/research/build_tmm_paper_experiments.py`
recomputes local reproduction aggregates, verifies frozen hashes and builds five
comparison tables plus the matched mechanism-control table. `refresh` invokes it;
`verify` binds generated tables and source receipts through the method manifest.

Reuse verified Occam-LERF2D, VALA-LERF3D, VALA-ScanNet-paper8 and LUDVIG-NVOS
reproductions; do not retrain them merely to populate comparisons. **Correction:**
VALA published LERF3D is 58.02 / 82.85, whereas 43.29 / 64.30 belongs to Dr. Splat.
The old freeze is preserved; the active paper overrides its wrong citation.
Local VALA remains 54.1249 / 79.3526. P/R/B/M identify paper / reproduction /
frozen provider / measured new policy; missing learned results remain pending.

A14 rescored all eight frozen v10 ScanNet predictions using the existing VALA
Gaussian metric: mIoU19/15/10 = 66.9480 / 67.8641 / 78.0653 percent. This is a
metric bridge, not RGB-geometry equivalence or a new method gain; original
10-scene mesh mIoUs remain 61.0115 / 62.5276 / 72.4861. The audit persists
96,507 bytes and no dense predictions. UQIS has no cited external paper row;
its regional provider, full v10 and latest L11 are compared under the local
four-modality protocol. LERF/NVOS learned transfer remains open.

See `docs/experiments/2026-10-16-tmm-paper-experiment-organization.md` for the
comparison sources, completed controls and minimal remaining experiments.

## Protocol correctness audit (A15/A16)

Independent CPU scoring reproduces all 208 retained LERF selected-mask IoUs,
all ten ScanNet mesh scores and all eight Gaussian scores; 1,134,207 pseudo
labels have identical hashes and 256 samples match the released VALA function.
LERF semantic 70.4007 is reaggregated, not freshly rendered in this audit.
Provided-mesh8 mIoU19 is 61.0567 versus Gaussian8 66.9480: +5.8913 points is
metric-domain change, not method gain. Strong teachers/carriers and v10 response
paths still confound attribution of external comparison gains to L11.

NVOS retained IoU 86.5134 versus local three-seed LUDVIG-SAM 91.2577. Both use
positive scribbles; ours uses original source RGB but excludes target RGB from
geometry fitting/semantic tracking, while released LUDVIG uses all-view RGB.
Some GT PNGs have gray edges: ours uses >127, LUDVIG uses integer division by
mask maximum. Changing GT only yields 86.6835, borrowing its response min-max /
0.25 readout yields 86.6085, and bilinear implementation change is negligible.
Preserve original historical scores; these are audit controls, not promotion.
Flower/trex/orchids account for 78.04% of the net gap. Flower/orchids miss whole
foreground parts; trex additionally includes background. A GT-chosen 99-threshold
grid obtains 90.0921 but is NOT deployable or a general upper bound. RGB,
geometry and evidence-inference effects still require matched-input controls.
Details: `docs/experiments/2026-10-16-tmm-protocol-correctness-audit.md`.

## Verified state

Four complete scientific studies E01–E04, each 31 jobs before scoring. E01 tests
partial versus complement permissions; E02 available-domain normalization; E03
scope agreement versus matched signed budget; E04 diagonal/squared/Huber solvers
at original response generation. None is promoted. Three interrupted stage
attempts F01–F03 are failures, not additional scientific results. A01–A06 denote
compatibility and attribution audits, not new accuracy methods. Q01 is a separate
targeted identity-conditioning study; Q02 tests selection among preserved complete
responses. Each intervenes on 67 image queries and scores all 268 UQIS queries;
other benchmark values are historical references, not new inference.

E04 reduces squared source objectives in all 22 source-assisted jobs, but task
accuracy is mixed: LERF IoU improves while AP falls; ScanNet loses accuracy;
Huber recovers part of the squared-solver loss. UQIS is unchanged compatibility
only. All independent selection heads are frozen. Original solvers retain
threshold/category behavior within the documented numerical tolerances.
A lower fitting loss or a common API is not evidence of shared mechanism gains.

Positive historical evidence: text identity admission improves LERF selected
projection by 2.541 pp; admitted-response use improves LERF semantic IoU by
19.789 pp in a DIFFERENT control. They are nonadditive local controls. Local image
correspondence and complete membership gains belong to retained providers.
No universal admission algorithm, independent test, or SOTA-equivalence claim.

Effective UQIS implementation: all nine v10 maps use `DirectAnchorMembership` for
actual points and estimated image anchors. `QueryConditionedMembership` is an
optional candidate, NOT the deployed v10 image provider. `infer_region` calls
`infer`, which also forces the estimated anchor to probability one. The metadata
says identity is uncertain, but positive authority does not yet reflect that.
This limitation is explicit in the manuscript; inference defaults are unchanged.
Completed A06 isolates only the estimated-anchor override: all 67 raw classifier
self-scores are 0.914–0.939, above .35. Actual sets are exact; image AP changes only
+0.000144 pp (descriptive 95% interval crosses zero). All 268 UQIS queries were
freshly scored after patch sealing. Other families are retained compatibility
references, NOT new all-31 inference. This is an audit, not a fifth common study.
Of 67 estimated anchors, 44 are correct and 23 incorrect; 16 of the latter have
target elements in the selected region, while 7 regions have none. Conditional
IoU statistics are diagnostic, not causal effects. Raw self-member confidence
has identity-correctness AUROC .5316. Do not treat it as query-identity confidence.

Q01 retains the full-teacher seed and complete decoder, but chooses an anchor
using other-view query-weighted spatial evidence; its control permutes the same
weights within each view. Original image outputs replay exactly. Image IoU
39.8980 -> 38.7674%, versus 38.7483% for the control; query-minus-control is only
+0.0191 pp, descriptive interval [-.3648,+.4363]. Both arms change 55 anchors;
correct counts are 44/67 (query), 43/67 (control), 44/67 (v10). Q01 is not promoted.
Among 43 correct-to-correct cases, query-mean IoU also declines, so identity-bit
accuracy alone cannot verify complete-response preservation. See the Q01 report.

Q02 keeps the original complete answer and adds each existing teacher's identity
winner, decoded by the same frozen classifier. A common-view query-rank contrast
selects one whole answer, with a within-view permutation control. Image AP rises
59.3088 -> 59.5690%, but actual IoU falls 39.8980 -> 38.2163%; control 36.4864%.
No promotion. The finite three-answer label oracle is 45.0975% (diagnosis only):
27 queries have a better candidate; 5.1995 pp available gain and 6.8811 pp
selection regret explain the -1.6816 pp actual change. Five formerly correct
anchors become wrong; three old errors are repaired. See Q02 report/feedback.

L01/L02 now execute the learned replacement architecture (ADR0023). Fit 629
natural expressions on 22 rooms; calibrate on 207/3 rooms; evaluate 175/4 opened
development rooms. Development IoU 7.0088 -> 9.9063%, but only two rooms improve:
the declared gate FAILS. Permuted query evidence gives 9.6243%. The frozen policy
then changes text/image candidates on UQIS and scores all 268 queries: text IoU
16.3865 -> 24.3498%, AP 26.5959 -> 37.8907%; image IoU 39.8980 -> 34.7244%, AP
59.3088 -> 52.2711%. Actual points are exact. No promotion or modality stitching.

A07 replays all 134 complete candidate pools EXACTLY. Same-identity provider
swap gives text IoU 20.0983%; the full policy reaches 24.3498%, with a finite-pool
oracle of 31.7813%. Query permutation gives 19.6002%; the full-minus-permuted
descriptive 9-scene interval still crosses zero. Image finite-pool oracle is
49.3903%, available gain 9.4923 pp and decision regret 14.6659 pp: the candidates
retain useful ability, while the learned replacement decision loses it. These
oracles are diagnostic, not automatic performance. Do not attribute every text
gain to the learner; some comes from frozen complete-provider reuse. Results:
`docs/experiments/2026-10-16-tmm-learned-response-replacement.md`.

## Next scientific work

Completed: D02 `category_native_responses_v1` has original Mask-Adapter
responses for all 29 independent rooms, 2,643 views and 97,532 regions, in exact
19-class order. Only small cosine responses are saved; no new dense field.
D03 `category_utility_supervision_v1` completed literal-category SAM3 and
original SigLIP foreground/context identity, producing intact dual-native/source,
dual-native and source-readout menus for all 29 rooms / 116 episodes. Its
depth-tested mesh carrier is NOT the benchmark Gaussian
renderer. Three candidate/evidence view folds plus one full-map episode per room
remain 29 rooms; full-map evidence is reused and full-scan geometry is inherited.

L07 is complete in `response_value_multitask_v1`: one set-utility estimator
jointly fits natural text, image and complete category menus. No task ID enters
the learner. Fit uses all four category episodes; calibration/development use
only the full-map episode. One shared margin protects 15 room/output cells:
text IoU, image IoU and menu mIoU19/15/10 in each of three calibration rooms.
The six missing calibration classes, AP guard,
LERF independent-set candidates, source Gaussian transfer and full-five gate
remain open. `run_tmm_multitask_program.py` sequences corpus completion, joint
fit, refresh and verify; a failure stops it, and it never promotes the default.
Live generated status: `paper/tmm_response/category_supervision_feedback.json`.
The full model uses margin 0.09309921251417219. On opened development rooms,
image IoU improves 8.7794 pp, below query-blind's 9.7302 pp; text and mean-menu
utility are unchanged. All four category pools have zero mean-utility headroom,
so strictly positive category gain is impossible in this fixed development pool.
The declared gate fails and remains unchanged; this is not proof that a better
candidate family or the ideal architecture cannot work.

L09 `response_value_centered_v1` was sealed BEFORE L07/L08 results, motivated by
label-free input-domain differences. It removes absolute/set-mean context and
keeps only candidate-minus-mean inputs, with identical data/capacity/calibration.
Both full and blind models calibrate to reject-all. Do not adopt this control or
repeat dense inference to report an algebraic incumbent identity as a new score.
A11 `multitask_utility_audit_v1` audits sealed gains and input ranges. Thirty of
thirty Gaussian candidate selected fractions are outside the fit full-map range;
this is not causal attribution. Absolute-context removal did not solve the issue.
Generated combined report: `docs/experiments/2026-10-16-tmm-joint-utility-diagnosis.md`.
All 111 registered tests passed. Current method figure is
`paper/tmm_response/figures/replacement_overview.pdf`; the old admission figure
remains only for the archived method.

D04 `category_transfer_inputs_v2` now seals all ten original Gaussian ScanNet
candidate pools: 1,158 views and 73,905 regions. Original primary/fused responses
replay with maximum probability error 7.16e-7 and ZERO changed labels in every
19/15/10 submenu. Candidate zero retains original bytes; its argmax also matches
the original logits exactly. CPU preparation v1 changed one tied 10-class label
before GT/model access and was stopped (F05); v2 restores original CUDA native
computation. Three dense menus use 357,954,300 bytes in private tmpfs, with zero
new persistent dense predictions. L08 completed and released this bank.

L08 is complete after D03/L07: the SAME checkpoint and
global margin choose ten complete category menus and 134 UQIS text/image answers,
then score all 268 UQIS queries. All 19 job predictions precede target scoring.
ScanNet mIoU19/15/10 stays 61.0115/62.5276/72.4861%; UQIS image IoU/AP becomes
40.8668/60.4594% from 39.8980/59.3088%, with one query improved and one harmed.
Text and both actual point outputs remain exact. Image full-minus-blind is
0.6056 pp and full-minus-permuted 0.3598 pp; both descriptive scene-bootstrap
intervals include zero. L06 image IoU 40.9859% remains higher. No new best claim.
A10 separates intact-candidate capacity from decision regret; cached UQIS
candidate utilities require exact set/order and target/geometry hash matches plus
fresh selected-score agreement. These are diagnostics, never GT routing.
A10 finds 15.3947 pp text capacity unused, image capacity 9.4923 pp with 8.5235 pp
decision regret, and menu mean capacity 0.9131 pp (all-output protected 0.8480 pp).
Do not conflate the small category candidate capacity with the large instance
decision gap. The ideal architecture is not disproved by these intermediate pools.
`run_tmm_multitask_transfer_program.py` (active run `multitask_transfer_program_v2`)
completed evaluation, diagnosis, refresh and verification; no default promotion.
The earlier waiting orchestrator was stopped before predictions to correct
old/new evaluator metadata binding; its receipt remains. Five-output status:
`paper/tmm_response/multitask_transfer_feedback.json`. LERF/NVOS are still pending.
Category signatures on retained Gaussians use positive native-alpha support and
its per-view union, unlike full geometric mesh visibility in supervision; this
domain shift is explicit and must be assessed, not called representation parity.
All transient category/UQIS predictions (peak 1,777,706,508 bytes) are released;
no new persistent dense predictions. There are no outstanding jobs in these two
programs. LERF and NVOS learned-policy transfer remain unfinished. NVOS's one
cached positive region per view cannot provide the current signature's required
within-view query contrast. Next work should address this observation contract
and the unexploited query-conditioned instance utility, not weaken margins or
repeat teacher compilation without a new candidate/evidence hypothesis.

L10/L11 now complete a controlled structural-prior plus query-residual utility
experiment (`response_value_residual_v1` / `response_value_residual_transfer_v1`).
Five room-grouped folds construct out-of-room prior targets; matched query nulls
have zero correction, NOT zero actual utility. Full and no-null each use 128
trees versus L07's 64. Same teachers, candidates, splits, weights and 15-cell
global guard. Full margin .1505214775443651; development gate fails. D05 repeats
the original CUDA category preparation in `category_transfer_inputs_v3`.
All 10 menu plus 134 instance decisions for five arms seal before fresh scoring.
Full image IoU/AP 40.9859/60.5771% matches L06's best, repairs one query and harms
none; text, points and ScanNet unchanged. Full and same-capacity no-null choose
the same answer in 144/144 decisions: no demonstrated null-penalty gain.
Zero-margin full text/image IoU reaches 22.1666/41.1659%, but menu mIoU19 drops
to 60.5368%. Do not deploy this tradeoff or task-wise stitch its gains.
A12 validates exact pools; A13 separates ranking regret from signed retention
cost: text 9.6147/5.7800 pp, image 8.2244/.1800 pp, mean-menu 1.5187/-.6057 pp.
Negative retention cost is damage prevented. Thus the image gap is mostly
ranking, whereas text also loses useful replacements at the common guard.
Training null-correction MSE drops 97.91%, but this is not utility accuracy.
Report: `docs/experiments/2026-10-16-tmm-residual-utility.md`; paper includes the
controlled hypothesis, not a promoted new module. All transient arrays released
(peak 2,132,644,560 bytes); no new persistent dense predictions or active jobs.
Next work must address informative observation/utility evidence and legal
LERF/NVOS adapters; do not continue opened-benchmark margin/capacity searches.

A08 `source_decision_audit_v1` recomputes legal source agreement for all
22 source jobs on exact E02 candidates, seals choices, then joins cached scores.
LERF chooses one query response across frames; ScanNet one menu across 19/15/10
splits. This is a diagnostic, not trained transfer or fresh target rendering.
Authorized Brier agreement chooses all ten ScanNet alternatives but harms seven;
LERF IoU gain survives correspondence permutation exactly. Do not substitute
source fitting for actual utility. Historical E01-E04 pool oracle protecting
EVERY reported metric within each decision unit leaves only +1.3434 pp LERF IoU,
+0.0481 pp NVOS IoU and +0.2053 pp ScanNet mIoU19. These are label diagnostics.
LERF's independent selection still has no alternative. Source tasks need
functionally different complete candidates; UQIS retains larger pool headroom.
See `docs/experiments/2026-10-16-tmm-cross-task-capacity.md`.

A09 `cross_task_targets_v1` reconstructs official menu/union target manifests
on the same 29 separate rooms (22/3/4), validates exact official mesh order and
room-prefix disjointness, and defines 87 chronological cross-view episodes.
They remain 29 independent rooms; full-scan geometry and boundary pose dependence
remain. No dense labels/features saved. Fit covers 19 classes; calibration lacks
bookshelf, counter, shower curtain, toilet, sink and bathtub. Target preparation
is complete; its historical receipt does NOT establish utility readiness. D02
has since completed Mask-Adapter; D03 also completed literal-category SAM inputs.
Do not substitute UQIS DINO for ScanNet
teachers or claim data readiness from target counts.

Do not return to estimated-anchor overrides, rank-max anchor shifts, weighted
centers, Q02 contrast-margin sweeps, or source-solver fitting-loss optimization.
The shared signature/guard is `radio_gs/querying/response_replacement.py`, and
the current common-potential implementation is `radio_gs/querying/response_value.py`;
protocol: `configs/research/response_value_multitask.json`. It compares both frozen
direct-member and scope answers, using all applicable teachers' observable
query/response agreement. The target is actual complete-set IoU difference,
with consistent potential differences and explicit incumbent retention. The
earlier antisymmetric pairwise learner is a recorded control.

L01 (`response_replacement_train_v2`) fits on 22 separate rooms, calibrates on
three, and checks four previously opened development rooms. Fit expressions are
deterministically capped at 32/room; all calibration/development expressions are
used. The inherited member provider also used the 22 fit rooms. L02
(`response_replacement_transfer_v1`) transfers the same frozen policy to both
text and image UQIS queries, all 268 scored. Its guards explicitly distinguish
independent training from already-opened evaluation. See registry/summary files
for completed status; do not infer success from a scheduled or in-progress run.
The initial L01 v1 preparation stopped BEFORE fitting/calibration to include the
second complete provider; its stop receipt is retained.

Prioritize joint text/image utility supervision on independent rooms, keeping
the complete frozen providers and the same retain/replace interface. In the
25 replaced UQIS image queries the predicted mean gain is +2.1255 pp but actual
mean change is -13.0713 pp (query mean, not scene macro). Investigate this utility
transfer failure with independent paired data, not a margin sweep on opened
benchmark queries or dropping the image teacher to match text-only training.
The current data do not uniquely prove which training/evidence factor causes it.

The joint phase is now implemented: D01 `response_replacement_images_v2`, L03
`response_replacement_joint_v1`, L04 `response_replacement_joint_transfer_v1`.
Protocol: `configs/research/response_replacement_joint.json`. It constructs real
image queries on all 29 independent rooms using original RGB-D, SAM regions,
SigLIP and DINO local Chamfer. Reference views and nearby time/poses are removed
from every candidate's map inputs; full-scan geometry is retained. A constructor
preflight found 290 eligible image episodes. Read run summaries for actual
completion/results, not this planned count. The initial image preparation v1
stopped at 10 rooms because fixed temporal references missed targets; its stop
receipt is retained and no model was fitted. V2 uses independent annotated
geometric coverage to choose references, requiring two remaining target views
and 50% target visibility. This changes independent training construction only.
Text reuses sealed L01 candidate supervision with the deployed scope incumbent;
image uses the direct incumbent. One model, modality-balanced loss, one global
margin constrained in all six calibration room/modality cells. No modality label
enters the learner. Do not modify sealed compiler sources while this run is live.
GPU lanes used disjoint room lists and the same bounded run; no dense map copies.
D01 completed all 29 rooms / 290 real image queries (237 fit, 28 calibration,
25 development). L03's full pairwise learner calibrated to reject all: the
largest predicted image calibration gain (+15.45 pp) actually loses 33.39 pp.
L04 full therefore preserves v10 exactly; unrestricted text/image IoU is
24.6407/37.2297%, still an image loss. This is NOT a successful joint mechanism.

L05/L06 (`response_value_train_v1` / `response_value_transfer_v1`) tests one
architecture change: common set-conditioned utility instead of unrelated ordered
pairs. `response_value.py` takes candidate signature, set mean and their
difference, and learns query-centered actual candidate IoU. Gains are consistent
potential differences. The split, examples, candidate providers, signature,
tree capacity and six-cell global margin rule are unchanged. Protocol
`configs/research/response_value.json` was fixed from L03 CALIBRATION feedback
before opening new L04 benchmark scores. Reuse of calibration for architecture
design is explicit; it is not an untouched model-selection test. Do not present
consistency as proof of accuracy. L05 calibration margin is 0.12074169985454469;
development image IoU 20.5197 -> 25.6403%, text unchanged; query-blind is stronger
and the declared gate fails. L06 completed 134 interventions and all 268 scores:
image IoU 39.8980 -> 40.9859%, AP 59.3088 -> 60.5771%; one image query improves,
none worsens, all text and actual point outputs are unchanged. All 134 complete
candidate pools replay exactly against A07. Full and query-blind choose identical
candidates; explicit query-feature contribution is NOT established. Unrestricted
text/image IoU is 23.0100/37.8403%, so reducing retention still harms images.
Matched zero-margin diagnosis from sealed gains plus exact A07 utilities gives
full-minus-permuted -0.0071 pp text and +0.6659 pp image; both descriptive intervals
cross zero. These extra controls have actual IoU only, not new dense AP results.
Reports: `2026-10-16-tmm-joint-response-replacement.md` and
`2026-10-16-tmm-response-value.md` under `docs/experiments/`. L06 is not promoted.

Next work must address informative utility evidence AND cross-task supervision,
not continue to sweep this UQIS gate. Existing candidates retain headroom, while
the current utility learner cannot reliably recover it. The source-category
training/evaluation chain has completed its full corpus and fixed-model
measurements under the same complete-answer contract. A small conservative improvement or algebraic cycle
consistency does not close the query-specific mechanism contribution.

The main manuscript has been rewritten around the complete-response decision;
the prior partial-evidence main source is archived at
`paper/tmm_response/archive/partial_evidence_main_20261016.tex`. Main and standalone
method notes share `sections/replacement_method.tex`; historical provider gains
remain background, not new-learner claims.

Full-five validation remains required. The regional signature now has instance
and complete-category adapters. Source-assisted LERF/NVOS still need independent
supervision/adapters; ScanNet has complete joint training and measurements but
no learned accuracy gain. Keep LERF/NVOS reference values labelled transfer
pending, not new mechanism successes.
In particular, preserve LERF's independent selected set and ScanNet's whole-menu
competition. Do not apply per-query/per-class winner stitching or benchmark
threshold fitting. Method definition: `paper/tmm_response/replacement.tex`.

## Reproduction and storage

Study root: `output/tmm_response_20261016/study`, linked to the shared results pool.
Budgets remain 2 GiB per study, 256 MiB per run, 16 MiB per artifact, 1 GiB free
reserve. Keep code/configs, input/source hashes, seals, logs and compact metrics;
reference protected teacher/baseline assets instead of copying. Do not delete
unrelated historical artifacts as part of documentation normalization.

E01 has 8 missing source-version receipts (some are repeated stage/path versions);
its sealed reports and metric aggregates can be checked, but exact historical
inference replay is NOT certified. E02 has matching archived source text. E03/E04
prediction sources match current files; dense predictions were released from RAM /
private tmpfs and need recomputation for fresh scoring. Hashes are not backups.
The registry disables automatic replay of E01/E02's old persistent writers.
The bounded E03/E04 launchers require a new run name and matching code hashes.

`status`/`verify` are read-only and stdlib-only. `refresh` rebuilds small artifacts;
`test` runs the targeted suite; `paper` builds PDFs. Numerical jobs use
`/root/miniconda3/envs/cybersim_agent/bin/python` with OMP/OPENBLAS threads set to 4/1.
The workspace includes extensive unrelated historical edits; preserve them.

## History

The previous 3349-line context and obsolete entry docs are preserved byte-for-byte
in [the snapshot manifest](docs/history/2026-10-16-normalization-snapshots.json).
They are historical evidence, not active continuation instructions. Do not append
new daily logs here: keep this file current and concise, detailed results in
experiment reports, and decisions in ADRs.
