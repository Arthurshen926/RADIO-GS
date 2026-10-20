# RADIO-GS — current context

Updated: 2026-10-20 (migration handoff only). Active worktree `/root/RADIO-GS-query-field`, branch
`codex/tmm-query-response`. Read [ADR0022](docs/adr/0022-query-response-paper-on-frozen-providers.md)
and [the study guide](paper/tmm_response/README.md) before continuing. The active research decision is [ADR0024](docs/adr/0024-v10-method-ownership-and-nvos-evidence.md).

Migration handoff (2026-10-20): [eden to radio](docs/EDEN_TO_RADIO_HANDOFF_20261020.md)
and [external dependencies](docs/EDEN_RADIO_EXTERNAL_ASSETS_20261020.md). This local checkpoint
preserves source/config/test/report changes; no push, radio access, GPU run or shared-asset mutation.
Historical 349 comparisons and 253-passed profile results below were not rerun. The migration's
narrower fixture-only CPU checks are recorded separately. Default and scientific claims stay unchanged.
Read the handoff before treating older source-only/N14 descriptions as current.

Latest user direction (2026-10-18): consolidate the five outputs into an actual
shared evidence core, preserve effective typed backends, then freshly evaluate
all five with performance and cost regression. The proposed implementation and
evaluation sequence is [ADR0042](docs/adr/0042-shared-evidence-core-and-five-output-migration.md).
Separate unchanged-input parity migration from method changes. The requested
ScanNet single-SigLIP2 semantic-teacher main profile differs from historical
dual-native scores; retain both profiles and measure the change explicitly.
NVOS matched-access DINO/SAM attribution and UQIS crop-only SAM refinement are
the first proposed mechanism studies. UQIS map-only access and existing learned
membership remain intact. Latest steering explicitly permits architecture and
CPU implementation work before full GPU regression: follow
[ADR0043](docs/adr/0043-cpu-architecture-consolidation-before-full-regression.md)
and [the CPU consolidation report](docs/experiments/2026-10-18-shared-evidence-cpu-consolidation.md).
Stage 1 is saved as local commit `5674e4b3fe5b3115103be9560fd5963337d2cd40`
on `codex/shared-core-stage1-checkpoint`; the active branch/index were preserved.
CPU consolidation is saved at `2c082a078f357c714b9564d3664a17ce184fc4b4`
on `codex/shared-core-cpu-consolidation`.
Shared operations, read-only compiled/registered adapters, public query permission
checks and optional complete-member chunking are implemented. The current v7
CPU audit passes 31/31 jobs and 349 exact comparisons, including full-carrier
member predictions for 27 fixed anchors, 20 actual ScanNet compiled priors and
Gaussian contribution algebra on 20 retained source views.
Both legacy/shared test runs pass 253, with one CUDA-only test skipped. These
are NOT fresh benchmark scores or complete renderer validation.
The 20 original ScanNet fields remain immutable. All now have optional verified
region/view overlays, reconstructed from 1,158 original caches / 73,905 regions
and replacement-teacher build records; the migration recipe enables them. New
metadata occupies 3.56 MiB; no large field is copied. Original descriptor rows
and replacement CSR/view order match exactly; replacement encoders and original
SAM pixels are NOT replayed. Independent visibility remains missing for these
old fields. See [ADR0044](docs/adr/0044-preserve-region-provenance-and-independent-visibility.md)
and [the source/build report](docs/experiments/2026-10-19-region-provenance-and-build-core.md).
Gaussian build arithmetic now has a common implementation and an optional
separate unconditioned-compositing visibility output; no new GPU map was built.
Shared build remains incomplete. Default stays legacy; chunking and visibility
retention are explicit. Do not infer visibility from region support/nonzeros.
Full logs and pre-scoring prediction seals are now retained within the budget.
Next verification order: fixed representative end-to-end -> one-module small
controls -> frozen five-output regression. N14 is now integrated as an explicit
whole-provider option, using a fresh source-only initializer, signed reference
repair, two-frame transport and the existing native-ray solver. Quality and
feasibility are separate whole-cohort profiles; historical primary remains
feasibility. All 16 input profiles, normalized inputs and source decisions pass
CPU checks; the worker function AST matches the historical implementation.
Actual DINO/SAM/GPU end-to-end parity remains pending; these are not new scores.
The NVOS evaluator now discloses actual target-RGB/negative-scribble permissions
and separates direct-image/native outputs and two GT tracks. Independent sets
remain explicitly inherited from the source-only initializer.
See [ADR0045](docs/adr/0045-integrate-complete-signed-reference-provider.md) and
[the integration report](docs/experiments/2026-10-19-complete-reference-provider-integration.md).
Single-SigLIP2 ScanNet, same-access DINO/SAM controls and crop-only UQIS SAM remain
separate pending experiments. The original source-only migration stays default.
Historical source bindings and scores are preserved; no background queue or full
GPU evaluation was launched in this CPU development phase.

Previous paper milestone (2026-10-18): scientific paper organization under
[ADR0041](docs/adr/0041-scientific-question-and-evidence-led-paper.md).
Read [the question/mechanism/evidence outline](docs/experiments/2026-10-18-paper-scientific-organization.md).
Main paper: partial-evidence authority -> complete response; identity, scope and
output-compatible formation are linked decisions with typed implementations.
Two scoped propositions and a readout counterexample explain failure modes,
not accuracy guarantees. N12/N15 matched reference-scope controls carry the
NVOS mechanism evidence. Preserve N14 primary/secondary and N16 IoU/AP tradeoff.
Chronological variants and optional L11 move to supplements without deleting
results. No fresh inference/default promotion in this manuscript reorganization;
N15/N16 stopping decisions remain active. Older entries below are research history.

## User scope and scientific contract

The TMM paper studies **query-conditioned evidence fusion for 3D responses**.
The latest user explicitly confirms that **v10 is our method**, including its
high scores, native identity, admitted-response fusion and complete membership.
Do not demote these in-house contributions to an external-like frozen provider.
Active decision: [ADR0024](docs/adr/0024-v10-method-ownership-and-nvos-evidence.md),
which supersedes ADR0023's exclusive paper framing. Complete-response replacement
remains an optional measured extension with its own controls and missing transfer.

One evidence-to-response problem has typed implementations: query applicability,
support/exclusion/unknown authority, geometry and complete outputs. Do not claim
all query types use the same solver or a validated universal learned admission.
Do not attribute historical component gains to L11; controls from different
cohorts/interventions are non-additive. External teachers remain attributed.

Teacher representations, latent dimensions and compression are outside scope.
Preserve the full cohort: LERF 4 scenes / 208 observations (semantic and projected
selection outputs), NVOS 8 tasks / 7 scene clusters, ScanNet 10 rooms, UQIS 9 rooms /
268 queries. LERF projected-set IoU is not true 3D GT IoU. Declare source-assisted
versus map-only access and inherited membership supervision. No per-scene winners.

NVOS study N01 crosses old eroded evidence / confidence-weighted area coverage
with 256x192 / 512x384 source grids, keeping original teachers, geometry and 0.5
threshold. All eight predictions precede scoring. Both GT boundary conventions
are reported separately. Other tasks are hash-bound unchanged references, not
fresh inference. GPU v1 failed its explicit memory cap before scoring; CPU CSR
v2 preserved that failure and passed retained-field compatibility (max error
4.65e-5). Complete N01: legacy-low / coverage-low / legacy-high / coverage-high
IoU = 86.5134 / 86.7069 / 85.6138 / 87.7776. The combined change improves 7/8
tasks; horns-center loses 0.1731 pp. AP 98.0222 -> 98.1527, but orchids loses
2.0161 AP points while gaining 1.9049 IoU points. Candidate is not promoted;
the opened cohort is not an independent final test.

N02 (`nvos_source_readout_v1`, ADR0025) completes all eight tasks. The same
N01 coverage/high field yields fixed / fit-source / reconstruction-holdout IoU
87.7776 / 87.5151 / 87.5836. Holdout gains 6/8 on retained GT but loses mean
0.1940 pp; trex loses 3.3978 pp from +118,640 FP versus +30,527 TP. On LUDVIG
GT it gains 4/8 and loses 0.2818 pp. AP is unchanged across readouts. All source
criteria improve, but source-calibrated thresholds do not improve the aggregate.
The common-pixel source split and histogram thresholds independently verify.
No default promotion or task-wise winner stitching; N01 remains the best uniform
measured NVOS variant. A20 uses exact N02 responses and freshly rendered v10:
orchids AP change decomposes into foreground +0.2212 and background -2.2373 pp.
Background scores in [0.01,0.05) grow 94,415 -> 197,928 despite almost unchanged
FP at 0.5. GT-conditioned hybrid scores are diagnostic, not deployable or an
upstream causal ablation. All 8 spatial error plots are retained. N02+A20 persist
974,174 bytes; private prediction banks are released. See
`docs/experiments/2026-10-17-nvos-source-readout.md` and source-readout feedback.
Next test should address local foreground recovery while preserving background
rejection, and verify complete reference scope using legal source evidence.
Do not use A20 GT groups as update gates or continue target-driven threshold
search. Target-RGB/geometry causes remain unisolated.

N03 (`nvos_local_credit_v1`, ADR0026) is complete: six fixed arms, all 48
predictions before scoring. Held-local gives IoU 87.5538 / AP 98.2704; all eight
IoUs improve over v10 (+1.0403 pp). It exceeds signed-edit-matched uniform
attenuation by 0.3209 IoU pp (7/8; descriptive scene interval [0.0750,0.6771]),
but loses 0.2238 pp to N01 coverage (7/8 harmed), while AP gains 0.1177 pp.
Do not call this a full performance improvement or promote it over N01. The
observation-domain control is negligible (+0.000065 pp); holdout adds no IoU
gain over fit-local. A21 identifies lost foreground and added threshold FP
relative to coverage despite lower mean background scores. A22 finds same-frame
seed--track agreement 96.49–99.82%, not GT accuracy or complete extent proof.
See `docs/experiments/2026-10-17-nvos-local-credit.md`. Stop credit-gate constant
searches; prioritize source extent, cross-view transport/geometry, and matched
input-access tests. Do not use diagnostic GT groups as inference permissions.

N04 (`nvos_image_response_v3`, ADR0027) admits observed target RGB as an explicit
input extension. Matched rendered/observed RGB with dense response prompts give
direct-image IoU 87.5575/89.0636 and actual 3D writeback 86.9242/88.3201. The
primary observed-RGB 3D candidate improves 6/8 versus v10 (+1.8067 pp), and exceeds
N01 by 0.5425 pp. AP falls from 98.0222 to about 96.3108. No all-metric or SOTA
claim. The 0.7435-point image-to-3D loss is not yet attributable solely to geometry.
Source teacher losses increase; the update is view-conditioned, not a proven
query-global field. Other benchmarks remain unchanged references. v2 had a worker
guard change after snapshot; only v3 is canonical. Replay: 40/56 exact float
hashes, max difference 1.88e-6, zero binary changes at native and scoring grids.
See `docs/experiments/2026-10-17-nvos-image-response.md`.

N05 (`nvos_partial_image_response_v1`, ADR0028) compares identical inferred
positive points with signed dense, positive-only dense and sparse-only prompts.
All 56 predictions precede scoring. 3D IoU: 88.0159/85.3829/83.6596. Withholding
negative dense evidence harms all eight relative to the signed control (-2.6331
pp). Sparse prompts recover flower but include a second object in horns-center.
No AP measured. No task-wise winner stitching or default promotion. Complete
spatial diagnostics also motivate N06's matched removal of inherited box extent
(ADR0029); see the N05 report. N01 remains the best source-only IoU recipe;
image-conditioned outputs must never be relabelled as that input contract.

N06 (`nvos_unboxed_image_response_v1`, ADR0029) is complete. Boxed sparse /
unboxed sparse / unboxed signed 3D IoU = 83.6596 / 84.1524 / 87.1612.
Removing the box recovers horns-center but harms flower and orchids; the boxed
control reproduces N05's direct-image IoU exactly in both GT tracks. Thus a
missing region outside the old box does not imply that box removal restores the
query's required extent. Stop prompt-constant searches; inspect original query
identity/extent evidence and its reference-to-target transfer. Separately isolate
N04's image-to-3D loss. Keep N04 as best observed-RGB 3D candidate and N01 as best
source-only IoU candidate; neither is silently promoted to a universal method.
A source prompt audit confirms orchids retains all 10,605 positive scribble
pixels; six tasks have full positive-pixel coverage, fern misses 9 and trex 33.
This is sparse input consistency, not complete object GT accuracy. All 150
registered tests pass. New run/diagnostic records total about 3.72 MiB and all
private prediction banks are released.
See `docs/experiments/2026-10-17-nvos-unboxed-response.md`.

N07 (`nvos_memory_response_v1`, ADR0030) is complete: static / two-frame memory /
source-sequence memory direct IoU = 89.0636 / 88.4020 / 88.3470; coarse 3D IoU =
88.3201 / 87.9449 / 87.8852. Thus reference memory alone does not improve N04.
All eight static arrays match N04 exactly; identical orchids memory paths agree.
The incomplete reference seed remains a scope limitation; do not infer complete
object extent from positive-scribble coverage. N08 (ADR0031) was fixed before
N07 scores and isolates native versus pooled-grid writeback of all three exact
image responses. Other four benchmarks remain unchanged references, AP unmeasured.
VCAR's official release now confirms seven targets excluding horns-left and its
own prompt frames/points/boxes; alpha readout also differs. Keep our eight-task
contract and disclose the differences, rather than relabeling a mean as SOTA.
See `docs/experiments/2026-10-17-nvos-memory-response.md`.

N08 (`nvos_native_writeback_v3`, ADR0031) is complete. Fixed static / pair /
sequence evidence gives native 3D IoU 88.8424 / 88.2002 / 88.1124, versus
coarse 88.3201 / 87.9449 / 87.8852. The static secondary arm improves all eight
tasks (+0.5223 pp), leaving only 0.2212 pp of its image-to-3D loss. It is the
best measured observed-RGB 3D variant, +2.3290 pp over v10. The preregistered
sequence primary still fails to exceed static; no task-wise winner stitching,
default promotion, AP or SOTA claim. All source-teacher residuals rise, so this
is still view-conditioned. Other four benchmarks remain unchanged references.
An optional CUDA footprint backend fixes cancellation for thin conics and
matches the installed renderer within 8.35e-7 on all native checks. Matched
coarse-grid controls change mean IoU by at most 0.0000053 pp; legacy defaults
are unchanged. Failed attempts are preserved, without partial GT scoring.
N07 v2 replay has one scoring FP difference in sequence/trex; N08 compares to
its exact v2 parent, not N07 v1. A23 refines A15 to an exact threshold oracle:
90.1098 mean IoU, diagnostic only. Scalar threshold changes cannot bridge the
remaining performance gap. Stop resolution/threshold sweeps; the next method
work should address complete reference extent and spatial discrimination,
with the same all-eight evaluation and explicit input access. All 164 registered
tests pass. See `docs/experiments/2026-10-17-nvos-native-writeback.md`.

N09/N10 are complete (ADR0032/0033), neither improves the static mean.
N09 `nvos_signed_affinity_v1` adds frozen DINOv2 and official negative reference
scribbles; sparse signed authority / reference SAM pseudo-mask authority give
81.5803 / 86.0812 native 3D IoU. All72 responses precede GT. Signed evidence
recovers flower (85.9782 -> 95.0682) and orchids (73.1712 -> 88.8919), but
confuses other instances in horns-center and leaves (67.1628 / 45.5701).
Reference LOO accuracy is 99.8024--100%, so source separability does not prove
target identity. N10 `nvos_direct_affinity_v2` fits the exact raw affinities
without SAM; all48 before GT, with four exact parent controls. Signed/pseudo
direct IoU = 77.7307 / 84.2098. Removing SAM improves AP over its respective
branch but lowers IoU; it is not a solution by itself. Both GT tracks agree on
this direction. No task-wise winner stitching, promotion or SOTA claim.

At the end of N09/N10, the best observed-RGB 3D variant is static native 88.8424 IoU
(subsequently exceeded by N11 below).
N09 replays its image evidence exactly and its per-task binary counts exactly
under both GT conventions; 3D float arrays are not bitwise identical to N08.
The new replay measures AP 96.5817, below v10 98.0222. Source-only N01 remains
87.7776/98.1527 IoU/AP. Other four benchmarks are 23 unchanged references,
not fresh inference. All updated source-teacher residuals rise; outputs remain
view-conditioned. The next method question is instance-constrained extent
recovery using original query/geometry evidence, not source-fit, threshold or
resolution sweeps. This coupling remains unimplemented/unvalidated; do not write
it as an established contribution. See the full N09/N10 report and failure map:
`docs/experiments/2026-10-17-nvos-signed-affinity.md`.

N10 waiting v1 was stopped before any prediction/GT to overlap v2 fitting with
N09 CPU scoring, with receipts retained. Three new run directories total
1,197,630 bytes; all five private prediction/parent-link directories are released.
The immutable 2GiB study budget has 295,033 bytes left. No new weights/durable
dense predictions. 167 registered tests pass. Table generation and paper build
bind new results to code, input hashes and both GT tracks; E01 historical source
archive limitations remain. Logs are under `paper/tmm_response/build/logs/`.

The paper now specifies inherited image Chamfer/rank matching and the actual
15-feature Extra Trees complete-member decoder (64 trees; 22 training / 3
calibration rooms; threshold 0.35; benchmark splits disjoint). No new training.
Its checkpoint and nine map bindings are in `artifacts/inherited_membership_spec.json`
under the paper directory. Historical v10 negative reference evidence is a
predicted mask complement. N09/N10 explicitly add official negative user
scribbles; do not silently attribute that extra input to v10.

N11 (`nvos_instance_extent_v2`, ADR0034) is complete. Exact N09 static/signed
image proposals feed unrestricted, v10-response-anchored, or original-scribble
geometry-anchored additive region recovery. All72 predictions precede GT.
Their 3D IoUs are 83.5805 / 89.9673 / 89.9734. The predeclared geometry primary
is the new best observed-RGB 3D candidate, +1.1310 pp versus static, six tasks
improved and two harmed. AP falls 96.5817 -> 96.4276; LUDVIG GT IoU is 90.0478.
Flower improves 85.9782 -> 94.9302. Instance constraints avoid the unrestricted
horns-center/leaves failures, but orchids drops from unrestricted 87.8176 to
73.5362. Only 2/23 orchid components are admitted and 16 have no geometric
witness; per-region unknown versus negative attribution remains open. Fern/trex
accumulate FP and regress. The geometric anchor exceeds v10 anchoring by just
0.0061 IoU pp; do not claim a demonstrated independent geometric advantage.
The result supports retaining v10's identity capability while adding extent,
not a proven universally better witness. No task-wise winner stitching or SOTA.

Image monotonicity does not guarantee monotone 3D fitting; actual lost TP/added FP
are recorded. Image-to-3D loss is 0.1884 pp. Source teacher residuals increase in
all tasks; still view-conditioned. Other four benchmarks remain 23 unchanged
references. No global default promotion. Next isolate unknown-region admission
and within-region FP correction, not threshold/resolution/source-fit sweeps.
N11 v1 failed a Tensor-to-NumPy digest interface before predictions/GT; preserve
its archive. v2 fixes only that interface. The full eight-task spatial figure,
matched effects and method decision are in
`docs/experiments/2026-10-17-nvos-instance-extent.md`.

170 registered tests pass. New N11/failure/spatial directories total 287,337 bytes;
three private prediction/diagnostic directories and an unused 24-array continuation
cache are released. The original 2GiB study has only 7,696 bytes remaining; do not
start another full run against that limit or silently enlarge it. Source/input
archives are inherited by hash; no new weights or durable dense predictions.
Storage audit: `paper/tmm_response/artifacts/nvos_instance_extent_storage.json`.

N12/N13 are complete (2026-10-18, ADR0035/0036). N12 original / semantic / signed
reference-completion memory yields native IoU 88.2002 / 88.5226 / 91.4774.
The signed primary's AP is 98.7761 and LUDVIG GT IoU 91.4729. It exceeds N11 by
1.5040 IoU pp (4 better / 4 worse); orchids reaches 94.5224, flower 96.5990,
but horns-left falls to 84.4815. The semantic reference already satisfies all
original annotations there; forced point prompting adds unwanted source/target
extent. N13 paired probability / log-odds residuals give 90.7469 / 89.4416 IoU.
Probability residual harms ALL8 versus N12 complete replacement (-0.7305 IoU,
-1.7347 AP pp). Do not retain residual fusion as a required method component.
All72 / all56 predictions respectively preceded GT; source archives and parent
controls validate. Both runs and spatial outputs have released owned RAM banks.

N14 is complete (ADR0037): fixed source feasibility (primary) / source-quality
(secondary) admission gives 92.9424 / 93.1730 native IoU and 98.8599 / 98.9451 AP.
LUDVIG GT IoU is 92.9852 / 93.2104. Quality was the strongest fixed development
recipe at N14, +3.1996 pp versus N11 and +4.3305 versus static, with 5 better / 3 worse
tasks. It improves only trex versus feasibility while harming fern/horns-center;
no universally superior quality selector claim. Both choose source responses
using original-query agreement or SAM-predicted SOURCE quality, no scene names,
target scores, task-wise oracle or tuned margin. All56 before GT; teacher arrays
and actual 3D confusion counts reproduce their selected N12 candidates exactly.
Orchids 95.6011, flower 96.7139, horns-left 94.8079. Quality image/native gap is
0.3241 pp; fern/trex retain internal FP and thin-structure errors. All source
teacher residuals still rise. No independent test, same-protocol SOTA certification,
query-global field or all-five default promotion. Other23 jobs stay references.

The current NVOS architecture is signed reference completion, reference-only
whole-response admission, frozen instance transport and native 3D response fit.
Do not stack the rejected N13 residual or N11 target component gate into it.
N15 now compares original/completed-reference source-view evidence and its fusion;
see the completed result below. Keep all8 and both GT tracks; stop adding
prompt/threshold sweeps or pretending source fit alone certifies accuracy. N14 was designed AFTER N12 scores, explicitly development.

180 registered tests pass; the N13 feedback-validator scope bug was corrected
without changing inference/results and has planned/running/failed status tests.
All owned prediction/parent/plot RAM directories for N12--N14 are released.
The continuation study is `output/tmm_response_20261018/study`, immutable 8MiB
/ 2MiB run / 1MiB file, not an increase to the old unchanged 2GiB study.
New run/figure records total 1,747,141 bytes (1.666MiB); continuation study totals
1,747,244 bytes with 6,641,364 remaining. Old study remains 2,147,475,952 bytes.
See `docs/experiments/2026-10-18-nvos-reference-completion.md` and the three
`paper/tmm_response/artifacts/nvos_reference_*_storage.json` audits.

N15 is COMPLETE (`nvos_multiview_completion_v2`, ADR0038): all 48 outputs
preceded both GT tracks; all 318 source and eight target operator checks passed.
Matched original/completed-reference source-only fusion reaches 88.0061/92.7461
IoU (+4.7400 points), without target-image evidence. This establishes transferable
reference-completion capability, not a source-GT correctness certificate.
Equal-group source+target fusion gives 93.0302 IoU / 98.9797 AP versus replayed
N14 93.1730 / 98.9451: precision rises, recall falls, and IoU loses on both GT
tracks. No fixed arm satisfies the predeclared improvement rule. STOP THIS JOINT
FUSION BRANCH; at N15 retain N14 for maximum IoU and N15 source-only as a distinct
capability result. No weight, view-subset or per-scene winner search. This is
not a universal upper bound. All eight tasks lose TP and reduce FP versus N14
under joint fusion; stronger teacher consistency alone did not increase IoU.
First v1 failed shared-GPU allocation before any prediction/GT; v2 changes only
the run name/GPU. 184 registered tests pass. All owned prediction/source/plot
RAM directories are released; no new weights or durable dense predictions.
N15 records add 585,403 bytes (0.5583MiB); continuation study is now 2,332,647
bytes, 6,055,961 remaining. Old study remains 2,147,475,952 bytes.
See `docs/experiments/2026-10-18-nvos-multiview-completion.md` and
`paper/tmm_response/artifacts/nvos_multiview_completion_storage.json`.
Recent primary-source comparison audit adds LEGO 94.2, WildSeg3D 94.1,
VCAR 93.5 and Trace3D 92.5 as PUBLISHED, not local reproduction rows.
Do not treat exceeding 93.5 as sufficient SOTA certification. Benchmark
comparability requires task/input/prompt/GT/metric agreement, not identical
teacher or geometry implementations; the latter are internal causal controls.

N16 is COMPLETE (`nvos_target_redecode_v1`, ADR0039). All56 sealed before both
GT conventions, exact N14 image replay and binary confusion controls verified.
Full-frame target re-decoding gives 93.0998 IoU / 98.2626 AP; local primary gives
93.2805 / 98.4663, versus fresh N14 93.1730 / 98.9450. N16 is the new highest
observed native IoU, but loses 0.4787 AP points and fails the fixed joint-metric
guard. Retain N14 as the BALANCED recipe, not as the highest-IoU observation.
Four tasks improve/four regress; excluding the whole horns scene removes the
mean-IoU improvement. Local versus full decode improves 0.1806 IoU / 0.2037 AP,
but does not recover N14 response ranking. AP is not probability calibration.
Fern/horns-center crop to the full frame and have exactly identical image arrays.
STOP THIS TARGET RE-DECODING BRANCH and the earlier N15 joint fusion branch;
no context-size, threshold, recursive-decode or per-task winner search. This
is empirical saturation of these fixed refinement recipes, not a global upper
bound or completed SOTA goal. Preserve N15 source-only capability and N16 tradeoff.
188 registered tests pass; eight operator checks <=8.94e-7, unsupported edits 0.
New experiment/plot records total 279,148 bytes (0.2662MiB), no new weights or
persistent dense predictions, all owned RAM directories released. Continuation
study totals 2,611,795 bytes with 5,776,813 remaining; old study 2,147,475,952 unchanged.
Report: `docs/experiments/2026-10-18-nvos-target-redecode.md`; audit:
`paper/tmm_response/artifacts/nvos_target_redecode_storage.json`.

A24 is COMPLETE (`nvos_score_precision_v2`, ADR0040). Read-only decoder observers
replay all24 N14/full/local image probability hashes and decisions exactly;
24 raw-logit maps are sealed with them before both GT tracks. Original AP and
binary counts reproduce exactly. Float64 sigmoid plus the SAME bilinear
probability readout, and stable independent p/q interpolation, change mean AP
by less than 0.002 points. Original -> stable AP: N14 99.133326 -> 99.135218,
full 97.142820 -> 97.142705, local 96.923000 -> 96.922783. All24 hard masks are
unchanged. The local-vs-N14 gap remains -2.212436 AP points; this is real image
response ordering loss, not the suspected sigmoid/interpolation artifact.
No actual 3D rerun or new performance gain: retain N14 balanced/N16 highest-IoU
tradeoff, STOP the current N15/N16 refinement routes and their parameter sweeps.
This empirical plateau is not a universal architecture bound or SOTA success.
A new attempt needs new discriminative evidence and a separate validation design.
Do not keep reopening these routes solely in response to generic continuation.
V1 stopped before GT after one task because a whole-canvas sigmoid assertion
ignored local-operation rounding. V2 reconstructs the original local-sigmoid/
paste order exactly; all old probability hashes remain mandatory. Failure kept.
192 registered tests pass; all owned scratch released. New records 180,806 bytes;
study 2,792,601 / 8,388,608 bytes; old study 2,147,475,952 unchanged. Report:
`docs/experiments/2026-10-18-nvos-score-precision.md`. Other23 jobs remain references.

## Canonical sources and entry point

- Hand-maintained registry: `configs/research/tmm_response.json`.
- Scientific reference: `configs/pipeline/system_v10.json` and its recorded summary.
- Executable study entry: `python -m radio_gs.response_study` (see `--help`).
- Actual kernels remain in `radio_gs/querying/`; precise symbol/status bindings
  are generated in `paper/tmm_response/method_manifest.json`.
- Active method equations: `paper/tmm_response/sections/method.tex`; main paper:
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
Local VALA remains 54.1249 / 79.3526. P/R/O/M identify published / reproduction /
our v10 implementation / optional measured policy; missing learned results remain pending.

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

## Verified study state

v10 is our implemented method, not an external baseline. Main method:
`paper/tmm_response/sections/method.tex`. Historical positive controls attribute
identity admission (+2.541 pp LERF projection), admitted source use (+19.789 pp
LERF semantic), local image correspondence (+4.137 pp), complete-carrier reuse
(+5.022 pp UQIS image), dual-native category probabilities (+2.348 pp on three
rooms), and real-observation point membership (about +9.62 pp). Their interventions
and cohorts differ; never add them or call them L11 gains. Source records and
code bindings are in `evidence.json` and `method_manifest.json`.

E01–E04 are four complete 31-job controlled studies. Permission, normalization,
scope-agreement and solver changes do not produce common gains. Lower squared
source loss in all 22 source-assisted jobs does not guarantee task improvements.
Q01 anchor recentering and Q02 hand verification are negative targeted controls.
The independent set is a separate optimized output, not a thresholded semantic
score: `radio_gs/pipeline/subset.py` fits admitted-pixel silhouette BCE/Dice plus
a footprint cost. v10 NVOS stores both outputs; audit them separately.

L01–L11 and their D/A records cover optional learned response replacement.
L11 scores all ten ScanNet rooms and 268 UQIS queries with one fixed model/margin.
Image IoU 39.8980 -> 40.9859 and AP 59.3088 -> 60.5771; one query improves, none
harms. Other measured outputs remain unchanged. Full and no-null decisions
are identical; that constraint has no demonstrated accuracy effect. LERF/NVOS
transfer of this OPTIONAL extension remains unvalidated. This is not a reason
to label our underlying v10 LERF/NVOS method missing. Do not splice extension
best scores into an unevaluated combined recipe.

The NVOS evidence experiment and inherited-output audits are complete and distinct:
new response construction versus fresh evaluation of already fitted v10 sets.
A17 finds inherited selected-set IoU 85.2245, below continuous 86.5134. A19 crosses
the two fields/readouts: semantic + selected alpha 84.3934 (recall up, precision
down); member + continuous 6.9147 despite 94.3836 AP. Member scores are not
calibrated semantic probabilities; preserve each objective/readout pairing.
A18 scores AP/AUROC on the exact 32 sealed N01 images, with identical binary
counts. The source erosion diagnostic removes 49.27% of trex and 29.42% of orchids
hard teacher-positive pixels at the coarse grid (not ground-truth positives).
All private tmpfs banks are released; six run/preparation directories total
637,397 bytes, with no persistent dense predictions or new weights.
Complete numerical results are in
`docs/experiments/2026-10-17-nvos-evidence-and-v10-method.md` and
`paper/tmm_response/nvos_coverage_feedback.json`. Never inspect partial task scores
and change the remaining experiment. Next method changes should address complete
foreground recovery and source-only response calibration with matching readouts,
using held-out source views and the same all-eight evaluation. Do not compensate
orchids AP or horns IoU with GT-selected per-scene switches. Target-RGB/geometry
access requires a separate matched-input experiment, not causal speculation.

Historical learning chronology and prior continuation instructions are preserved
in `docs/history/2026-10-17-v10-attribution/CONTEXT.md` and daily experiment reports.
They do not supersede ADR0024 or the latest user request.

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
