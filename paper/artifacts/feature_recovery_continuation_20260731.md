# Feature-field recovery continuation — 2026-07-31

## Decision summary

- Keep the frozen LERF total-alpha observation operator.  In the valid-conditioned
  parameterization this is exactly `valid_normalization=true, coverage_power=1`.
- Reject `coverage_power=0` on the already-opened four-scene LERF diagnostic.  It
  improves localization by three samples but lowers both scene-macro and
  sample-micro mIoU.
- Reject `reliability_attention_mode=input_only` as an isolated change on the
  legacy SurfaceRegion cache.  The scene-disjoint validation score drops by
  `0.0011307288`.
- Continue the SurfaceRegion route with a fixed core-geodesic teacher target
  that is independent of student input sampling.  Compare candidate pools
  `256→1024`, context, and reliability only by exact teacher-tensor replay.
- Screen canonical field capacity only with held-out, query-free raw
  RADIO/DINO/SAM fidelity.  Do not use LERF or final NVOS scenes to choose a
  capacity rung.

## Why the current field is weaker than the historical HCD path

The gap is not explained by the label “two-channel fusion” alone.  The
historical HCD geometric and semantic encoder branches are architecturally
similar 1x1 networks and do not receive branch-specific supervision; moreover,
the old `no-hybrid` ablation can outperform the full hybrid path.  Reintroducing
an extra branch by name would therefore not be a causal repair.

The material differences are capacity and supervision.  The current primitive
fusion is effectively a local 128-D code plus coverage/agreement reliability,
passed through a 192-D gate and affine 1280-D readout; its coarse/spatial
content dimension is zero.  Reliability conditions trust but does not supply a
second semantic or geometric content stream.  Historical HCD instead had a
deeper nonlinear decoder, a screen-space refiner, dense pixel-level training,
and direct supervision of the final downstream feature.  Those mechanisms can
preserve small text-cosine margins and local peaks even when mean descriptor
cosine changes little.

This distinction determines the recovery plan.  Generic independent-cosine
response distillation repairs the final descriptor-to-text consumer without
benchmark vocabulary or a new inference branch.  The query-free V5 ladder then
isolates width (`R->W`), primitive spatial content (`W->S`), and HCD-like
nonlinear depth (`S->D`) under raw RADIO/DINO/SAM fidelity.  The old
screen-space GroupNorm/refiner is not copied directly because it would break
the canonical primitive-truth contract and can make point and rendered queries
inconsistent.

## LERF valid-conditioned control

Outputs:

- `output/optimization_20260731/text_specificity_validnorm_beta1_control`
- `output/optimization_20260731/text_specificity_validnorm_beta0_candidate`

For primitive score `s`, semantic validity `v`, and raster weight `w`, define

`N=sum(w*v*s)`, `V=sum(w*v)`, `A=sum(w)`, and `C=V/A`.

The tested family is `S_beta=(N/V)*C^beta`.  Therefore `S_1=N/A`, exactly the
historical total-alpha renderer.  The beta-one run reproduces the 2026-07-24
baseline exactly at scene and category level.

| Aggregate | beta=1 control | beta=0 candidate | delta |
|---|---:|---:|---:|
| scene-macro mIoU | 0.31963464 | 0.31768984 | -0.00194480 |
| sample-micro mIoU | 0.33408335 | 0.33163702 | -0.00244633 |
| scene-macro localization | 0.59194661 | 0.60322616 | +0.01127954 |
| sample localization | 122/208 | 125/208 | +3/208 |

The mIoU loss is broad: among 67 scene-query categories, 35 decrease, 10
increase, and 22 are unchanged.  Ramen accounts for approximately 99.8% of the
net IoU loss.  The localization gain is limited to `hand`, `wavy noodles`, and
`yellow pouf`; two are singleton categories.  This does not justify removing
coverage abstention.

Any intermediate beta must be pre-registered and frozen on an independent
development set.  The four LERF scenes above are now evaluation diagnostics,
not a beta-selection set.

## SurfaceRegion isolated reliability ablation

Legacy query-free caches:

- train: `output/scannet_pfir_small_v1/readout_v3/train_cache.pt`
- validation: `output/scannet_pfir_small_v1/readout_v3/validation_cache.pt`

| Readout | best selection score | token cosine | descriptor cosine | all-view cosine |
|---|---:|---:|---:|---:|
| legacy `log_prior` | 0.93809109 | 0.84545965 | 0.95161662 | 0.92456556 |
| isolated `input_only` | 0.93696036 | 0.83813489 | 0.95044989 | 0.92347083 |

The isolated readout-mode change is rejected.  It cannot restore information
that was never admitted into the region token set.

## Restoring actual context support

The legacy cache declares radial stratification but omits
`token_candidate_limit`, so compatibility resolution sets the limit to the
stored budget of 256.  Dense cores can therefore consume the candidate list
before the context shell is observed.

The student replacement cache contract is:

- relation-weighted DINO/SAM/geometric shortest paths;
- `token_candidate_limit=1024`;
- `core_context_radial_stratified_v1`;
- `maximum_tokens=256`;
- `core_token_fraction=0.60`;
- `reliability_semantics=uniform_valid`;
- full physical-space exclusion of known ScanNet evaluation and PFIR
  development/test candidates.

The first completed 8-scene, 96-region diagnostic shard is:

`output/optimization_20260731/surface_context1024_uniform_v1/train_shard0.pt`

On the same eight scenes, the token composition changes as intended:

| Cache | mean core tokens | mean context tokens | regions with context | context >=64 |
|---|---:|---:|---:|---:|
| legacy candidate limit 256 | 198.30 | 20.08 | 43/96 | 7/96 |
| new candidate limit 1024 | 150.56 | 73.81 | 91/96 | 63/96 |

This is a same-scene controlled comparison, not a paired-region causal
comparison: only 1/96 region keys and 0/96 exact teacher crops coincide after
rebuilding.  The new cache also has more three-view teachers (58/96 versus
41/96), so teacher resampling is an additional confounder.

The diagnostic shard has no failed scene or below-minimum region.  Its input
contract digest is
`36f45e4b0e223bb13763df46d405f0d35dc00223b71d074654f73d25f540d191`.
It predates the fixed-teacher target semantics below and is therefore obsolete
for promotion.  It must not be mixed with the new v2 caches.

With `uniform_valid`, every valid token has reliability one, so `log_prior`
adds exactly `log(1)=0`; `input_only` and `log_prior` are numerically
equivalent under this contract.

The promotion-grade implementation now defines the teacher as a
`fixed_core_geodesic_support_without_input_context_v1` region.  The input
context shell only conditions the student and can no longer expand its target.
The rectangular teacher crop is explicitly a core-support-defined unmasked
bounding box, not a pure-core pixel mask.  A candidate-limit hit at 4096 fails
closed instead of silently claiming a complete teacher ball.

The candidate-256/geometric control writes fresh official summary tokens,
official crop descriptors, and masks.  Every treatment reuses the same
seed/radius order and copies those target tensors bit-for-bit.  Replay verifies
the physical support hash, crop boxes, medoid, target hash, scene set,
checkpoint, exclusions, builder hash, and teacher protocol.  Any mismatch
aborts the whole cache; per-scene partial rows are rolled back and output is
written atomically.

Runner:

`radio_gs/scripts/run_surface_region_context_recovery_screen.sh`

Frozen query-free candidates:

| Candidate | input context | candidate pool | reliability |
|---|---:|---:|---|
| `control_c256_geometric` | 1.20 | 256 | geometric agreement |
| `context_c1024_geometric` | 1.20 | 1024 | geometric agreement |
| `context_c1024_uniform` | 1.20 | 1024 | uniform valid |
| `core_c1024_geometric` | 1.00 | 1024 | geometric agreement |

The runner uses 32 scene-disjoint training scenes, eight validation scenes,
four train/two validation shards, 12 regions per scene, and readout seeds
0/1/2.  A treatment is eligible only if its mean query-free selection score
improves by at least 0.001, it wins at least two seeds, and no validation
component drops by more than 0.002.  Benchmark queries remain closed until
this rule freezes one candidate.

The promotion finalizer no longer trusts cache/checkpoint reports as the
selection authority.  It independently reconstructs the 4/2 shard scene sets
and all 49 excluded physical spaces, requires exactly 12 regions per scene,
validates every student/teacher tensor and teacher-target digest, and
recomputes the three validation metrics from the bound checkpoint weights on
CPU.  Reported GPU metrics remain only a tolerance-checked audit witness.

For the next text-specific increment, the independent-cosine response loss is
now connected to a three-seed Surface readout trainer.  It matches exactly the
quantity consumed by the text scorer,
`SmoothL1(cos(student_descriptor,text), cos(teacher_descriptor,text))`, without
adding a custom descriptor residual or changing cache/inference schemas.  The
fit bank is the only text supervision opened during training; its response
weight is fixed from the first-batch gradient-norm ratio, and control/candidate
share initialization, row order, seeds, architecture, caches, and the original
query-free checkpoint-selection rule.  Existing benchmark-derived text banks
remain ineligible for training or model selection.

The promotion bank is therefore the locally installed timm ImageNet-1K
primary alias for each synset, normalized mechanically and de-duplicated by
stable first occurrence.  This produces 997 target-blind queries from 1000
classes.  A synset-only SHA256 bucket rule freezes 806 fit, 101 development,
and 90 audit queries; neither bank construction nor splitting reads LERF,
NVOS, ScanNet, or SPIn names.  The prompt template remains exactly `{query}`.

A separate 872-query list can be reproduced by excluding 125 synsets after
opening those benchmark vocabularies, followed by the same three duplicate
removals.  Its zero post-hoc overlap is useful only as an auxiliary leakage
sensitivity audit: it is explicitly ineligible for promotion training because
the exclusion itself used target knowledge.  The frozen promotion artifact
must bind the two timm source hashes, canonicalization and split contracts,
SigLIP2 model revision, vocabulary and tensor semantic hashes, serialized-file
hash, and `benchmark_vocabulary_opened=false`.

The target-blind vocabulary is now materialized as
`paper/artifacts/target_blind_imagenet1k_primary_text_bank_v1.json`, with its
source/split/builder contract in
`paper/artifacts/target_blind_imagenet1k_primary_text_bank_v1.manifest.json`.
Its canonical JSON SHA256 is
`2644c8454c12b0d6ca16fc453ee63e5289112172b82b61136e003ddf65a090ab`;
the fit/dev/audit counts are 806/101/90.  The CPU-only embedding builder now
rejects a self-consistent replacement vocabulary or same-named model snapshot:
it independently pins the canonical vocabulary/count/source/split hashes and
all official SigLIP2 snapshot file hashes.  The real three-split tensor bundle
is materialized under
`output/optimization_20260731/target_blind_siglip2_text_bank_v1`.  Independent
reopening verifies CPU float32 `[806,1536]`, `[101,1536]`, and `[90,1536]`
unit-normalized tensors.  The fit/dev/audit artifact SHA256 values are
`d67b632e8ccce13d84479379e8f674f5ec31b729acf02a79ce6c4bb2a4f170f4`,
`37c8d1f160b3ad69b5d6372c40dcc6207bca5fb9ef0143e139965e95e7beceb4`,
and `46dd338340a310e2b59997d1b6ea4882590c76f8aca389d4aa0abc2b3c5c2721`.

The response gate also fails closed on provenance rather than JSON agreement.
It reopens every descriptor, readout/report, validation cache, RADIO checkpoint,
authority bundle, and text-bank file; replays scene/region row identities; and
recomputes every response report.  Development selection and audit confirmation
are explicit phases over the same pre-registered eight ScanNet validation
scenes.  Audit cannot be opened by jointly rewriting a rejected development
manifest and completion file.  All three seeds must retain each original
Surface metric within 0.002 before any response improvement can advance.

### 2026-08-01 capacity and unary correction

The two completed paced control shards contain 192 regions from 16 scenes and
make an important naming distinction explicit: `context_c1024_*` raises the
*candidate* limit to 1024 but the published tensor still has
`maximum_tokens=256`.  It is therefore not yet a four-times-larger feature
field.  By radius, the control's mean core/context token counts are
`120.67/46.64`, `231.70/12.08`, and `252.63/1.22` at 0.25, 0.45, and 0.70 m.
At 0.70 m, 63/64 regions hit the 256-token cap and only 2/64 retain any context.
The corresponding fixed teacher core has mean 745.9 and maximum 1649 tokens;
8/64 regions already have at least 1024 core tokens.  A larger candidate list
alone cannot admit context for those regions.

The retrieval geometry is also much less tolerant than row-wise descriptor
cosine suggests.  The target-blind fit bank has 806 queries, while the mean
within-scene teacher support top-one margin is only 0.00379 (p05 0.000139).
In a deterministic tangent-error simulation, a descriptor cosine near 0.958
retains only 41.19% of region/query support top ones.  This explains why a
pointwise SmoothL1/cosine checkpoint criterion can report a strong descriptor
match while still producing a weak text unary.

Two backward-compatible repairs are therefore implemented for the next frozen
screen.  `core_context_separate_attention_v1` normalizes core and context
attention independently, preserves a core-only base, and introduces context
only through a residual conditioning stream; empty context is exactly a zero
stream.  The legacy `joint_attention_v1` remains the default, keeps the same
state-dict schema and architecture digest, and is bit-exact with the old
forward formula.  Separately, text-response epoch selection now uses
`within_scene_fit_query_support_top1_then_descriptor_relation_then_surface_v1`:
target-blind within-scene support top-one agreement is primary, descriptor
relation error is secondary, and the original Surface score is the final
non-inferiority witness.  A deterministic regression rejects the failure case
where a higher row cosine flips the teacher top one.

The next query-free ablation must first compare joint versus separate attention
on byte-matched caches/teachers and seeds 0/1/2.  Promotion still requires mean
selection gain at least 0.001, at least two seed wins, and no descriptor
component drop above 0.002.  Only after that isolated comparison passes should
the cache budget advance to a complete-core-then-context 4096-candidate screen;
the text development/audit banks remain closed until the winning Surface
configuration is frozen.

### 2026-08-01 formal attention and text-response result

The byte-matched post-cache attention continuation retained
`joint_attention_v1`; the isolated core/context-separate attention treatment
did not pass the frozen Surface gate.  The retained three-seed control has mean
query-free selection score `0.9279770393`, mean descriptor cosine
`0.9432072590`, and all-view descriptor cosine `0.9127468196`.

The first formal text-response treatment used the 806-query fit split and the
one-shot initial gradient-norm ratio `response_lambda=285.8855974008441`.
Within-fit-scene support top-one agreement increased from approximately
`0.1591` at random initialization to `0.2796/0.2855/0.2785` for seeds 0/1/2.
This establishes that independent response supervision is active, but the
frozen 101-query development gate rejects the resulting method and does not
open the 90-query audit split.

The development error metrics improve for all three seeds: the paired mean
control-minus-candidate improvement is `3.4107038e-05` for SmoothL1 and
`0.0020650960` for MAE, with scene-bootstrap 95% intervals
`[6.3876e-06,6.5764e-05]` and `[0.00056018,0.00364012]`.  Those pointwise gains
do not preserve retrieval geometry.  Candidate-minus-control mean Spearman is
`-0.04415345`, with a strictly negative 95% interval
`[-0.09948901,-0.00541049]`; mean top-decile overlap is `-0.02640264`.

Surface retention also fails and must not be relaxed.  The per-seed changes in
summary/mean-descriptor/all-view cosine are respectively
`(-0.044153,-0.010259,-0.010084)`,
`(-0.040600,-0.010215,-0.010013)`, and
`(-0.019645,-0.004629,-0.004485)`, versus the preregistered per-component floor
of `-0.002`.  The formal decision is therefore `reject_no_audit`, with
`main_result_eligible=false` and benchmark queries/masks still closed.

The causal correction for the next treatment is not a larger scalar response
weight.  The rejected trainer started again from the seed-matched random
initialization, optimized a pointwise response error, and selected epochs by
fit support before Surface feasibility.  The next candidate must warm-start
from the frozen seed-matched Surface readout, preserve its output field through
an explicit non-inferiority constraint, and distill scene-wise response shape
or ranking rather than only independent region/query values.

Authority artifacts:

- distill completion:
  `output/optimization_20260801/surface_text_response_distill_joint_c1024_gpu1only_src50c48dfab98e/text_response_distill.complete`;
- development decision:
  `output/optimization_20260801/surface_text_response_promotion_joint_c1024_consumer2e4563cebbd9/dev_decision.json`;
- paired text gate:
  `output/optimization_20260801/surface_text_response_promotion_joint_c1024_consumer2e4563cebbd9/dev/paired_gate.json`.

The guarded three-seed CUDA work used physical GPU1 only.  Its observed maxima
were 50 C, 146.94 W, 33% utilization, and 488 MiB, with no thermal pause, Xid,
PCIe fault, or unclean release.  Consequently the `65/75/70/78 C`
start/pause/resume/abort policy is not limiting this workload.  The earlier
self-termination was a host/container PID-namespace ownership mismatch, now
fixed by the fail-closed `exclusive-singleton-after-clear-v1` contract.  The
dominant wall-time cost is CPU artifact revalidation, not GPU protection.

### 2026-08-01 trust-region interpolation and warm-start correction

The rejected response checkpoint still contains a useful direction in the
same seed-matched readout weight space.  Each control/candidate pair shares the
exact architecture, ordered 24-tensor fp32 state schema, cache provenance, and
original initialization seed.  A frozen CPU-only diagnostic therefore tested
same-seed interpolation
`theta_alpha=(1-alpha) theta_control + alpha theta_response` on the fixed grid
`{0,0.1,0.25,0.5,0.75,1}`.  Held-out development responses were recorded only
as post-hoc diagnostics and were excluded from the selection view.

The selection rule uses only query-free Surface validation and the target-blind
806-query fit split.  Every seed must retain summary-token, mean-descriptor,
all-view-descriptor, and relation fidelity within `0.002` of its own alpha-zero
control; aggregate fit support top-one must strictly improve and aggregate fit
response SmoothL1 must not worsen.  The minimum positive feasible step is then
selected.  The formal feasible set is `{0,0.1}` and `alpha=0.1` is the unique
positive feasible value.  Its three-seed mean Surface score changes from
`0.9279770594` to `0.9283155972`, fit support top-one from `0.2638544242` to
`0.2647849619`, and fit response SmoothL1 from `1.2204091e-4` to
`1.1587676e-4`.  `alpha=0.25` already violates the per-seed summary floor for
seeds 0 and 1.  The frozen selection artifact is
`surface_readout_weight_interpolation_selection_alpha01_joint_c1024_src50c48dfab98e.json`
with SHA256
`428b92d18dab62a5747ea0602fb7ce36251430f712ce9e5346c18d9f2aa9dbf8`;
it contains no development values and records the audit split as unopened.

An independent query-free replay reopened the exact two validation caches and
three control/candidate endpoint pairs, reproduced all selected Surface and fit
metrics for 96 regions in the eight preregistered scenes, and reconfirmed all
Surface retention checks.  Independent review found and fixed a concurrent
opening race in the first one-shot executor draft: the final opening receipt
uses an atomic no-clobber claim, a forced two-executor race test admits exactly
one loader, and opening/finalization rehash a fixed 14-source implementation
closure.

The formal audit attempt nevertheless failed closed after its one allowed
opening.  Receipt
`surface_readout_weight_interpolation_audit90_alpha01_joint_c1024/audit_opening_receipt.json`
was committed with SHA256
`2cfaa5e41527fae0e6a8d8ee61159e3554b003541be25ec19dff9d45dccf28b4`,
and the process loaded the audit bank and computed its in-memory gate.  Before
the terminal could be written, the final receipt provenance check rejected the
lexical `/root/RADIO-GS/output/...` path because `output` is a symlink to the
canonical `/mnt/pool/...` root.  No confirmation artifact or recoverable metric
report was published.  The audit split must not be reopened merely to recover
those lost in-memory metrics.  This interpolation family therefore has no
confirmation decision, is ineligible for benchmark opening, forbids post-audit
retuning or selection of another alpha, and is closed as
`confirmation_failed_family_closed_no_retry`.  The immutable no-audit failure
terminal is
`surface_readout_weight_interpolation_audit90_alpha01_joint_c1024/audit90_failure_terminal.json`
with SHA256
`cc32d71175d9f0f7fdc3a9f646c92d8a316c7debe882e981af7d26546c770bb2`.

The alternative training correction is also now structurally implemented:
each treatment seed warm-starts from its externally SHA-bound Surface control,
epoch zero is the immutable fallback, and an epoch is selectable only after
all Surface components pass the same `0.002` non-inferiority floor.  A new
scene-wise objective separates centered query-response profile preservation
from listwise within-scene ranking.  A protected seed-zero gradient diagnostic
at the real Surface warm start measured gradient L2 norms `0.1050601` for the
Surface objective, `0.00163209` for independent response SmoothL1, and
`1.1603833` for the combined scene profile/ranking loss.  The corresponding
equal-Surface weights are `64.3715` and `0.0905391`; the old shared random-start
weight `285.8856` would apply about 4.44 times the full Surface gradient at the
actual warm start.  Formal retraining remains blocked until calibration is
made seed-specific at the warm-start point with an explicit response-gradient
budget rather than reusing that random-start scalar.

GPU1 telemetry quantifies the efficiency conclusion.  The three formal seed
guards ran for 343 seconds in total, peaked at 44/47/50 C and
140.73/146.94/142.15 W, and produced zero start wait, pause, resume, cooldown,
or thermal-abort event; no sample reached the 65 C start threshold.  The fixed
guard overhead is approximately 10 seconds, below 0.7% of the 1403.2-second
end-to-end wall time.  Approximately 95.1% is CPU loading, hashing, artifact
audit, receipt, and finalization.  Raising temperature limits would therefore
provide no throughput gain.  The active physical-GPU1-only policy remains
`65 C` start, `75 C` pause, `70 C` resume, `78 C` hard stop, three-second poll,
and no peer-GPU query; optimization should instead cache immutable validation
receipts, share hashes/loads, and move pure CPU validation outside guarded CUDA
windows.

## Registered-region unary and readout correction

The independent LUDVIG-SAM reproduction now closes a useful reference check on
two scenes with three seeds each.  Under released-all-view geometry, NVOS Fern
is `84.5073%` versus the paper's `85.5%`, while Leaves is `96.6494%` versus
`96.4%`; their equal-scene macro is `90.5783%`, `-0.3717` points from the
same-scene paper reference.  On SPIn-NeRF the same Fern/Leaves macro is
`96.7066%`, `+0.0566` points from the same-scene paper reference.  These are
per-scene protocol validations, not full-cohort or strict-unseen claims, but
they show that the large current-method gap is not explained by a broken local
reference implementation.

The frozen NVOS eight-scene chain is:

| Stage | macro foreground IoU |
|---|---:|
| unary prior | 0.74013043 |
| propagated random-walker posterior | 0.74902901 |
| seeded connected component | 0.72640440 |

The graph adds `0.00889858`; the later hard component filter removes
`0.02262461`.  Orchids alone falls from `0.70009114` to `0.48204156`.
`SEEDED_COMPONENT` is not another inference step: it thresholds primitives at
0.5, rebuilds an active graph with an absolute affinity cutoff, and deletes
every active island not touched by a reference seed.  Occlusion and graph
fragmentation therefore turn a small probability perturbation into a large,
non-continuous recall loss.  Neither the NVOS scribble protocol nor a full
binary region mask asserts that the target is one connected 3-D component.

The new `registered-region-v1` candidate makes three method-level corrections:

1. Solver/prototype masses are derived jointly as
   `relu(s)` and `relu(-s)` on one shared observation scale.  Equal
   positive/negative raster evidence is neutral instead of being assigned to
   the positive side, and a weak tail of either sign is never independently
   max-normalized into a hard seed.
2. Direct registered evidence is fused in Bernoulli space:
   `p=(1-c)*sigmoid(u_field/T)+c*q`, where
   `m_pos,m_neg` are raw alpha-compositing adjoint masses,
   `c=1-exp(-(m_pos+m_neg))`, `q=m_pos/(m_pos+m_neg)`, and
   `s=c*(2q-1)`.  One alpha-weighted native-raster pixel is the fixed
   observation unit.  This is dimensionally compatible with the solver: a
   tiny raster tail has tiny confidence, an unobserved row leaves the field
   bit-exact, and strong pure evidence can override an adversarial feature
   margin without a tuned additive weight.  The observation is constructed
   over all Gaussians before capability-valid rows are selected.
3. A `REGION` prediction uses the continuous propagated posterior.  Connected
   selection remains an explicitly reported diagnostic, never the main
   output.  Scalar scores are rendered first and thresholded once in pixel
   space.

The legacy winner-take-all/additive paths remain the defaults, so historical
frozen commands keep their numerical semantics.  The new behavior is explicit
through `--registered-seed-construction joint_signed`,
`--registered-observation-fusion probability_mixture`,
`--registered-observation-confidence poisson_mass`,
`--registered-selection-mode seeded_component`, and
`--registered-readout-stage propagated`.

The audit also found and fixed a stage-reporting defect.  When the final stage
was `propagated` or `unary_prior`, the evaluator reused that final render under
the name `connected`.  Thus the 2026-07-31 propagated-final JSON files have
valid unary/final values but their connected columns are not independent
measurements.  New runs render every stage from its own primitive tensor.

Every new report now carries separate dataset, evaluation, and method
contracts and their canonical digests.  They bind registration, raw
observation semantics, seed construction, fusion, solver, renderer, RADIO
checkpoint, threshold/resize semantics, and every implementation source that
affects the result.  Aggregation recomputes all three digests, checks their
cross-links, requires a propagated result eligible for the frozen diagnostic,
and rejects mixed contracts.  It explicitly preserves
`main_result_eligible=false` until the disjoint promotion gate passes.  The
report also binds the run-manifest method declaration and eligibility, hashes
every final/stage score array, requires byte-identical final and propagated
arrays, and checks their per-frame metrics.  The legacy `d21...` hash remains
only as a compatibility/source identifier because it describes a cosine
margin at threshold 0 rather than this posterior at threshold 0.5.  The
immutable run manifest also hashes every canonical field, capability cache,
graph, renderer checkpoint, camera mapping, and resolved pose input.

Candidate declaration:

`paper/artifacts/nvos_registered_region_v1_candidate_20260731.yaml`

GPU1 runner:

`radio_gs/scripts/run_nvos_registered_region_v1_queue.sh`

The runner uses exact native prompt registration, observable native-prompt
scalar resolution, total-alpha (`coverage_power=1`) rendering, fixed 0.5
threshold, and no target calibration or per-scene stage switch.  It is
explicitly diagnostic until a disjoint registered-prompt gate confirms the
frozen formulation.

The first three frozen NVOS scenes reject v1 decisively.  A second diagnostic,
`registered-region-v2`, additionally conditions Poisson confidence on the
labeled fraction of each Gaussian's visible footprint,
`c=(1-exp(-m))*((m/visible_mass)^1)`.  This is equal to v1 for a full mask and
prevents a few scribble pixels from treating a broad-footprint Gaussian as a
fully observed unary.  It improves Flower and Fortress but worsens Fern, so it
also fails the predeclared three-scene stop rule:

| Three-scene propagated IoU | Fern | Flower | Fortress | macro |
|---|---:|---:|---:|---:|
| historical frozen | 0.764200 | 0.850600 | 0.928600 | 0.847800 |
| registered-region-v1 | 0.460618 | 0.743765 | 0.568632 | 0.591005 |
| registered-region-v2 | 0.448010 | 0.788665 | 0.696753 | 0.644476 |

V2 recovers `+0.053471` macro over v1 but remains `-0.203324` below the
historical path.  Full-eight execution is therefore rejected.  The evidence
rules out the original single-cause explanation: footprint coverage matters,
but the historical advantage is primarily upstream in the unary feature
channel/score calibration, not in graph propagation.  On all three v2 scenes,
propagation lowers unary IoU and connected selection lowers it again.  Further
target-set tuning of coverage powers or thresholds is not eligible; recovery
must come from query-free feature-capacity screens or a disjoint
registered-prompt development gate.

`registered-region-v3` freezes one remaining semantic correction without a
new fitted constant.  A native adjoint observation that already passes the
solver's inherited absolute signed hard-seed threshold (`0.20`) is promoted to
unit fusion confidence in the unary itself; weaker raster tails keep the v2
Poisson-by-labeled-coverage mixture, zero observations preserve the field
exactly, and positive/negative evidence is sign-symmetric.  This directly
tests the stated failure mode that accepted scribble/full-mask evidence was
still diluted before propagation.  The contract is frozen in
`paper/artifacts/nvos_registered_region_v3_candidate_20260731.yaml`.  The
GPU queue evaluated only Fern, Flower, and Fortress first and invoked
`radio_gs/scripts/screen_nvos_registered_region_v3_continuation.py`.  The gate
was configured to run the remaining five scenes only if the three-scene macro
was strictly above v2, at least two scenes improved, and no scene regressed;
rejection writes a hash-bound partial completion instead of a misleading full
aggregate.

The formal GPU1-only run completed all eight scenes with eight independently
validated scene receipts.  Its strict aggregate is:

| Eight-scene macro | IoU |
|---|---:|
| foreground/final | 0.6276568528 |
| unary | 0.6275825302 |
| propagated | 0.6276568528 |
| connected | 0.6067071316 |

Thus graph propagation adds only `+0.0000743226` over the unary, whereas
connected selection has delta `-0.0209497211` from the propagated result.  The
hard-seed unary correction is theoretically consistent and positive on the
frozen continuation gate, but its magnitude is far too small to explain or
recover the historical feature-field gap.  The first three scenes all improve
strictly over v2: Fern by `+0.0000960905`, Flower by `+0.0000513162`, and
Fortress by `+0.0001518982`; their macro changes from `0.6444761519` to
`0.6445759202` (`+0.0000997683`).  This satisfied the predeclared continuation
rule and opened the remaining five scenes without starting them before the
decision.

This result is a frozen diagnostic only: `frozen_diagnostic_eligible=true`,
but `main_result_eligible=false` and the candidate remains
`diagnostic_until_disjoint_registered_prompt_gate`.  The authoritative
aggregate is
`output/optimization_20260801/nvos_registered_region_v3_gpu1only_v3_srca982f019_rt22f4a8e7/summary.json`,
with the gate decision in `three_scene_screen.json` beside it.  The run used
physical GPU1 exclusively.  Its 264 telemetry rows recorded maxima of 44 C,
170.43 W, and 4792 MiB, with zero thermal pause or hardware/telemetry fault;
all eight scene releases were independently verified.  The data therefore do
not support a thermal-policy bottleneck for this NVOS workload.

The strict aggregator now closes every metric path, not just the final and
propagated pair.  Unary, propagated, connected, and final frame IDs and unit
metrics are validated; every aggregate must equal its per-frame mean; final
and propagated remain equal per frame and by score-array SHA.  Joint JSON
tampering of final/propagated aggregates and isolated diagnostic-stage
tampering therefore fail closed.

## Canonical capacity screen

Runner:

`radio_gs/scripts/run_canonical_v5_capacity_screen.sh`

Frozen query-free rungs:

| Name | coarse dim | hidden dim | residual blocks |
|---|---:|---:|---:|
| V5-R reliability control | 0 | 192 | 0 |
| V5-W width control | 0 | 512 | 0 |
| V5-S spatial capacity | 64 | 512 | 0 |
| V5-D nonlinear depth | 64 | 512 | 2 |

All candidates share one normalized mean-resultant raw MPR and exact
responsibility reuse for native official DINO/SAM MPRs.  Selection uses four
held-out label-free frames and requires raw/DINO/SAM mean drop at most `0.005`,
p05 drop at most `0.01`, unsupported visible fraction at most `0.005`, and
official relation gain at least `0.005`.  Visible support is rendered from the
field's MPR-valid primitive rows; a missing support statistic now fails closed.
If every rung misses the visible-support gate, selection returns
`support_gate_failed_no_promotion` with no selected variant instead of calling
`max()` on an empty candidate set.
The width-only rung makes `W→S` an isolated spatial-capacity comparison and
`S→D` an isolated depth comparison.

Primary development input:

- scene: `scene0011_01`
- overlay: `paper/artifacts/canonical_v5_scene0011_01_overlay_20260731.yaml`
- held-out frames: `336,545,1132,2516`
- official maps: 480 aligned raw/DINO/SAM frames
- benchmark labels, masks, and text queries opened: false

The older 480-frame extraction manifest records the C-RADIO version but not
the checkpoint SHA that produced its adaptor maps.  It is therefore retained
only as a legacy input audit and is not accepted for v5 promotion.  Before the
next screen, the runner will perform a fresh extraction into
`output/optimization_20260731/v5_verified_features/scene0011_01`, loads an
explicit checkpoint, and records its SHA256 in the frame manifest before any
MPR is built.

The v5 output root has an immutable run manifest binding the resolved config,
geometry and RADIO checkpoint hashes, held-out frame IDs, native capability
manifest scene/image/frame identity, training contract, seed, epochs, runner
hash, and every implementation source that affects extraction, MPR, training,
rendering, or selection.  Field checkpoints record raw-MPR SHA256 and the full
actual training configuration.  Audits record alpha, support, and boundary
thresholds and require residual mode `none`.  Resume aborts on any mismatch
rather than silently mixing stale stages.

Official extracted maps are normalized in float32 per pixel but retained in
their native float16 precision in CPU memory.  Registered means are finalized
by row chunks after releasing teacher maps.  For the 120-view, 4096-D DINO
scene0011 screen at approximately 300k Gaussians, the known overlapping CPU
allocation estimate is now 11.55 GiB (teacher 4.39, accumulator 4.58, output
2.29) versus 27.4 GiB currently available.  The builder performs a live
preflight and fails before loading maps if the bound is unsafe.

A result from one scene is provisional.  The identical frozen screen must
confirm the selected rung on `scene0046_00` before promotion.  Its frozen
overlay is
`paper/artifacts/canonical_v5_scene0046_00_overlay_20260731.yaml`, with
held-out frames `561,1111,1494,2100`.  The CPU-only confirmer
`radio_gs/scripts/confirm_canonical_v5_cross_scene.py` promotes only when two
distinct scenes select the same non-control candidate under byte-matched
training, audit, selection, runner, and implementation contracts.

## Query-first scalar compositing

The old HCD screen refiner cannot be restored after feature splatting without
breaking point/render parity.  The consistent alternative computes one scalar
query score per primitive and changes only its front-to-back pixel mixture:
`S_q(p)=sum_i a_i(p)s_q(i)`.  Primitive `s_q(i)` is exactly the direct-point
score; `a_i(p)` depends only on geometry, opacity, depth order, and the frozen
compositor, never on a benchmark query.

Existing query-free diagnostics show why this needs a cross-scene gate.
Gamma-two improves Ramen raw cosine from `0.737653` to `0.741180` and SAM
boundary retention from `0.037447` to `0.040350`, but reverses on Figurines:
raw mean `0.659581→0.651737` and p05 `0.409146→0.361165`.  Top-one/top-two and
depth-band variants lose still more dense fidelity, while expected-depth leaves
about `10.59%` of visible pixels unsupported.  These benchmark-scene
diagnostics can exclude unsafe families but are ineligible for selection.

The independent CPU selector
`radio_gs/scripts/select_query_free_scalar_compositor.py` therefore accepts
only alpha-mean, gamma `1.25/1.5/2`, and top-four audits from at least two
explicitly non-benchmark development scenes.  Raw/DINO/SAM mean, p05, and
support are hard per-scene guards; DINO and SAM affinity Pearson and boundary
retention must each improve without any head or scene being averaged away.
Until a V5 field is frozen and produces those new audits, alpha-mean remains
the baseline and no scalar compositor is promoted.

## Current infrastructure status

The peer-coupled canary chronology below is retained as a failure record.  It
is superseded for current work by the physical-GPU1-only policy quantified in
the trust-region section above; current runners do not query, gate on, or
control GPU0.

The repository Python wrapper now selects the host `libcuda` whose version
matches the loaded NVIDIA kernel module, avoiding CUDA error 804 caused by the
newer compatibility library.

`/mnt/pool` is healthy and writable.  Physical GPU1 (`0000:82:00.0`) now has a
valid PCI configuration prefix, is queryable, and has no blocked UVM teardown
process.  Both boards report a 300 W limit, but the container lacks permission
to lower GPU1's power limit or lock its clocks.  A thermal guard therefore
polls GPU1 PCI/temperature/power every second and terminates only its own
process group on PCI loss, telemetry loss, or temperature breach.  It also
observes the externally occupied GPU0 as a chassis heat source without ever
controlling it.  The original activity-based peer gate proved too restrictive:
it serialized GPU1 behind an unrelated GPU0 queue even when both boards had
thermal margin.  The active policy therefore gates only on chassis heat,
pausing the guarded GPU1 process group if GPU0 reaches 77 C and resuming after
GPU0 cools to 75 C.  GPU1 starts at or below 52 C and hard-stops at 75 C.
An attempted GPU1 soft pause was rejected because a stopped CUDA context
remained in P2 at roughly 130--140 W; explicit synchronized pacing between
RADIO encodes cools the board more reliably without changing any tensor.

The 2026-08-01 restart quantified which parts of this policy were conservative.
Fixed eight-second pacing consumed approximately 95--98% of each completed
control shard's wall time, so it is not retained as an unquestioned default.
Full-shard canaries were attempted at two, three, and four seconds per image
under a concurrently running GPU0 LUDVIG job.  The p2 run produced 695 telemetry
samples, reached 75 C, and was terminated by the hard guard.  The p3 and p4
runs each crossed the 71 C long-run contract at 72 C and were stopped early.
Adding a 285 W peer-power interlock did not solve the chassis coupling: the
next p4 run reached 73 C.  Even after requiring a 120-second quiet launch
window, GPU0 later returned to near-300 W and the p4 run reached 72 C.
All five runs released GPU1 cleanly and showed no Xid, PCIe, telemetry, or NFS
fault; none produced a cache terminal or canary authority report.

The telemetry also shows why longer `SIGSTOP` intervals are not an efficient
runtime cure.  During peer cooldown, the live CUDA context remained in P2 and
typically drew about 115--143 W, so GPU1's thermal baseline stayed high beside
the 300 W peer.  The active canary instead requires GPU1 at or below 52 C and
GPU0 at or below 75 C and 200 W for 60 continuous seconds; it does not require
GPU0 to have no compute owner.  Memory and utilization gates are disabled for
this launch, so an allocated but thermally idle peer does not serialize GPU1.
If GPU0 rises above the power/temperature envelope after launch, the guard
terminates its own GPU1 process group rather than leaving a stopped CUDA
context resident.  Exit 87 is retryable only after independent PCI, UUID,
compute-owner, kernel-journal, telemetry-interval, and runtime-closure checks
prove a clean CUDA release.

The Surface builder is now per-scene durable rather than merely planned.  Each
scene publishes an immutable tensor partial followed by an external SHA-bound
terminal; the resume contract binds the complete CLI, selected RGB-D/pose
inputs, checkpoint, implementation/runtime closure, teacher replay, and Python
RNG predecessor/successor state.  A peer interruption therefore resumes only
hash-validated complete scenes, while Xid, PCIe, telemetry, artifact, closure,
or contract failures remain non-retryable.  The main file lock is reinforced by
an abstract AF_UNIX kernel singleton shared by Surface and text/NVOS runners,
so unlinking and recreating the lock pathname cannot permit two physical-GPU1
owners.

An unpaced Surface cache build reached 77 C, and progressively paced trials
showed that one, four, and six seconds per RADIO image still left too little
headroom in this chassis under the then-active peer workload.  Each trial was
stopped cleanly before driver loss, kept in a separate output root, and never
mixed with an authoritative result.  The completed conservative reference run
used adaptor batch 64, eight seconds per RADIO image, a one-second telemetry
interval, and the 75 C hard limit.  Its output root is
`output/optimization_20260731/surface_fixed_teacher_replay_v2_gpu1_p8_hard75`;
its manifest SHA256 is
`77d162b286355c5ce1d369790f299c819397bfbd67302efdde4f72ae63409c2a`.
Observed utilization is intentionally bursty rather than continuously high.
An independent 12-minute audit captured an 83% utilization sample and measured
GPU1 temperature at 52/63.69/71 C (minimum/mean/maximum) and power at 131/196 W
(mean/maximum).  The 71 C peak occurred once near startup; the following ten
minutes did not exceed 70 C and showed no upward drift, thermal abort, PCI
error, Xid, or storage error.  Peer thermal interlocks all resumed normally.
The first train shard completed atomically after 40 minutes 49 seconds and the
second train shard also completed.  Shard 0 cache/sidecar SHA256 values are
`02cfa45af46cf8274c17ccde28e8953fb4b659527652525340703a241cad22bc` and
`37f19d5778b59e9efb32af97d0c9eeec3daf7b2ebf9f88c9886886a9aaf33e0c`.
Shard 1 values are
`dfd2c0856a612c1d58ac83ccbcd21b775b04a82c45d7d1569d70d4a493d4460` and
`4b3dffb2c270d8780df8ba0d6ece3769d02c0f50d7ffcc0623845c5e0aed7d2`.
Independent CPU reopening verifies sixteen scenes, 192 regions, no failed
scene, finite tensors, and valid masks and anchors.  The full four-candidate
cache and twelve-readout screen remains estimated at 10--14 hours at p8; the
current durable canary tests p4 only during the 60-second heat-qualified
window, with a 71 C promotion ceiling and 75 C hard stop.  A content audit of
shard 0 also confirms the intended capacity pressure: 67/96 control
regions hit the 256-token budget, mean core/context counts are 208.99/21.09,
only 42/96 regions contain any context, and only 10/96 contain at least 64
context tokens.  The fixed teacher support averages 434.65 core tokens and has
zero 4096-candidate saturation.  The candidate-1024 treatment therefore tests
a real missing-context mechanism rather than width alone.  This pressure is
scale-selective: mean context counts are 50.94, 10.41, and 1.94 at radii 0.25,
0.45, and 0.70 m; all 32 large-radius regions hit the 256-token budget while
their fixed teacher core averages 819.38 tokens.  Multi-view target
ambiguity is also material: within-region official summary-token pairwise
cosine is 0.8127 on average but 0.5852 at p05 (minimum 0.4398); descriptor
pairwise cosine is 0.9000 on average and 0.7531 at p05.  This supports response
distillation as a retrieval-geometry repair after the query-free context
screen, while remaining only a one-shard diagnostic until the full screen
completes.

A follow-up implementation should cache the exact fused scene
intermediate after the control candidate: the three replay candidates repeat
960 RADIO image-equivalents, including at least 7,680 seconds of fixed pacing,
even though only their context/reliability construction differs.  A standalone
scene-intermediate v1 contract is now implemented and independently reviewed:
it externally binds the intended contract and file SHA, uses one trusted file
descriptor for hash/load/rehash, publishes with atomic no-clobber semantics,
revalidates every source/checkpoint/implementation file, and enforces the exact
active symmetric support-graph invariants.  Its CPU suite passes 34 tests with
no remaining P0/P1.  It is intentionally not wired into this already-running,
hash-bound screen; integration belongs to a new output root.

The v5 runner remains queued behind the Surface query-free screen and must use
a new output root and frozen scene0011 inputs.  Its 480-frame feature extractor
now provides per-frame atomic commits, a strict batch-one resumable contract,
fixed incomplete staging with a process lock, and post-synchronization
eight-second pacing.  The extraction/manifest regression set passes 45 tests.
The later continuous field-training phases still require their own thermal
characterization before the complete v5 runner can be called launch-ready.

## Verification

The combined post-patch NVOS/query/support/reliability, strict aggregation,
Surface cache/contract/summary/readout/finalizer, target-blind artifact, and
text-response suite completes with `179 passed, 7 warnings`.
The current sources also pass Python byte-compilation, shell parsing,
embedded-runner AST parsing, exact runner/evaluator method-contract comparison,
and `git diff --check`.
