# Structured Universal Gaussian Memory v3: isolation and first instance upper bound

## Scope and contracts

The historical `codex/query-native-memory-v2` state at `0c59e80` is frozen by
the local tag `archive/query-native-v2-0c59e80`. Development moved to
`codex/structured-universal-memory-v3`. ADR 0004 defines SUGM-v3 as a new
candidate rather than an extension of a historical benchmark readout.

The new `radio_gs/v3/` namespace now contains fail-closed D512+R5 scene-state
validation, static legacy-import and scene-token checks, global visual,
scale-conditioned instance, and boundary projections, ternary exact-MPR mask
evidence, one shared Gaussian membership posterior, differentiable exact-hit
rendering, source-heldout metrics, and the temporary non-deployable D16 oracle.
No v3 core module imports a historical query or benchmark script.

## Real source authority audit

The first source-only cohort is the existing 32-view multiscale official-SAM3
authority for one LERF mapping scene. It contains 277 mask episodes, of which
252 have nonempty exact-MPR support. The source raster closure contains
8,626,604 unique-view exact compositor hits. All inputs are hash-bound and
declare that benchmark RGB, masks, and metrics were not opened.

The fixed split is now:

- source train: view index modulo four in `{1,2}`;
- source dev: residue `3`;
- source audit: residue `0`.

The audit split remained sealed in the v2 development execution.

## Executions

An initial execution used train residues `{1,2,3}` and audit residue `0`. It is
retained only as a diagnostic because it did not separate source dev from
source audit. It also exposed an invalid boundary-logit parameterization. Both
arms reconstructed known pixels but increased unknown false-positive mass;
neither was eligible for promotion.

Two subsequent dev-only executions exposed an authority/evaluator defect:
all known evaluation pixels were positive because the supplied relation
artifact contained no same-view different edges. Those results are marked
evaluator-invalid. The valid authority is now constructed directly within
each original source SAM hierarchy: masks must be area-comparable and
pixel-disjoint to become explicit negatives; ordinary mask exterior remains
unknown. This changes the dev known-positive fraction from 1.0 to 0.6322 and
provides 66 same-view different pairs. Ten positive-only episodes are excluded
from proper evaluation but remain legal positive training evidence.

The final v4 dev execution used 300 steps and four mask episodes per step for
both arms. Unknown pixels were excluded from the negative loss and received
only a one-sided restraint against growth above the neutral prior. Boundary
magnitude was converted to a signed logit before proper scoring. Cross-view
same-object and same-view mutually-exclusive different-object prototypes used
a balanced proper log score; unknown relations were excluded.

| arm | state | dev IoU | dev Brier | boundary F | unknown FP |
|---|---|---:|---:|---:|---:|
| D16 oracle baseline | untrained | 0.29033 | 0.39698 | 0.09742 | 0.71470 |
| D16 oracle candidate | trained | 0.29907 | 0.27199 | 0.15533 | 0.53393 |
| internal pre-fusion L512 baseline | untrained projection | 0.45591 | 0.21619 | 0.09001 | 0.54663 |
| internal pre-fusion L512 candidate | trained projection | 0.45068 | 0.19093 | 0.16524 | 0.53337 |

The immutable reports are:

- D16: `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260826/sugm_v3/oracle16_dev_v4_valid_authority/upper_bound.pt.json`
- pre-fusion diagnostic (invalid as Phase 1): `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260826/sugm_v3/frozen512_dev_v4_valid_authority/upper_bound.pt.json`

## Decision

The valid D16 oracle improves mask IoU by only `+0.00874`, below the required
`+0.05`, although Brier, boundary F, and unknown false-positive mass improve.
Its trained absolute IoU (`0.29907`) is also below frozen D512 (`0.45068`). The
D512 arm slightly regresses IoU by `-0.00523`. Therefore the mandatory D16
upper-bound gate fails and stopping rule 1 is active. Neither arm is promoted,
source audit remains unopened, and no benchmark execution is authorized.

The initial 300-step result triggered an authority/MPR/geometry audit before
any latent work continued. An unthresholded exact-MPR oracle produced only
`0.51577` same-view round-trip IoU and `0.48795` known-same cross-view union
IoU. A target-driven, diagnostic-only visible-Gaussian posterior reached
`0.99540` same-view IoU over 39 proper dev masks (37 were approximately 1.0).
Frozen geometry can therefore express the masks; one-pass exact-MPR is an
initial transport rather than a sufficient membership posterior.

The earlier D16 rejection was also under-budget: it used 300 steps while the
preregistered upper-bound runner default is 2000. With all authority and loss
settings frozen, a source-dev budget curve gave:

| D16 steps | IoU | IoU gain | Brier change | boundary F change | unknown FP change |
|---:|---:|---:|---:|---:|---:|
| 600 | 0.35074 | +0.06041 | -0.14359 | +0.05091 | -0.20208 |
| 1200 | 0.40330 | +0.11297 | -0.15766 | +0.07452 | -0.19434 |
| 2000 | 0.41535 | +0.12502 | -0.16138 | +0.10215 | -0.19873 |

The faithful D16 instance upper bound therefore passes every numeric dev
criterion on `figurines`. It does not yet satisfy the multi-scene clause of
Gate 1; `ramen` is the second currently available scene with a complete
membership-plus-ternary-relation authority pair and must also pass. The D16
arm remains explicitly non-deployable and adds no authority for benchmark
execution.

A representation audit then found that all earlier nominal D512 executions
read the checkpoint's internal `local_codes`. This checkpoint has
`use_fusion=true`, so those tensors are pre-fusion L512 and are invalid as the
Phase 1 canonical-memory arm. The corrected runner materializes only the
public `query_memory(representation="coefficients")` post-fusion D512 in
chunks, records that representation in the receipt, and static audit now
rejects direct `local_codes` access anywhere in v3 core. Corrected matched
2000-step runs for `figurines` and `ramen`, plus the `ramen` D16 run, were then
launched under the same source-only split. Low-rank D512 writeback remains
conditional on per-scene D16 success and an insufficient corrected heads-only
arm. Source audit remains unopened.

The completed corrected results are:

| scene | arm | baseline IoU | candidate IoU | IoU gain | Brier change | boundary F change | unknown FP change |
|---|---|---:|---:|---:|---:|---:|---:|
| figurines | D16 oracle | 0.29033 | 0.41535 | +0.12502 | -0.16138 | +0.10215 | -0.19873 |
| ramen | D16 oracle | 0.34091 | 0.60037 | +0.25946 | -0.20154 | +0.39927 | -0.24551 |
| figurines | post-fusion D512 heads only | 0.45762 | 0.45522 | -0.00240 | -0.01747 | +0.11396 | -0.05472 |
| ramen | post-fusion D512 heads only | 0.51170 | 0.54984 | +0.03813 | -0.01998 | +0.11019 | -0.05920 |

Both complete-authority scenes therefore pass the D16 per-scene criterion.
The canonical D512 heads-only arm has only `+0.01787` scene-macro IoU gain and
fails the `+0.05` gate (figurines also regresses). Phase 2 is authorized on
both scenes. Its rank-16 optimization parameterization is discarded
at serialization and folded into exactly one D512 per Gaussian; the artifact
also records whole-field RADIO reconstruction cosine and PCGrad conflict rate.
The two-step smoke artifact contains only the folded D512 plus global head and
has RADIO mean/p05 cosine `1.0/0.99999994`. Matched 2000-step writeback runs
for both scenes are in progress; no audit or benchmark authority has been
opened.

The final shared-projection implementation uses the exact identity
`P(z+UB) = Pz + U(BP^T)` and reuses that projection across the four mask
episodes and relation loss in each step. This removes repeated N-by-512
autograd graphs without changing the folded field or loss. The matched Phase
2 results are:

| scene | IoU gain | Brier change | boundary F change | unknown FP change | RADIO mean / p05 / min cosine | PCGrad conflict rate |
|---|---:|---:|---:|---:|---:|---:|
| figurines | +0.01132 | -0.04291 | +0.14274 | -0.12708 | 0.99969 / 0.99831 / 0.96875 | 0.7020 |
| ramen | +0.09009 | -0.04196 | +0.20217 | -0.11062 | 0.99921 / 0.99712 / 0.98071 | 0.6575 |

The exact pre-registered source-heldout gate returns `passed=True`: scene-macro
IoU gain is `+0.0507046`, neither scene regresses, Brier decreases, boundary F
increases, and unknown false-positive mass decreases in both scenes. The
margin over `+0.05` is narrow, so this is sufficient to continue to the
capability no-regression check, not sufficient to open source audit or a
benchmark. Serialized artifacts contain only one folded D512 per Gaussian and
the global projection/scale head; training-time low-rank codes and basis are
absent.

Whole-field coefficient preservation was not treated as a substitute for a
capability gate. The folded candidates were rendered through the immutable
exact compositor hits on the same source-dev residue and compared with the
hash-bound cached RADIO backbone teacher. Both scenes fail strict
no-regression:

| scene | base RADIO render cosine | candidate | delta | candidate/base render cosine |
|---|---:|---:|---:|---:|
| figurines | 0.668494 | 0.667946 | -0.000548 | 0.999583 |
| ramen | 0.779581 | 0.778969 | -0.000612 | 0.999508 |

All eight dev views regress slightly in each scene. Under the preregistered
decision rule this is “C improves instance but semantics regress”; the
candidate is not promoted. A single controlled retry uses the fixed Phase 3
order: a source-train RADIO render-fidelity step on sampled exact-hit pixels,
then the existing instance/boundary step with PCGrad and the base-field
anchor. Dev teachers remain validation-only. A 600-step two-scene screen is
run before any matched 2000-step retry.

The controlled alternating screen produced:

| scene | 600-step IoU gain | Brier change | boundary F change | unknown FP change | RADIO dev delta |
|---|---:|---:|---:|---:|---:|
| figurines | +0.04068 | -0.04502 | +0.12665 | -0.11247 | -0.000280 |
| ramen | +0.04476 | -0.01567 | +0.08767 | -0.05009 | -0.000267 |

Alternation greatly improves the short-budget instance trend compared with
the non-alternating figurines 600-step result (`+0.00203`) and roughly halves
the RADIO regressions. It nevertheless fails both continue conditions:
scene-macro IoU gain is only `+0.04272`, and strict source-dev RADIO
no-regression still fails in both scenes. The retry is therefore stopped at
600 steps. No visual-weight search, threshold relaxation, 2000-step extension,
source audit, sentinel, or benchmark run is authorized. The evidence supports
the decision-table outcome that retrofitting instance state into this
historically optimized shared D512 has a capability conflict; the next method
work must organize the shared D512 jointly during source mapping rather than
continue tuning a residual on this field.

## Joint shared-D512 source-mapping screen

The next arm opens the sole post-fusion D512 itself during the alternating
source-mapping objective. It is initialized from the hash-bound public
canonical coefficients, but its deployable state is exactly one trainable
`N x 512` latent plus the global projection and scale head. The historical
pre-fusion codes, low-rank residual codes, sidecars, and extra per-Gaussian
state are absent. The frozen RADIO affine decoder supplies the source visual
teacher loss and no-regression measurement; instance, boundary, and ternary
relation losses use the same projected D512.

Opening the full table exposed two implementation-only peak-memory costs.
They were removed without changing the objective or registered budgets:

- AdamW retains its exact first/second-moment and decoupled-weight-decay
  equations, while the denominator/update transient is evaluated in fixed
  chunks;
- the RADIO anchor touches at most the registered 2048 unique Gaussian rows,
  so PCGrad differentiates an explicit leaf row block and scatters it into the
  primary full-table gradient. This is algebraically identical to autograd's
  dense zero-filled anchor gradient and avoids a second `N x 512` allocation.

Both equivalences have direct numerical tests. A two-step `ramen` smoke using
the complete formal per-step configuration (four episodes, 32 relation edges,
2048 RADIO-anchor rows, and 64 exact-hit visual pixels) completed with IoU
change `+0.04531`, Brier `-0.01069`, boundary F `+0.00108`, unknown FP
`-0.00176`, and whole-field RADIO mean/p05/min cosine
`1.0/0.99999988/0.99999774`. This is only a runtime validation, not promotion
evidence. The frozen 600-step two-scene screen is the next gate; a 2000-step
extension remains forbidden unless both its instance trend and strict RADIO
source-dev no-regression criterion pass.

The completed screen is:

| scene | IoU gain | Brier change | boundary F change | unknown FP change | RADIO dev delta | PCGrad conflict rate |
|---|---:|---:|---:|---:|---:|---:|
| figurines | -0.03931 | -0.02718 | +0.14387 | -0.07465 | -0.000229 | 0.9350 |
| ramen | +0.04287 | -0.01687 | +0.10834 | -0.05494 | -0.000202 | 0.9217 |

The scene-macro IoU gain is only `+0.00178`, and `figurines` regresses. Both
strict source-dev RADIO render gates also fail. Consequently this arm fails
both required continue conditions and is stopped at 600 steps. No 2000-step
extension, weight/threshold search, source audit, sentinel, or benchmark run
was performed. The immutable evidence is:

- joint instance reports:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/joint512_figurines_dev_v1_budget600/upper_bound.pt.json`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/joint512_ramen_dev_v1_budget600/upper_bound.pt.json`;
- RADIO reports:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/source_visual_no_regression_joint_v1/figurines_budget600.json`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/source_visual_no_regression_joint_v1/ramen_budget600.json`.

This rejects writing instance state into the shared D512 under the current
plan, even when the full table rather than a residual is opened. Reusing this
candidate, continuing its budget, starting another shared-D512 joint update,
or tuning around its gate failure would violate the stopping rule. A clean
multi-teacher source-mapping restart is therefore recorded only as a possible
future-plan hypothesis, not as an authorized continuation of v3.

## Structured shared-private revision

A revised analysis separates the diagnostic role of raw RADIO reconstruction
from the capabilities consumed by queries. Strict zero raw-RADIO regression is
removed as a hard promotion condition. It remains a soft regularizer and
diagnostic; the hard condition is now a preregistered source-capability Pareto
gate over instance, semantic identity, image correspondence, category, geometry,
and rendering metrics. This revision does not retroactively promote the old
rank-16 candidate and does not authorize unrestricted joint-D512 updates.

The new candidate is one physical `N x 512` table with the fixed first layout:

| block | dimensions | write authority |
|---|---:|---|
| shared core | 320 | general source visual structure only |
| semantic identity | 128 | source semantic/category authority only |
| instance membership | 48 | SAM co-membership and ternary relation losses |
| boundary | 16 | SAM edge and visibility-conflict authority |

Instance and boundary readers may later consume stop-gradient context through
zero-initialized global one-way bridges. The first causal screen disables those
bridges. The optimizer owns contiguous table columns: Adam moments, weight
decay, and updates for one private task cannot touch another block. Direct
gradient and optimizer tests enforce this property.

Initialization is fresh and source-only. A fixed seeded JL map projects source
RADIO pixels into the shared block and exact-MPR lifts train-residue evidence to
Gaussians. A separate fixed seeded JL map projects source mask-aligned SigLIP2
descriptors into the semantic block and lifts only train-residue proposals.
Instance rows receive a seeded random initialization and boundary rows start at
zero. No historical field checkpoint, pre-fusion code, post-fusion coefficient,
residual candidate, target RGB, audit mask, or benchmark metric is opened.

Two-step full-per-step smoke tests validate both available complete-authority
scenes. `figurines` and `ramen` both serialize exactly one structured D512 plus
global heads; shared and semantic maximum absolute changes are exactly `0.0`,
while instance and boundary blocks change. The `ramen` smoke exposed and fixed
an implementation-only allocation where row indexing preceded column slicing;
column-first indexing avoids materializing a hits-by-512 tensor and preserves
identical values. Frozen 600-step source-dev screens are running on two GPUs.
They are structure evidence only: source audit and all benchmark data remain
sealed, and promotion additionally requires the missing registered text and
ScanNet category capability cohorts.

The frozen 600-step results are:

| scene | IoU gain | Brier change | boundary F change | unknown FP change | shared max change | semantic max change |
|---|---:|---:|---:|---:|---:|---:|
| figurines | +0.08422 | -0.32848 | +0.11387 | -0.36032 | 0.0 | 0.0 |
| ramen | +0.16793 | -0.33380 | +0.33145 | -0.39083 | 0.0 | 0.0 |

Scene-macro IoU gain is `+0.12608`; neither scene regresses and every proper
instance metric improves. This is substantially different from the
undifferentiated joint-D512 screen (`+0.00178` macro with a Figurines
regression): the private instance block can learn co-membership without
rewriting the source visual or semantic coordinates.

The source-dev capability diagnostics are invariant between two-step smoke and
600-step artifacts, as required by exact protected-block ownership:

| scene | projected RADIO render cosine | projected SigLIP proposal cosine | proposal retrieval top-1 |
|---|---:|---:|---:|
| figurines | 0.44080 | 0.70823 | 0.14894 |
| ramen | 0.70101 | 0.89169 | 0.14286 |

These values establish preservation, not absolute superiority over the
historical field. In particular, category and registered text-query source
metrics remain absent, so source audit and benchmarks stay sealed. The strong
two-scene instance trend authorizes a matched 2000-step development extension.
The original rank-16 writeback is also being rerun from scratch at its fixed
configuration as the non-structured shared-D512 comparator under the revised
capability contract; it is not used to initialize the structured candidate.

An apples-to-apples historical comparator then showed why exact protected-block
preservation is necessary but not sufficient. Both the historical RADIO render
and the fresh shared block were evaluated against the same seeded projected
RADIO teacher. Before any shared visual optimization, fresh structured D320
lagged the historical field by `-0.21997` on Figurines and `-0.08175` on Ramen.
The scheduled 2000-step private-only extensions were stopped without artifacts
because their parameter ownership made this capability deficit impossible to
repair.

The next registered screen added the required phase order with separate
partition optimizers:

```text
sampled source-train visual step (shared only)
-> instance/relation step (instance only)
-> boundary step (boundary only)
```

At 600 steps the two seeds both retain strong instance gains:

| seed | Figurines IoU gain | Ramen IoU gain | macro gain |
|---:|---:|---:|---:|
| 20260826 | +0.07186 | +0.17544 | +0.12365 |
| 20260827 | +0.08898 | +0.17638 | +0.13268 |

However, the visual phase does not pass the capability Pareto gate. The
historical-projected correspondence deltas are `-0.25491/-0.25647` for the two
Figurines seeds and `-0.09240/-0.09129` for the two Ramen seeds. The
source-train sampled visual loss falls,
but source-dev correspondence worsens, indicating mapping overfit rather than
instance-gradient interference. No 2000-step extension is authorized.

The next capacity diagnostic is therefore the preregistered hard-block S1
layout `448 visual + 48 instance + 16 boundary = 512`, with no semantic-private
block. It tests whether the shared D320 bottleneck itself causes the image
capability deficit. S1 is an explanatory arm, not the final shared-private
candidate, and uses the same source split, steps, losses, and thresholds.

S1 was run from a fresh source-only initialization with the visual block frozen.
This isolates representational capacity from the source-train visual overfit
already observed in the D320 alternating arm. Partition-owned writes remained
exact: the shared-block maximum absolute delta was `0.0` in both scenes. Its
600-step source-dev results are:

| scene | IoU gain | Brier change | boundary F change | unknown FP change | projected image delta from historical |
|---|---:|---:|---:|---:|---:|
| figurines | +0.08281 | -0.32846 | +0.11392 | -0.36030 | -0.21414 |
| ramen | +0.16793 | -0.33380 | +0.33145 | -0.39083 | -0.08242 |

The scene-macro IoU gain is `+0.12537`, confirming again that the private
instance/boundary coordinates work. The image-correspondence capability does
not: D448 reaches `0.45282` versus historical `0.66695` on Figurines and
`0.69861` versus `0.78103` on Ramen. Increasing the fresh visual block from
D320 to D448 therefore does not materially close the deficits (`-0.21997` and
`-0.08175` at D320). S1 fails the capability Pareto gate. S2 and S3 allocate
fewer visual dimensions, so they cannot answer this failed capacity hypothesis
and are not opened as image-repair searches. There is no 2000-step extension,
source-audit access, or benchmark access.

The fixed rank-16 historical writeback comparator was independently rerun from
scratch at its original configuration. It gives IoU gains `+0.01132` on
Figurines and `+0.09019` on Ramen (macro `+0.05075`) while its raw RADIO
diagnostic changes are `-0.000548` and `-0.000614`. Raw RADIO is no longer a
hard gate, but this rerun still lacks the registered source text/category suite
and does not dominate the structured arm's instance gains. It is retained only
as a comparator and is not promoted or used to initialize the new structure.

The structured revision is therefore a useful architectural result but not a
promotion candidate: block ownership solves instance-gradient interference,
whereas a fresh fixed-JL source visual initialization fails real held-out image
correspondence and sampled visual updates overfit it further. The next method
revision must improve the source visual mapping authority itself while
preserving private writes; it must not recycle a historical field into training
or resume any rejected arm under a new label.

## Product-space architecture comparison

Two previously proposed refinements were then implemented and compared with
the fixed hard-block product space under the same `320/128/48/16` layout,
source split, seed, 600 steps, four episodes, and 32 relation edges. All arms
persist exactly one `N x 512` table and no Gaussian-indexed sidecar.

- The learned orthogonal product arm uses 192 global disjoint Givens rotations.
  It is exactly orthogonal for every parameter value. Source visual authority
  may update only these constant-size angles; the Gaussian shared block stays
  frozen. An identity-initialized run exposed the expected stationary point, so
  the decisive rerun used fixed seeded angles in `[-0.05, 0.05]`.
- The shared-core low-rank-private arm adds zero-output rank-8 shared-to-instance
  and rank-4 context-to-boundary one-way branches. Context reads are detached;
  only the private blocks and global branch factors receive private-task writes.
  The branches add 6,016 global parameters and no per-Gaussian state.

The matched source-dev results are:

| architecture | Figurines IoU gain | Ramen IoU gain | macro IoU gain | macro Brier change | macro boundary F change | macro unknown FP change |
|---|---:|---:|---:|---:|---:|---:|
| fixed hard block | +0.08422 | +0.16793 | +0.12608 | -0.33114 | +0.22266 | -0.37558 |
| learned orthogonal product | +0.07186 | +0.17544 | +0.12365 | -0.33057 | **+0.24759** | -0.38092 |
| shared core + low-rank private | +0.07774 | **+0.20594** | **+0.14184** | **-0.33581** | +0.21468 | **-0.39148** |

The perturbed orthogonal angles converge back to identity: final mean absolute
angle is `1.07e-6` on Figurines and `1.21e-15` on Ramen. Its image and semantic
capability values are consequently identical to the hard-block initialization.
This is evidence that, for the current cosine/prototype objectives, the learned
orthogonal basis is an isometric reparameterization rather than additional
identifiable capacity. It is not selected.

The low-rank branches are active rather than dormant. Final instance-up norms
are `0.543/0.569` and boundary-up norms are `0.909/0.970` for
Figurines/Ramen. They produce the best macro instance IoU, Brier, and unknown-FP
change, though the gain is concentrated in Ramen and the orthogonal arm retains
the best macro boundary F. Among the three instance structures, low-rank
private branches are the preferred next development arm; the fixed hard block
remains the simpler robustness baseline.

No arm passes the full capability Pareto gate. All three preserve the same
projected SigLIP source-dev values but retain image-correspondence deficits of
`-0.21997` on Figurines and `-0.08175` on Ramen relative to the hash-bound
historical comparator. Therefore “best instance structure” does not mean
promotion: Teatime, Waldo, ScanNet, source audit, and benchmarks remain sealed.
The next authorized work is a new source visual mapping authority paired with
the low-rank-private instance structure, not further tuning of these private
branches or relaxation of the image gate.

Three implementation-only peak allocations were removed while retaining exact
losses and budgets: low-rank projections accumulate directly from column
blocks, repeated compositor Gaussian rows are compacted before private reads,
and episode/relation gradients are accumulated in exact weighted chunks before
one clipping/update. A numerical test verifies that compact relation loss and
its full-table gradient match the original formulation.
