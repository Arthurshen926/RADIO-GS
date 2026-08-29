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

## Historical-comparator fairness audit and learned codec screen

The historical projected-RADIO delta was subsequently audited against the
historical field's hash-bound `feature_frame_manifest` and
`selected_dataset_indices`. The comparator had opened every current source-dev
frame: all `8/8` Figurines residue-3 frames and all `8/8` Ramen residue-3 frames
overlap its construction views. The current train and audit cohorts also overlap
completely. The evaluator now fails closed on missing lineage, manifest hash
mismatch, or invalid selected indices, and labels both comparators
`diagnostic_nonheldout_comparator`. Consequently the earlier
`-0.21997/-0.08175` deltas are not held-out rejection gates.

The next visual-semantic screen kept the registered `320/128/48/16` layout and
did not open historical weights. A single cross-scene source-train PCA codec was
fit jointly on Figurines and Ramen native RADIO and SigLIP2 samples. RADIO D320
retains `0.94145` of sampled variance and SigLIP2 D128 retains `0.91008`.
Initialization applies the linear codec before exact-MPR and normalizes only the
aggregated Gaussian row. Before any private training, its source-dev results are:

| arm / scene | RADIO cosine | same-pixel top-1 | top-5 | margin | SigLIP cosine | proposal top-1 |
|---|---:|---:|---:|---:|---:|---:|
| fixed-JL / Figurines | 0.44080 | 0.09595 | 0.25049 | -0.17843 | 0.70823 | 0.14894 |
| PCA codec / Figurines | 0.42749 | 0.09302 | 0.24976 | -0.19265 | **0.93555** | **0.19149** |
| fixed-JL / Ramen | 0.70101 | 0.16748 | 0.46851 | -0.07833 | 0.89169 | 0.14286 |
| PCA codec / Ramen | 0.65775 | **0.17871** | **0.48218** | -0.09604 | **0.99970** | **0.16327** |

Thus the learned semantic codec passes a two-scene improvement screen, while
plain PCA visual mapping is mixed and cannot authorize private training. A
fixed-JL-visual/learned-SigLIP control using the corrected normalization order
is nearly identical to old fixed-JL on image metrics (`0.44128/0.70109` cosine,
`0.09521/0.16650` top-1), proving that normalization order alone is not the
visual bottleneck.

A bounded global D320 positive diagonal render metric was then learned from
source-train exact-hit render cosine and in-view correspondence cross entropy.
It updates no Gaussian and is shared across both scenes. On the PCA visual basis
it improves same-pixel top-1 to `0.11475/0.21143` and top-5 to
`0.29565/0.54736`, but Ramen margin is `-0.08704` versus the fixed-JL
`-0.07833`. This remains a Pareto failure. The final authorized codec control is
the same bounded render metric on the stronger fixed-JL visual basis; private
low-rank training remains closed pending that result. That final control reaches
only `0.09082/0.17432` top-1, `0.24707/0.47266` top-5, and
`-0.19963/-0.09009` margin on Figurines/Ramen. It regresses Figurines and is
rejected. This closes diagonal render-metric learning on both PCA and fixed-JL
visual bases; neither is attached to the low-rank-private branch. The successful
piece retained for the next method revision is the cross-scene learned SigLIP
semantic codec. The visual branch requires a more expressive correspondence-
trained encoder/decoder than PCA plus a diagonal metric.

Frozen evidence:

- comparator fairness reports:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/architecture_comparison_capability_v2_fairness_audit/`;
- PCA codec and render-metric codecs:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/cross_scene_source_codec_v1/`;
- source-dev capability reports:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/cross_scene_codec_capability_v1/`,
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/render_metric_codec_capability_v1/`,
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/hybrid_codec_capability_v1/`;
- rejected fixed-JL render-metric control:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/hybrid_render_metric_capability_v1/`.

## SUGM-v3.1 native-teacher ceiling and nonlinear visual writer

The v3.1 continuation freezes the `320/128/48/16` layout, the rank-8/rank-4
private branches, and the learned SigLIP semantic codec. It stops all orthogonal,
width, rank, projection-order, and diagonal-metric searches. Ramen's missing
official DINOv2 ViT-B/14 source maps were extracted for the exact same sealed 32
source views; Figurines already had complete `32/32` coverage.

Before training another D320 writer, uncompressed native-teacher exact-MPR
ceilings were measured from train residues 1/2 to dev residue 3:

| teacher / scene | dimension | render cosine | same-pixel top-1 | top-5 | margin |
|---|---:|---:|---:|---:|---:|
| RADIO / Figurines | 1280 | 0.45358 | 0.09814 | 0.25366 | -0.16267 |
| DINOv2 / Figurines | 768 | 0.44038 | 0.07300 | 0.18408 | -0.21236 |
| RADIO / Ramen | 1280 | 0.69638 | 0.18042 | 0.48584 | -0.07500 |
| DINOv2 / Ramen | 768 | 0.65269 | 0.11035 | 0.32031 | -0.12469 |

Compared with fixed-JL D320 RADIO (`0.09595/0.16748` top-1), removing
compression recovers only `+0.00220/+0.01294`. Thus compression is not the
dominant loss; ordinary multiview aggregation and task alignment remain limiting.
One preregistered single-step cosine-Huber IRLS aggregation was tested with
residual delta `0.10` and weight floor `0.05`. It fails the two-scene Pareto
screen: RADIO top-1 becomes `0.09863/0.17847`, with worse top-5 or margin, and
DINOv2 also remains mixed. No robust-weight threshold search is authorized.

The one registered nonlinear writer is therefore:

```text
RADIO LayerNorm -> rank-160 projection --\
                                         gated D320 fusion -> residual MLP -> unit D320
DINO LayerNorm  -> rank-96 projection  --/
```

It is trained jointly across Figurines and Ramen using source-train cross-view
same-Gaussian positive pairs. Distinct Gaussian identities in each batch are
hard negatives. The loss combines symmetric correspondence cross entropy,
positive-pair cosine, RADIO/DINO decoder reconstruction, and preservation of
the two teachers' local similarity graph. It opens no dev view, historical
field, target RGB, audit residue, or benchmark metric. A two-step four-GPU smoke
reduced total loss from `5.15176` to `4.52823`. The single formal nonlinear-codec
run used 300 steps and batch size 128; its loss fell from `4.92620` to `1.86040`
(`correspondence 4.13970 -> 1.59975`, dual reconstruction `1.98350 -> 0.62051`).
No architecture, rank, width, loss-weight, or threshold arm was searched.

Frozen ceiling evidence:

- `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/uncompressed_exact_mpr_ceiling_v1/`;
- `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/uncompressed_robust_mpr_ceiling_v1/`;
- `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_dinov2_source32/ramen/`.

The codec was then frozen and written by ordinary exact-MPR into a fresh D512,
while the learned SigLIP D128 codec initialized the semantic block. On the
unseen residue-3 source views this MPR-only nonlinear candidate obtains:

| scene | encoded render cosine | same-pixel top-1 | top-5 | margin | SigLIP cosine | proposal top-1 |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | **0.83212** | 0.09473 | 0.26294 | **-0.07914** | 0.93555 | 0.19149 |
| Ramen | **0.88149** | **0.21143** | **0.55786** | **-0.03680** | 0.99970 | 0.16327 |

Relative to fixed-JL, Ramen improves every correspondence statistic. Figurines
improves cosine, top-5, and margin, but top-1 changes from `0.09595` to
`0.09473`. This is not a strict two-scene Pareto pass, so private training was
not opened.

To complete the one missing registered component rather than create another
codec arm, the same frozen encoder was followed by one source-train
renderer-aware D320 refinement. Only visual columns were optimized through the
immutable exact compositor with render cosine, within-view correspondence, and
an initialization anchor. Semantic, instance, and boundary maximum absolute
change is exactly `0.0`. Its residue-3 results are:

| scene | encoded render cosine | same-pixel top-1 | top-5 | margin |
|---|---:|---:|---:|---:|
| Figurines | 0.57859 | **0.11523** | **0.30371** | -0.11447 |
| Ramen | 0.70395 | **0.24561** | **0.60010** | -0.04896 |

The refinement raises top-1/top-5 in both scenes and exceeds fixed-JL on all
three retrieval statistics, but it regresses cosine and margin relative to its
MPR-only parent. It is therefore retained as diagnostic evidence that
renderer-aware correspondence can recover ranking, but rejected as the new
visual-core checkpoint under the parent-Pareto rule. No loss-weight retuning is
performed. Phase 4 remains closed because neither visual candidate is a strict
two-scene Pareto successor and the preregistered source text/category gates are
still absent. Teatime, Waldo, source audit, and all benchmarks remain sealed.

Frozen nonlinear evidence:

- codec:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_visual_codec_v1/`;
- MPR-only D512 and capability:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_visual_initialization_v1/`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_visual_capability_v1/`;
- renderer-aware D512 and capability:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_visual_render_refinement_v1/`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260827/sugm_v3/native_visual_render_capability_v1/`.

## SUGM-v3.2 correspondence-authority and observation-set screen

The next revision keeps the `320/128/48/16` layout, learned SigLIP D128, and
rank-8/rank-4 private design frozen. Before training, a five-level source-only
error ladder adds Recall@1/5, MRR, positive similarity, hardest-negative
similarity, and margin. It also buckets the current pair authority by SAM
boundary and Gaussian footprint.

The ladder rejects the old training authority as a one-hot truth. Across all
residue-1 x residue-2 view combinations, only `1.56%/2.88%` of Figurines/Ramen
same-Gaussian best-pixel pairs are simultaneously bidirectional-mutual under
both native RADIO and DINO. Native fused-teacher Recall@1 on those labels is
only `10.43%/14.01%`, with margins `-0.18542/-0.18133`. The dominant loss is
therefore correspondence authority, not D320 capacity. No new width or codec
arm is authorized from this result.

A fresh authority is built only on the eight adjacent source-train view pairs,
using all valid `46 x 62` feature-grid pixels. Its tiers are fixed as:

- high: DINO bidirectional mutual plus the same exact-compositor top Gaussian;
- medium: DINO bidirectional mutual plus overlapping top-4 compositor support,
  while allowing different top Gaussians;
- weak: adjacent-view same-Gaussian best pixels, never used as one-hot truth;
- hard negative: the most DINO-similar candidate with disjoint top-4 support.

This yields `90/219` high/medium pairs on Figurines and `441/984` on Ramen,
versus `79,282/586,302` weak pairs. Positive-to-support-disjoint-negative DINO
margins are generally positive on overlapping adjacent pairs, validating the
new authority mechanism. Later Figurines pairs `25-26/29-30` yield no
support-consistent positive, which is retained as a coverage failure rather
than filled with weak labels.

Exactly one top-4 observation-set writer was then tested. It initializes the
existing RADIO+DINO pixel codec, applies a zero-initialized DeepSets update to
the four highest-responsibility pixels per Gaussian-view, and learns a global
view confidence. Only high/medium pairs enter the soft-distribution,
support-disjoint margin, neighborhood, and dual-reconstruction objectives. At
300 steps total loss falls from `1.45245` to `0.22866`, but the explicit
positive-negative score gap contracts from `0.06257` to `0.02925`; the hard
margin loss worsens from `0.03960` to `0.05166`.

The held-out residue-3 gate confirms rejection:

| scene | set render cosine | top-1 | top-5 | MRR | positive | hardest negative | margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 0.61054 | 0.06592 | 0.19727 | 0.13702 | 0.61090 | 0.72020 | -0.10930 |
| Ramen | 0.70203 | 0.12671 | 0.39380 | 0.25767 | 0.70292 | 0.75697 | -0.05405 |

Both scenes regress the MPR-only nonlinear parent (`0.09473/0.21143` top-1,
`0.26294/0.55786` top-5). The set writer is rejected. No top-K, loss-weight,
learning-rate, or threshold search, renderer refinement, private training,
source audit, or benchmark is run. The retained result is the new adjacent-view
multisource correspondence authority; the next method must avoid end-to-end
pixel-codec drift under this sparse authority, for example by freezing the
pixel codec and validating an authority-coverage mechanism before another
aggregator run.

Frozen v3.2 evidence:

- ladder:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/visual_mapping_error_ladder_v1/`;
- adjacent-view authority:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/multisource_correspondence_authority_v2/`;
- rejected set codec:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/native_visual_set_codec_v1/`;
- rejected D512 and held-out capability:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/native_visual_set_initialization_v1/`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/native_visual_set_capability_v1/`.

## SUGM-v3.3: source overlap graph and coverage repair

The adjacent-pair authority was replaced by a source-only overlap graph over
the residue-1/2 training views.  Edges are selected from symmetric exact-MPR
top-8 support overlap (four fixed neighbors per view).  Direct shared-support
anchors are locked; otherwise responsibility-weighted XYZ supplies an anchor
and a radius-2 window is scored equally by geometry, native DINO, and native
RADIO.  Every correspondence retains a continuous support/geometry/teacher/
cycle confidence and both edge directions are materialized.

A fixed view-authority gate rejects a view when its unique top-Gaussian count
is below 2% of the 46x62 feature raster.  This exposed a genuine responsibility
collapse in three Figurines training views: source-view indices 25, 29, and 30
have only 4, 8, and 2 unique top Gaussians.  The shards are sorted and their
frame indices agree with the source records, so this is not a loader error.
Those views are excluded rather than relabeled with geometry-only guesses.
Their removal changes top-8 row coverage only from 10.477% to 10.448%, and
any-hit row coverage only from 44.864% to 44.802%.

Frozen graph-union coverage (direct support and cycle distance no larger than
the fixed radius) is:

| scene | accepted views | graph edges | direct union mean/min | direct+cycle union mean/min | strict union mean/min | top-8 row coverage | any-hit row coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 13/16 | 32 | 99.51% / 95.41% | 76.51% / 50.11% | 75.02% / 47.48% | 10.45% | 44.80% |
| Ramen | 16/16 | 45 | 97.54% / 79.73% | 90.22% / 65.22% | 89.82% / 64.41% | 14.73% | 80.24% |

This repairs the correspondence-authority sparsity sufficiently to attempt a
frozen-codec structural propagator, especially on Ramen.  It does not repair
Gaussian-row coverage: the next stage must propagate from observed rows to
unobserved/weak rows using source-only XYZ, region membership, boundary
scalars, and confidence, while keeping the pixel codec frozen.  No private
training, source audit, target RGB, text query, or benchmark was opened.

Frozen artifacts:

- graph authority:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/source_overlap_graph_v3/`;
- union and row coverage:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/source_overlap_coverage_v1/`.

The next source-only step materializes, but does not yet train on, a nested row
propagation authority.  Observed rows map to themselves.  Unobserved rows first
use the strongest observed row in the same membership proposal and the same
128-cubed normalized-XYZ voxel; remaining rows use strongest-observed anchors
in 128-, 64-, then 32-cubed voxels.  Confidence combines a median-saturated
source evidence score with distance relative to the voxel diagonal.  Requiring
the region and fine voxel jointly is important: region alone produced
geometrically remote anchors and was rejected.

| scene | observed | + region/128 voxel | + 128 voxel | + 64 voxel | + 32 voxel | unassigned rows |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 44.80% | 47.10% | 94.83% | 96.59% | 97.92% | 3,508 |
| Ramen | 80.24% | 81.87% | 99.23% | 99.71% | 99.93% | 270 |

Fine-voxel confidence medians are `0.805` and `0.851`; their 99th-percentile
XYZ distances are `3.408` and `0.367` respectively.  The much larger absolute
Figurines scale is accounted for in confidence by the scene/voxel diagonal,
but this authority remains a candidate until a source-heldout feature
reconstruction gate confirms propagation fidelity.  The frozen candidate is:

`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/row_propagation_authority_v2/`.

## SUGM-v3.4: propagation fidelity gate and fail-closed coverage

Row-path coverage is not treated as visual fidelity.  Before opening source
dev, the fine propagation gate was fixed to fused native RADIO+DINO median
cosine at least `0.70` and at least `0.80` of the directly observed-row
median.  Residue 3 is used for this gate; the same 2% top-Gaussian diversity
gate rejects collapsed dev views.  Audit residue 0 remains sealed.

A single-best-pixel readout fails (`0.198/0.532` fine median for Figurines/
Ramen).  A frozen top-4 multiview observation prototype rendered with the full
exact-MPR weights improves the direct medians to `0.542/0.684`, but fine
propagation reaches only `0.276/0.638`.  Ramen preserves 93% of its direct
median but misses the fixed absolute gate; Figurines is an unambiguous failure.

Two preregistered structural repairs were then tested without changing the
gate.  Selecting one neighbor by geometry times observation reliability gives
`0.240/0.639`.  A top-4 normalized local mixture gives `0.267/0.636`.  Neither
passes, so no propagated row is authorized as a D512 visual write and no
codec, loss, threshold, or benchmark tuning follows.

The retained repair is fail-closed coverage modeling.  Every Gaussian receives
a coverage-confidence/unknown scalar policy.  Only rows with actual accepted
training-view compositor evidence have visual-write authority; region/voxel
paths remain explicit candidates with zero visual authority.  This authorizes
`75,621/168,791` Figurines rows and `307,087/382,687` Ramen rows, while all
remaining rows, including rows with no propagation candidate, abstain.

Frozen evidence:

- rejected single-row gate:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/row_propagation_source_dev_v1/`;
- rejected exact-MPR top-4 render gates:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/row_propagation_render_source_dev_v1/`,
  `v2/`, and `v3/`;
- final propagation candidates and fail-closed policy:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/row_propagation_authority_v4/`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/row_coverage_policy_v2/`.

The next eligible method change must learn an explicit unknown-aware structural
predictor from accepted rows and source correspondences while keeping the
pixel codec frozen.  Direct geometry copy or averaging is closed as the visual
coverage solution on this evidence.

## SUGM-v3.5: compact-space masked reconstruction and semantic-only repair

The frozen nonlinear parent supplies D320 RADIO+DINO shared codes and D128
SigLIP semantic codes.  A deterministic 20% mask is applied only inside rows
with true visual-write authority; reconstruction uses top-4 authorized
neighbors in the same 128-cubed voxel.  Before evaluation, the upper-bound gate
was fixed to at least 80% masked-row coverage and D320 median cosine at least
0.80.

Both scenes pass strongly:

| scene | masked coverage | D320 mean | D320 p10 | D320 median | D128 median |
|---|---:|---:|---:|---:|---:|
| Figurines | 99.78% | 0.97996 | 0.94926 | 0.99195 | 0.99987 |
| Ramen | 99.31% | 0.98717 | 0.96802 | 0.99604 | 0.99999 |

This shows that structure is predictable after the frozen codec even though
raw native-teacher copying is not.  Filling unknown shared+semantic D448 is
nevertheless rejected on source dev: Figurines visual top-1/top-5 regress from
`0.09473/0.26294` to `0.09106/0.25708`.  The block decomposition is then used
as intended: only unknown semantic D128 is interpolated.  This leaves every
shared D320 and private D64 value bitwise unchanged.

The semantic-only candidate is Pareto-positive on source dev.  Figurines
proposal cosine improves from `0.93555` to `0.99934` with top-1 unchanged at
`0.19149`; Ramen stays at `0.99970/0.16327`.  All visual cosine, retrieval, and
margin metrics exactly equal the parent.

One source audit on residue 0 confirms the result.  Figurines semantic top-1
improves from `0.09859` to `0.11268` and cosine is non-regressing
(`0.9854143` to `0.9854165`); Ramen remains unchanged within numerical noise.
All audit visual metrics exactly match the parent.  Target RGB and benchmark
metrics remain sealed.

The retained state is sealed as exactly one D512 plus five scalar reliability
values per Gaussian: visual-write authority, coverage confidence, unknown
probability, structural-candidate confidence, and semantic-membership
strength.  No Gaussian-indexed high-dimensional sidecar is persistent.

Frozen artifacts:

- masked upper bound:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/masked_structural_reconstruction_v1/`;
- rejected D448 candidate and dev capability:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/unknown_structural_initialization_v1/`
  and `unknown_structural_capability_v1/`;
- retained semantic-only candidate, dev, and audit:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/unknown_semantic_initialization_v1/`,
  `unknown_semantic_capability_v1/`, and `unknown_semantic_audit_v1/`;
- deployment-form scene state:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/unknown_aware_scene_state_v1/`.

## SUGM-v3.6: protected private continuation and architecture selection

Instance D48 and boundary D16 are trained from the retained semantic-only
candidate; shared D320, semantic D128, the native visual codec, and the
fail-closed coverage policy are not reinitialized.  A protected-continuation
loader rejects a mismatched membership hash, layout, historical-field access,
target-RGB access, or benchmark access.  It carries the sealed codec state and
requires visual loss weight zero.  Shared and semantic drift is checked again
after training and must be exactly zero.

The original partition optimizer still caused autograd to allocate a dense
D512 gradient, which exceeded the 2.7--3.2 GiB spare GPU budget on Ramen.  The
private phase now uses training-only D48/D16 owned-column buffers and merges
them into the sole D512 at deployment.  The buffers are absent from the
checkpoint state; tests confirm that the D512 gradient is absent and that only
the two private blocks change.  This is an optimization implementation change,
not an additional persistent Gaussian field.

Four fixed 600-step source-only runs compare hard blocks with shared-core plus
zero-output low-rank private branches.  Residues 1/2 train and residue 3 is the
only development split.

| scene | arm | mask-IoU delta | Brier delta | boundary-F delta | unknown-FP-mass delta | D320/D128 max delta |
|---|---|---:|---:|---:|---:|---:|
| Figurines | hard block | +0.03681 | -0.28451 | +0.11280 | -0.32710 | 0 / 0 |
| Figurines | low-rank private | +0.06193 | -0.29820 | +0.15333 | -0.34606 | 0 / 0 |
| Ramen | hard block | +0.20093 | -0.33276 | +0.38412 | -0.38533 | 0 / 0 |
| Ramen | low-rank private | +0.16956 | -0.32716 | +0.38436 | -0.41284 | 0 / 0 |

Hard block has a slightly larger macro IoU delta (`+0.11887` versus
`+0.11574`) because of Ramen, but it fails the preregistered no-scene-failure
gate on Figurines (`+0.03681 < +0.05`).  Low-rank private passes the IoU gate
on both scenes and has better macro Brier (`-0.31268`), boundary-F
(`+0.26885`), and unknown mass (`-0.37945`).  It is therefore the uniquely
eligible retained private architecture.  Orthogonal product is not reopened:
the earlier controlled arm was identity-equivalent and supplied no benefit.

The final low-rank checkpoints are then rerun through the native source-dev
capability evaluator.  Figurines visual cosine/top-1/top-5 remain
`0.832125/0.09473/0.26294` and semantic cosine is `0.999342`; Ramen remains
`0.881486/0.21143/0.55786` and `0.999699`.  Ranking metrics equal the parent;
only nondeterministic GPU reductions in two diagnostic means differ by about
`2e-8`.  No source audit is reopened because shared and semantic contents are
exactly protected and the development capability suite is unchanged.  Target
RGB and all benchmark metrics remain sealed.

Frozen evidence:

- hard-block and low-rank training reports/checkpoints:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_continuation_v1/`
  and
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_continuation_v2/`;
- hash-bound architecture decision:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_architecture_selection_v1.json`;
- retained low-rank capability suite:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_low_rank_capability_v2/`.
- deployment-form retained low-rank D512+R5 scene states:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/low_rank_unknown_aware_scene_state_v1/`.

## SUGM-v3.7: boundary metric correction and hybrid private candidate

The v3.6 boundary numbers are invalid as evidence for D16: the old evaluator
computed boundary-F from the image gradient of the rendered instance posterior
and never read the trained D16 block or boundary head.  Re-evaluating the
actual head preserves IoU, Brier, and unknown mass but exposes collapse in the
fully low-rank candidate (`0.00315/0.00469` boundary-F on Figurines/Ramen).
Accordingly, v3.6 retains only its instance-architecture conclusion; its
boundary conclusion is superseded here.

The first balanced-boundary repair also exposed two training defects.  Some
proposal episodes contain no positive boundary pixel inside known authority
and must be excluded rather than treated as all-negative.  In addition, zero
D16, zero low-rank up, and a zero head create a symmetry dead point that learns
only a constant bias.  The retained repair therefore uses only proposals with
both positive and negative boundary authority, class-balanced BCE plus soft
Dice, a fixed source-independent unit-axis head initialization, fresh boundary
down/zero up, and explicit non-degeneracy gates.

D320, D128, and the retained low-rank instance D48 remain exactly frozen.  The
corrected 600-step comparison is:

| scene | boundary arm | valid train proposals | corrected boundary-F | D16 std | branch-up norm |
|---|---|---:|---:|---:|---:|
| Figurines | hard D16 | 133 | 0.32658 | 0.00924 | 0 |
| Figurines | low-rank boundary | 133 | 0.32972 | 0.00906 | 0.37204 |
| Ramen | hard D16 | 111 | 0.32488 | 0.02213 | 0 |
| Ramen | low-rank boundary | 111 | 0.30625 | 0.02071 | 0.71823 |

Both arms are non-degenerate, but hard D16 has the better scene-macro
boundary-F (`0.32573` versus `0.31798`) and is simpler.  The retained private
structure is therefore **low-rank instance D48 plus hard boundary D16**.
Its original instance gains remain `+0.06193/+0.16956` mask IoU, while shared,
semantic, and instance blocks have zero drift during boundary refinement.

A canonical deployment query interface now loads only the sealed D512+R5 plus
constant-size global weights, creates one Gaussian posterior, and permits 2D
only as an exact rendering of that same posterior.  Both deployment sentinels
pass.  Deployment latent and every global tensor are exactly equal to the
selected checkpoint; IoU and boundary-F reproduce exactly, while Brier and
unknown-mass reductions use a fixed `1e-5` GPU numerical tolerance.

Frozen evidence:

- corrected diagnosis:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_boundary_source_dev_v1/`;
- rejected constant repair and retained non-degenerate repair:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/private_boundary_refinement_v2/`
  and `private_boundary_refinement_v3/`;
- hash-bound boundary selection:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/boundary_refinement_selection_v1.json`;
- retained hybrid deployment states:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/hybrid_private_unknown_aware_scene_state_v1/`;
- passing unified-interface deployment sentinels:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/deployment_source_sentinel_v3/`.

Target RGB, source audit residue 0, and all benchmark metrics remain sealed.

## SUGM-v3.8: native multimodal query boundary and image-identity diagnosis

A v3-native `QueryPacket` and canonical query interface are implemented
without importing the forbidden historical querying or query-native model
trees.  Text and image packets contain exactly one finite frozen-encoder
SigLIP D1536 token; prompt packets contain only Gaussian-domain seed
probability with NaN retained as unknown.  The sealed scene SigLIP codec maps
text/image tokens into D128 identity scores.  Every modality then compiles
anchor rows and calls the same instance function, producing one Gaussian
posterior that 2D can only render unchanged.  Unit tests show that identical
text/image tokens commute exactly through identity scores, anchors, weights,
and the final posterior.

The first source-dev image-query diagnostic uses held-out proposal crop tokens
only for identity selection.  Source relations are not used to create the
posterior; they measure whether selected anchors fall in independent
training-view support.

| scene | relation-anchor IoU ceiling | image top-8 IoU | Brier | unknown mass | peak support hit | top-8 support precision |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.35436 | 0.23400 | 0.24767 | 0.45954 | 50.00% | 41.54% |
| Ramen | 0.51226 | 0.42044 | 0.21498 | 0.48046 | 70.27% | 66.89% |

This localizes the current query gap to semantic identity-anchor selection,
not the instance memory or posterior interface.  A single immutable identity
peak was tested as one causal control rather than a K search.  It is rejected:
IoU falls to `0.21413/0.37763`, and both Brier and unknown mass regress.  Thus
multiple anchors add real information, but the raw semantic cosine metric
mixes wrong-object anchors.  Further K/threshold or geometry-locality tuning is
closed.  The next eligible change is a constant-size source-only cross-view
semantic identity metric/adaptor with D512 and the posterior frozen.

Frozen evidence:

- top-8 image-query diagnostic:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/image_query_source_dev_v1/`;
- rejected identity-peak-only control:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/image_query_source_dev_peak_only_v1/`.

No text-quality claim, prompt-quality claim, source audit, sentinel expansion,
or benchmark execution is authorized by this diagnostic.

## SUGM-v3.9: identity-adapter controls and first benchmark opening

A shared rank-16 residual identity adapter improves image-query source dev on
both calibration scenes, but fails the independent source audit macro check.
It is rejected and the raw sealed identity mapping remains the parent.  A
shared orthogonal text alignment is also non-causal: source-paired cosine is
already `0.99931` before and `0.99954` after alignment.  Finally, LERF-style
positive-versus-canonical-negative relevancy changes every top-8 text anchor
but leaves the compressed D128 scores within about `1e-4` of `0.5`; the first
benchmark control remains essentially unchanged.  None of these adapters is
retained.

The first benchmark opening on Figurines/Ramen exposes the actual deployment
failure.  With raw text cosine and the shared instance posterior, 45--49% of
Gaussian/query pairs exceed the fixed `0.6` selection threshold.  LERF-2D is
`0.01234/0.02913` mIoU and strict-threshold LERF-3D is
`0.01383/0.03050`.  A source-derived fixed 1.1436% extent cap makes both worse,
so it too is rejected.  The benchmark is used only to diagnose excessive
coverage; it does not select the following repair.

Frozen controls:

- rejected shared adapter dev/audit:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/image_query_source_dev_adapter_v2/`
  and `image_query_source_audit_adapter_v1/`;
- orthogonal text control:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/orthogonal_text_alignment_v1/shared.pt`;
- raw and fixed-fraction benchmark controls:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/lerf_full_eval_v1/`
  and `lerf_full_eval_extent_v1/`.

## SUGM-v3.10: centered identity--extent posterior

The D128 field is highly anisotropic: known Gaussian/text cosine values are
concentrated near `0.999`, while unknown rows are zero.  A deterministic
scene-field centroid removal expands the identity distribution to roughly
`[-0.40, 0.35]` without adding Gaussian state.  Identity then contributes a
robustly normalized logit to the existing D48 extent posterior.  This remains
one posterior; identity is neither a second output field nor a post-render
mask.  Candidate global weights `1`, `2`, and `4` are compared only on source
dev, with no hard gate and a minimum-intervention tie break.

| weight | source-dev macro IoU | selected-weight audit macro IoU |
|---:|---:|---:|
| 1 | 0.41081 | -- |
| 2 | 0.43914 | -- |
| 4 | **0.46153** | **0.45949** |

At the retained weight `4`, independent audit IoU is `0.36951` on Figurines
and `0.54947` on Ramen, versus raw `0.23834/0.42577`.  The source-selected
repair raises preliminary LERF-2D to `0.02471/0.04157`.  Strict-threshold
LERF-3D is `0.01402/0.03135`; the standard fixed top-2% readout, reported as a
less brittle evaluation view rather than a tuned method component, is
`0.02840/0.03232`.  Full four-scene evaluation is completed in the next
section after the two large scene states are materialized.

Frozen source selection:

- source sweeps and independent audits:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/source_identity_extent_sweep_v1/`
  and `source_identity_extent_validation_v1/`;
- hash-bound shared weight decision:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/identity_extent_selection_v1/shared.json`.

## SUGM-v3.11: complete four-scene LERF-2D/3D evaluation

The two large scenes use the same protected low-rank-instance plus hard-D16
structure for 300 source-only steps.  This reduced budget is sufficient for a
complete evaluation and avoids treating 600 steps as a brittle gate.  Teatime
source-dev IoU improves from `0.35551` to `0.46121`; Waldo Kitchen improves
from `0.33407` to `0.40352`.  Brier improves by `0.31204/0.30004` and unknown
mass by `0.37333/0.35428`.  D320 and D128 remain exactly unchanged.  The fp16
protected memory is assembled into fp32 on CPU at deployment, and posterior
evaluation is row-chunked; both are memory-only implementation changes.

All four scenes use the source-selected centered identity--extent weight `4`,
one D512+R5 state, and exactly the same Gaussian posterior for 2D and 3D.
LERF-3D is reported both with the strict absolute `0.6` probability threshold
and with one fixed, scene-independent top-2% readout.

| scene | samples | LERF-2D mIoU | 2D localization | 3D strict mIoU | strict Acc@.25/.50 | 3D top-2% mIoU | top-2% Acc@.25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Figurines | 56 | 0.02471 | 0.07143 | 0.01402 | 0 / 0 | 0.02840 | 0.01786 |
| Ramen | 71 | 0.04157 | 0.09859 | 0.03135 | 0 / 0 | 0.03232 | 0 |
| Teatime | 59 | 0.01695 | 0.01695 | 0.03365 | 0 / 0 | 0.02769 | 0 |
| Waldo Kitchen | 22 | 0.08915 | 0.18182 | 0.09356 | 0.09091 / 0.09091 | 0.05903 | 0.04545 |
| scene macro | 208 total | **0.04309** | **0.09220** | **0.04315** | **0.02273 / 0.02273** | **0.03686** | **0.01583** |

The complete result is still modest, especially for Figurines and Teatime,
but it is now a valid end-to-end result rather than a two-scene proxy.  The
coverage repair is strongly supported by source dev/audit and improves the
first two-scene 2D benchmark.  The fact that strict 3D outperforms top-2% on
Teatime and Waldo also confirms that no single aggressive gate should be made
the method definition.

Frozen evidence:

- large-scene private candidates and sealed states:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/full_lerf_private_low_rank_v3/`
  and `full_lerf_scene_state_v1/`;
- authority-bound query caches:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/lerf_renderer_bridge_final_v1/`;
- complete evaluator outputs and hash-bound aggregate:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/lerf_full_final_v1/`
  and `lerf_full_final_v1/summary.json`.

## SUGM-v3.12: posterior and text-anchor failure isolation

The poor full-LERF result was reopened as a staged source-only diagnosis rather
than another benchmark-selected gate sweep. Contribution-weighted held-out
hits confirm that `sigmoid(cos/T)` maps unrelated `cos≈0` evidence near
probability `0.5`. A positive-versus-explicit-negative upper bound reduces
background mass and Brier in all four scenes, but reduces source render IoU in
all four scenes, so it is not connected to deployment.

On identical held-out proposals, image-selected top-8 anchors already lose
substantial IoU relative to relation-authorized supports. A separate source
text diagnostic localizes a larger cross-modal failure: raw text top-8 support
precision is zero on the available Ramen and Waldo dev pairs and one third on
Teatime. The previously rejected orthogonal alignment was rechecked using
held-out anchor precision instead of anisotropy-dominated raw cosine and does
improve Ramen and Teatime. Thus the old `0.999` cosine criterion was
non-informative.

A strongly identity-regularized affine text-only alignment was fitted from 27
source-train pairs. Ridge `1` raises dev anchor precision and the independent
audit macro, but exact source render gives only a small dev gain
(`≈0.467→0.485`, pair weighted) and a large audit regression
(`≈0.380→0.267`). It is rejected. This closes posterior-only and post-codec
text-adapter repairs: the next change must repair D128 cross-modal
write/compression while D320/D48/D16 stay protected.

Frozen evidence:

- posterior and image-anchor diagnostics:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/membership_logit_diagnostic_v2/`
  and `membership_logit_diagnostic_v3/`;
- raw, orthogonal, and affine text-anchor diagnostics:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/text_identity_anchor_diagnostic_raw_v1/`,
  `text_identity_anchor_diagnostic_orthogonal_v1/`, and
  `text_identity_anchor_diagnostic_affine_grid_v1/`;
- rejected affine source render dev/audit:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/text_identity_render_diagnostic_v1/`.

Two codec-level controls were then tested without changing D320, D48, or D16.
A fixed-seed JL rewrite of D128 preserves no useful source text-anchor support
and regresses most scene/split renders, so generic distance preservation is not
enough. A second control keeps image D128 unchanged and learns a direct
D1536-text-to-D128 correction around the image-PCA basis. Its correction rank
is at most the 27 source-train pairs. The least-regularized candidate improves
Figurines and Waldo dev, but regresses Ramen from `0.735` to `0.308` and
Teatime from `0.229` to `0.122`; pair-weighted dev falls from about `0.467` to
`0.308`. Both controls are rejected. The remaining blocker is insufficiently
diverse source language-pair authority for learning a universal cross-modal
codec, not a posterior gate or an implementation fallback.

- fixed-JL protected-block control:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/fixed_jl_semantic_scene_state_v1/`
  and `fixed_jl_text_identity_render_v1/`;
- direct raw-text projection control:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260828/sugm_v3/direct_text_projection_candidates_v1/`
  and `direct_text_projection_source_dev_grid_v1/`.

## SUGM-v3.13: historical-language-lineage correction and clean semantic rewrite

The apparent 27-pair language authority used in v3.12 was reopened at the
artifact-provenance level.  Its `proposal_probability` was generated from a
historical latent-score cache and field-tail readout.  Static namespace audits
had only proved that v3 code did not import an old module; they did not prove
that an input artifact was independent.  Therefore every conclusion selected
from those 27 pairs is superseded.  The poor full-LERF numbers remain valid
measurements of the old method, but they are not evidence for retaining its
text alignment or semantic codec.

A fresh source-only ternary authority is built directly from query-independent
SAM proposal memberships, mask-aligned SigLIP crop summaries, official frozen
text embeddings, and canonical null phrases.  Cross-view proposal components
use mutual-best strong geometry/descriptor/context matches.  Descending-strength
union rejects repeated-view and explicit-different conflicts.  Text seeds use
only residues 1/2; dev/audit text responses never select labels.  The retained
medium-evidence rule requires mean foreground/context response above canonical
null, proposal top-3 membership, and evidence in two training views.  It gives
33/67 queries with positive authority, 426 explicit positive pairs and 3,400
explicit negative pairs; all other pairs remain unknown rather than becoming
false negatives.

The corrected error ladder localizes the old failure.  Native SigLIP D1536 has
positive held-out margins of roughly `0.03--0.056`.  The old image-PCA D128
compresses these to about `1e-6`, and old Gaussian writing makes them negative.
One shared query-discriminative D1536-to-D128 projection was therefore fitted
from scratch with canonical-null response, listwise retrieval, sibling margin,
and native cross-modal distillation.  Three seeds ran concurrently; seed
`20260829` was selected only on source dev.  Its source-dev macro Recall@1 is
`0.9167` over the three evaluable scenes (20 queries), with margin `0.4299`.
Independent audit gives macro Recall@1 `0.8373` over four scenes (27 queries),
with margin `0.3752`.  Waldo residue 0 has no query with both explicit positive
and negative authority and is reported as not evaluable, not as zero.

A query-free two-pass Gaussian writer then combines proposal quality with
directional consensus in the learned D128 space.  Weak conflicting writes are
downweighted continuously and zero-evidence rows stay unknown.  Held-out
written-memory Recall@1 is `0.75/0.875/0.75` on Figurines/Ramen/Teatime dev and
`0.857/0.875/0.889/1.0` on the four audit scenes.  Every written-memory margin
is positive (`0.0476--0.7678`).  Thus the scientific chain from native semantic
response through D128 compression into Gaussian D128 is closed for covered
rows.  Coverage is not closed: direct reliable writes cover only
`25.5%/54.7%/54.8%/21.0%` of the four Gaussian sets.  No geometry copy or
benchmark-selected threshold is authorized to fill the remainder.

Finally, a constant-size null-aware posterior module and synthetic scientific
suite expose the earlier D16 integration bug.  Boundary is unsigned, so it
does not hard-delete edge rows; it transfers authority from expanded instance
support back to local identity evidence.  Synthetic identical-sibling,
multipart, cross-boundary, null/unknown, and shared-2D/3D-posterior checks pass.
The module is not connected to benchmark deployment until it is trained from
real source authority and the remaining semantic coverage problem is solved.

Frozen evidence:

- clean language authority:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/native_language_authority_v3/`;
- corrected old-codec error ladder:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/semantic_mapping_error_ladder_v2/`;
- three semantic-codec seeds and held-out evaluations:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/query_discriminative_semantic_codec_v1/`
  and `semantic_codec_evaluation_v1/`;
- clean conflict-aware Gaussian D128 and held-out evaluations:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/conflict_aware_semantic_memory_v1/`
  and `semantic_writer_evaluation_v1/`.

The frozen source-only RADIO+DINO D320 writer was then provenance-audited as a
coverage input.  Its direct train-view coverage is
`44.9%/80.2%/75.8%/61.3%`, and its metadata seals historical-field, target-RGB,
and benchmark access as false.  A shared linear masked writer is fitted from
directly observed D320/D128 row pairs with a deterministic 80/20 row split.
Held-out median D128 cosine is `0.756/0.919/0.897/0.876`; Figurines has a weak
10th percentile of `0.391`, while the other scenes are `0.523--0.598`.

The existing v3 split convention is train residues 1/2, dev residue 3, and
audit residue 0.  An earlier summary in this work session accidentally named
residues 0 and 3 in the opposite order.  Replaying selection on the correct
dev residue leaves semantic-codec seed `20260829` unchanged.  However, because
both held-out residues had already been evaluated in the same batch, the
residue-0 results below are validation evidence, not a pristine independent
audit.  A future claim of independent audit requires a newly sealed source
cohort.

On correct source dev, masked coverage is retained for Ramen, Teatime, and
Waldo: Recall@1/MRR do not regress and explicit margin increases.  Figurines
Recall@1 drops from `0.857` to `0.714`, so its predicted writes are rejected.
The retained per-scene semantic coverage is therefore
`25.5%/80.2%/75.8%/61.3%`.  Descriptive residue-0 validation keeps retrieval
unchanged on Ramen and Teatime; Waldo has no evaluable query on that split.

- masked writer and expanded-memory candidates:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/masked_semantic_writer_v1/`
  and `masked_semantic_memory_v1/`;
- dev evaluations and provenance-honest selection replay:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/semantic_writer_evaluation_v2/`
  and `masked_semantic_writer_selection_v2.json`.

## SUGM-v3.14: disentangled null calibration and retained-private-route rejection

The complete query interface exposed two later-stage errors which were hidden
by the proposal-space semantic-writer evaluation.  First, canonical negatives
were converted to a positive-versus-null probability *before* identity-anchor
selection.  This changed the ranking even when the raw D128 text response was
correct.  On correct source dev, Teatime raw D128 identity has Recall@1
`0.889` and margin `+0.2025`; the prematurely null-adjusted identity falls to
Recall@1 `0.444` and margin `-0.0070`.  The interface now exposes raw positive,
canonical null, and unknown evidence separately.  Positive text response alone
selects anchors, while null remains an explicit final-posterior input.

Second, an initial calibrator was trained on proposal-mean features followed by
a sigmoid, while deployment applies a sigmoid per Gaussian and only then
renders or pools.  That proxy predicted Teatime Recall@1 `0.778`, but the exact
deployment order produced `0.444`.  This proxy result is rejected.  The
corrected training evidence preserves the actual order: Gaussian logit,
sigmoid, membership-weighted proposal mean.  Deterministic membership-mass
quantiles cap each explicit proposal/query support at 512 hits without turning
unknown pairs into negatives.  Three seeds were fitted from residues 1/2 only;
their train outcomes are effectively identical.

The exact source-dev comparison is decisive:

| route | Figurines R@1 | Ramen R@1 | Teatime R@1 | Waldo R@1 | scene-macro R@1 | scene-macro MRR | scene-macro margin |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw clean D128 identity | 0.857 | 0.875 | 0.889 | 1.000 | 0.905 | 0.942 | +0.397 |
| trained D128 + retained D48/D16 | 0.571 | 1.000 | 0.556 | 1.000 | 0.782 | 0.862 | +0.328 |
| trained semantic-local control, D48/D16 disabled | 0.571 | 1.000 | 0.667 | 1.000 | 0.810 | 0.874 | +0.306 |

The full posterior has Recall@5 `1.0` and a positive mean margin in every
scene, so the failure is not a strict gate or a total language collapse.
Nevertheless it is worse than raw identity by `0.123` macro Recall@1, and
removing D48/D16 improves Recall@1 and MRR.  Teatime shows the clearest causal
ordering: `0.889` raw identity, `0.667` without the retained private route, and
`0.556` with it.  Earlier direct D48 measurements were also negative on
Figurines and Teatime.  Therefore the retained old D48 instance geometry is
not merely under-calibrated; it is incompatible with the newly rewritten D128
semantic identity space.  D16 is unsigned and cannot repair a wrong instance
prototype by itself.

This satisfies the predeclared negative terminal condition.  The candidate is
not promoted to a new full LERF2D/3D benchmark run, because the source
scientific chain does not close and benchmark metrics must not be used to tune
around that failure.  The previously completed LERF full result (`0.0431` 2D
scene-macro mIoU and `0.0431` strict-3D scene-macro mIoU) remains evidence for
the superseded historical-language route, not for this corrected candidate.
The next method version must retrain or replace instance structure against the
clean semantic identity authority; retaining the old D48/D16 branch is
rejected.

Frozen evidence:

- raw/null/D48 error separation:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/clean_query_evidence_dev_v2/`;
- exact-order Gaussian evidence:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/clean_posterior_evidence_v2/`;
- three-seed full and semantic-local calibrators:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/null_calibrated_posterior_exact_v2/`;
- exact source-dev deployment comparison:
  `/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260829/sugm_v3/clean_query_exact_calibrated_dev_v2/`.

Verification after the correction: all `104` v3 tests pass in the repository
runtime, the v3 static historical-import/scene-token audit returns `[]`, and
`git diff --check` passes.
