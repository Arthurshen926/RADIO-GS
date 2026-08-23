# Capability order and object-relation closure

## Decision

This continuation identifies one positive field-level mechanism and closes one
negative proposal-level branch.

1. Applying the frozen nonlinear language capability head to each 2D source
   observation before exact-MPR lifting gives a stable ScanNet gain.
2. A small query-independent residual decoder over frozen L512 recovers almost
   all of that gain without adding per-Gaussian state.
3. The current LERF source-SAM proposal authority cannot reconstruct object
   extent even when proposal identity is replaced by a same-object oracle.
   Proposal retrieval, union and posterior tuning are therefore stopped.

The ScanNet experiment uses no benchmark labels, masks, category names, text
queries or evaluation RGB during target construction or decoder training.
Category text is opened only by the unchanged evaluator after every source
gate and query cache has been sealed.

## First-principles intervention

Exact compositor lifting is linear in the lifted observation, while the
official SigLIP summary head is nonlinear.  Consequently

`MPR(summary(RADIO_2D)) != summary(MPR(RADIO_2D))`.

The left side is the correct capability statistic.  The right side was the
deployed D512 typed readout and loses view-local nonlinear evidence before it
can be aggregated.  The direct teacher computes the left side.  The deployable
candidate uses a zero-initialized residual decoder

`normalize(D512_descriptor + decoder(L512))`.

It has one shared approximately 4.2 MB checkpoint per scene, changes no
Gaussian geometry, and adds no row-indexed parameter.  A fixed modulo-eight
Gaussian-row split supplies the source-only validation gate.  Unobserved rows
fall back to the old descriptor exactly.

## ScanNet three-scene semantic ladder

All values below are VALA pseudo-volume mIoU with the same text bank, prompt
ensemble, categorical competition, pseudo ground truth and Gaussian-domain
metric.

| scene | arm | split19 | split15 | split10 |
|---|---|---:|---:|---:|
| scene0000_00 | D512 baseline | 0.34441 | 0.32313 | 0.35116 |
|  | direct capability | 0.35992 | 0.33842 | 0.38325 |
|  | restored decoder | 0.35991 | 0.33759 | 0.38528 |
| scene0062_00 | D512 baseline | 0.27887 | 0.27821 | 0.36375 |
|  | direct capability | 0.30050 | 0.30037 | 0.39741 |
|  | restored decoder | 0.29997 | 0.30029 | 0.39745 |
| scene0070_00 | D512 baseline | 0.28825 | 0.30541 | 0.32357 |
|  | direct capability | 0.30177 | 0.32406 | 0.35416 |
|  | restored decoder | 0.30028 | 0.32226 | 0.35446 |
| three-scene macro | D512 baseline | 0.30384 | 0.30225 | 0.34616 |
|  | direct capability | 0.32073 | 0.32095 | 0.37827 |
|  | restored decoder | 0.32005 | 0.32005 | 0.37907 |
|  | restored delta | **+0.01620** | **+0.01780** | **+0.03291** |

All nine scene-by-split direct comparisons and all nine restored comparisons
improve.  This rules out one-scene category composition as the explanation.
The restored decoder also passes every source gate:

| scene | baseline mean / p05 cosine | decoder mean / p05 cosine |
|---|---:|---:|
| scene0000_00 | 0.99006 / 0.97778 | 0.99796 / 0.99568 |
| scene0062_00 | 0.99611 / 0.99164 | 0.99893 / 0.99765 |
| scene0070_00 | 0.99171 / 0.98183 | 0.99811 / 0.99554 |

The direct target is an attribution ceiling because it stores the full 1536D
teacher descriptor.  The residual arm is the field-compatible method.  The
unchanged configuration has now completed all eight paper scenes: baseline,
direct and restored 19/15/10-class macro mIoU are respectively
`0.33535/0.33370/0.42213`, `0.35535/0.35180/0.45769`, and
`0.35613/0.35300/0.45954`.  Restored gains are
`+0.02078/+0.01930/+0.03741`; macro mAcc also improves on all three splits by
`+0.00450/+0.01307/+0.02485`.  All eight deterministic held-out-row source
gates pass.  The typed categorical compiler is therefore promoted.  The
hash-bound full result is
`paper/artifacts/scannet_capability_before_mpr_paper8_result_20260823.json`.

## LERF cross-task falsification

The identical direct descriptor was frozen from 120 Figurines source views
and evaluated with the existing LERF2D and LERF3D readouts, thresholds, text
bank and geometry.  Of 168,791 Gaussian rows, 96,879 were directly observed;
the other 71,912 rows inherited the deployed D512 descriptor bit-for-bit.

| protocol | D512 baseline | direct capability | delta |
|---|---:|---:|---:|
| LERF2D sample-micro mIoU | 0.37021 | 0.28964 | -0.08058 |
| LERF2D LocAcc | 0.91071 | 0.89286 | -0.01786 |
| LERF3D mIoU | 0.44840 | 0.27927 | -0.16913 |
| LERF3D Acc@0.25 | 0.78571 | 0.53571 | -0.25000 |
| LERF3D Acc@0.50 | 0.44643 | 0.25000 | -0.19643 |

This is a useful negative result rather than a threshold failure to tune away.
The official summary statistic improves broad mutually-exclusive category
competition on ScanNet but suppresses several sparse fine-grained LERF
identity peaks.  Capability-before-MPR is therefore restricted to the typed
categorical compiler.  The LERF branch retains identity/extent factorization,
and no LERF residual decoder is trained from this rejected teacher.

## LERF source-relation gate

The frozen L512 global relation decoder improves source-heldout cross-proposal
same-object AUC from `0.78621` to `0.90852`, so object identity relations are
learnable from the latent.  That success does not solve extent.  On held-out
source visibility, even a known-same oracle over the existing SAM proposals
reaches only `0.08042` binary IoU with visibility-normalized marginalization;
the learned relation decoder reaches `0.04699`.  The corresponding hard-support
oracle remains about `0.20` IoU.

The bottleneck is therefore the semantic support geometry of the current
proposal/membership carrier, not another association score.  A benchmark run
was correctly not opened after this source gate failed.  The next LERF
experiment is the same capability-before-MPR compiler above, followed, only if
needed, by a direct Gaussian membership teacher or a split-only correction on
the measured high-risk support tail.

The direct membership continuation has now closed four additional source-only
controls on the same deterministic held-out views.  Primitive similarity is
`0.11764` held-out IoU; hard exact-MPR membership reaches `0.12761`, track
augmentation `0.11677`, soft exact-MPR membership `0.11794`, and the explicit
identity-seeded Gaussian-pair decoder `0.04719`.  None clears the fixed
`0.20` absolute and `+0.02` gain gate.  The pair decoder is the requested
`P(i~j)=D_object(z_i,z_j,delta_x)` realization, so its failure is direct
evidence against treating a stronger relation score as sufficient.  One final
unchanged control uses the already-sealed denser grid16 proposal carrier to
separate relation-model failure from proposal coverage.  Details are bound in
`paper/artifacts/lerf_frozen_l512_membership_source_gate_result_20260823.json`.

The denser 579-proposal grid16 controls are now also closed.  Direct membership
improves its weaker primitive held-out baseline from `0.09490` to `0.12269`
(`+0.02779`) but remains below the fixed `0.20` absolute gate.  The Gaussian
pair decoder regresses to `0.06016`.  Thus no learned relation/membership
decoder is promoted.  The valuable grid16 signal remains the unchanged
identity-extent benchmark readout (`0.40065` Figurines LERF2D mIoU and about
`0.45402` LERF3D mIoU).  Its unchanged full4 expansion is now complete and
rejects proposal density as the missing variable: LERF2D sample-micro mIoU is
`0.38884` versus retained `0.39584`, while LERF3D is `0.35657` versus
`0.39684`.  Localization is exactly unchanged at `0.87981`.  The complete
result is sealed in
`paper/artifacts/lerf_grid16_proposal_density_full4_rejection_20260823.json`.

## NVOS all-view compiler execution seam

The previously absent official-SAM execution layer is now implemented as
`build_nvos_synchronous_multiview_sam3_inventory.py`.  It requires the complete
Cartesian candidate/view cohort, explicit positive and negative evidence,
hash-bound captured RGB and exact assignment lineage, and seals all cells
before the existing streaming adjoint/marginal stage can run.  The missing
carrier-native plan producer is now implemented as
`build_nvos_synchronous_multiview_candidate_plan.py`: it exact-adjoints the
sealed field prompt and official signed scribbles, reprojects them to both
registered views, treats overlapping sign transport as unknown, and emits ten
equal-likelihood deterministic SAM trials.  Fourteen focused plan, producer,
marginal, and materializer tests pass.

The first point-only Fern closure failed: robust log-odds fusion reached
`0.49936` foreground IoU.  Treating an absent per-view detection as unknown and
composing only positive detections by noisy-OR raises the identical sealed
observations to `0.68301`, a causal gain of `+0.18364`, but still does not
recover the retained RGB-assisted result.  Replacing hashed points by ten
weighted-farthest trials also fails (`0.67118`).

Receipt inspection resolves the discrepancy: the retained `0.83047` method is
not point-SAM.  Its field mask generates a box, official SAM3 returns 200 region
proposals, and signed evidence selects one proposal without target metrics.
The new all-view box compiler restores exactly that identity--extent split.  On
Fern its native target observation is `0.83061` IoU (proposal 78/200), and the
sealed source+target exact-adjoint positive/unknown posterior reaches `0.83340`
IoU and `0.94626` pixel accuracy.

The carrier-native cold-start full8 is now complete.  The same-run target-only
macro is `0.81743`; unconditional all-view positive/unknown fusion is `0.81385`
(`-0.00358`).  It improves Fern, fortress and leaves, but regresses the horns
and trex scenes.  Thus the native execution seam is closed, while symmetric
source-view union is rejected as the final reliability rule.

Replaying the independently frozen component-local identity-density limiter
on the all-view masks raises macro IoU to `0.82624` (`+0.01239` over
unconditional all-view and `+0.00881` over same-run unfiltered target-only).
Seven of eight scenes improve relative to unconditional all-view; Fern changes
by `-0.00029`.  This replay occurred after the unconditional development
metrics had been opened, although its operator and `0.05` authority were
already frozen, so it is recorded as mechanism evidence rather than a
prospectively blind promotion.  It also does not beat the existing target-only
component-local development row (`0.82651`).  The remaining gap to the
historical LUDVIG-assisted `0.91623` ceiling is therefore source-view candidate
identity/transport authority, not the absence of native official-SAM regions.
The bound result is
`paper/artifacts/nvos_native_all_view_box_full8_result_20260823.json`.

## Contract and implementation corrections

The builder already projected `siglip_summary` per 2D token before MPR, but its
legacy receipt flag named only DINO and SAM as pre-MPR capabilities.  The flag
now includes `siglip_summary`; emitted tensors from the earlier run were
correct and their detailed observation-lifting contract already recorded
`feature_projection_order=per_view_before_mpr`.

Relevant implementation:

- `radio_gs/models/frozen_latent_capability_decoder.py`
- `radio_gs/scripts/train_frozen_latent_direct_capability_decoder.py`
- `radio_gs/scripts/materialize_scannet_direct_siglip_query_cache.py`
- `radio_gs/models/frozen_latent_relation_decoder.py`
- `radio_gs/scripts/train_evaluate_frozen_latent_relation_decoder.py`

The focused test set passes: `18 passed`.

## Remaining benchmark work

- LERF capability-before-MPR is closed as a rejected cross-task sentinel; the
  accepted identity/extent method is unchanged.
- ScanNet full-eight materialization and fixed evaluator are complete; the
  capability-before-MPR categorical compiler is promoted without changing
  the three-scene configuration.
- NVOS native box-region all-view full8 is complete (`0.81385` unconditional,
  `0.82624` with the pre-frozen component-local replay).  The compiler is
  complete, but unconditional source fusion and the replayed combination are
  not promoted over the retained target-only component-local row.
- SPIn fields remain on the persistent queue and do not block LERF, ScanNet or
  NVOS development.
- Exact/D512/D256 same-readout evidence already shows only about `0.00467`
  between the smaller compact field and exact MPR on the matched Figurines
  control, so capacity retraining is lower priority than capability order and
  object support.
