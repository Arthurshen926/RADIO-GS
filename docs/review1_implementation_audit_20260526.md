# review1.md Implementation Audit - 2026-05-26

## Expert Priority Route

The latest expert feedback identifies the current ceiling as a representation/training
issue rather than a final readout issue. I therefore prioritized fixes that make the
main method trainable/evaluable under the intended routes:

1. P0: VPR/raster registration should become a label-free multiview teacher for the
   compact direct field, not an inference-time crutch.
2. P1: feature-only SAM boundary refinement must use the same prompt geometry at
   training and evaluation time.
3. P2: DINO mask propagation must use topology/cycle evidence in the propagated
   score, not only for visualization or optional seed cleanup.
4. P3: ScanNet proposal/object awareness should become a training objective, not
   only an evaluation-time score fusion.

## Implemented Fixes

### P0 - VPR-to-direct distillation interface

- `save_registered_feature_cache()` now writes `feature_space="siglip_summary"`,
  `feature_key="summary_features"`, and a deterministic geometry fingerprint.
- `audit_vpr_cache_alignment.py` reports whether the saved fingerprint matches
  the cache xyz payload.
- `train_feature_field.py` can now consume VPR registered caches with
  `summary_features` directly. It auto-detects SigLIP summary-space targets and
  fails fast if no summary/text/adapter objective is enabled.
- `FeatureSupervisionMixin._compute_direct_point_loss()` now treats summary-space
  teacher caches correctly: it skips 1280d RADIO distillation and applies direct
  summary/adapter/text losses against the VPR teacher summary.

This removes the previous implementation break where VPR exported
`summary_features` but training only accepted `features`.

### P1 - Direct-3D feature-only SAM prompt consistency

- `refine_mask_with_prompt_conditioned_sam3_head()` now resizes the coarse prompt
  to the prompt-head feature grid before applying `coarse_dilate`, matching the
  training-time policy.
- The refinement report now records `coarse_prompt_input_shape`,
  `coarse_prompt_shape`, and `coarse_prompt_resized`, enabling the requested
  small-mask/scale diagnostics.

This addresses the expert's suspected train/eval prompt-scale mismatch.

### P2 - DINO topology-aware propagation

- Added transported match evidence for DINO propagation:
  cycle/RANSAC-filtered matches can now be splatted into the target score map via
  `--dino_transport_match_weight` and `--dino_transport_match_radius`.
- This evidence directly changes the propagation heatmap and mask, so it can
  affect LocAcc and mIoU rather than only improving qualitative match drawings.

### P3 - Proposal/object-aware direct-field training

- Added `direct_point_proposal_consistency_weight`,
  `direct_point_proposal_voxel_size`, `direct_point_proposal_min_count`, and
  `direct_point_proposal_space`.
- The new loss builds deterministic voxel proposals over sampled direct points
  and pulls predicted summary features toward proposal prototypes.
- This is label-free and compatible with ScanNet/LERF direct-field training.

### ScanNet protocol cleanup

- `generate_scannet_dino_cv_configs.py` now defaults to the VALA8 scene set
  instead of the legacy 10-scene v67 list.
- `build_submission_tables.py` now explicitly marks legacy v67/10-scene results
  as internal diagnostics only.

## Verification

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest \
  tests/test_direct_point_supervision.py \
  tests/test_lerf_sam_dino_tasks.py \
  tests/test_vpr_cache_alignment.py -q
```

Result: `66 passed in 6.03s`.

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest \
  tests/test_lerf_direct_3d_selection.py -q
```

Result: `68 passed in 10.14s`.

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m py_compile \
  radio_gs/scripts/eval_lerf_direct_3d_selection.py \
  radio_gs/scripts/eval_lerf_sam_dino_tasks.py \
  radio_gs/scripts/train_feature_field.py \
  radio_gs/training/feature_supervision_mixin.py \
  radio_gs/scripts/audit_vpr_cache_alignment.py \
  radio_gs/scripts/generate_scannet_dino_cv_configs.py \
  radio_gs/scripts/build_submission_tables.py
```

## Next Experiments

1. Generate VPR registered feature caches with `registration_assignment_mode=center`
   and, separately, the raster modes for each LERF scene.
2. Fine-tune direct field with `direct_point_teacher_cache=<VPR cache>`,
   `direct_point_teacher_cache_feature_space=siglip_summary`,
   `direct_point_summary_adapter_weight>0`, and
   `direct_point_proposal_consistency_weight>0`.
3. Re-evaluate LERF direct 3D without VPR inference and compare against the current
   VPR row.
4. Re-run the DINO Teacher-vs-Ours benchmark with
   `--dino_match_mutual --dino_transport_match_weight {0.25,0.5,1.0}` and the
   existing feature-boundary refinement.
5. Re-run ScanNet VALA8 with the proposal-consistency trained checkpoints and
   keep split19/split15/split10 all paper-facing.

Long GPU experiments were not launched in this pass because all visible GPUs were
already at high utilization with non-visible processes, so killing them would risk
terminating external work.

## 2026-05-28 Follow-up

### Implementation Corrections

- Fixed the trainer initialization order for
  `radio_adaptor_peak_background_anchor_strategy`. The previous placement
  validated the field before copying it from the config, which caused new DINO
  topology runs to fail at startup with `AttributeError`.
- Extended the DINO peak/background run wrapper so future configs explicitly set
  `radio_adaptor_peak_background_anchor_strategy=distinctive` by default. The
  active log now confirms `peak_background_anchor_strategy=distinctive`.
- Added distinctive self-anchor selection for the peak/background adaptor loss.
  This focuses the DINO peak/background objective on teacher tokens with a clear
  top1-top2 self-similarity margin instead of fixed linspace anchors.

### Verification

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_radio_adaptor_loss.py \
  tests/test_radio_adaptor_trainer_config.py
```

Result: `26 passed in 6.75s`.

### New Completed Results

- LERF direct-3D support-distill, Figurines w0.20:
  `mIoU=0.2628`, `Acc@0.25=0.3750`, initial `mIoU=0.2241`,
  delta `+0.0387`. This fills the missing w0.20 Figurines evaluation and is
  clearly above the earlier w0.10 Figurines row (`mIoU=0.2166`).
- LERF direct-3D support-distill, Ramen w0.30 + proposal contrast:
  `mIoU=0.4156`, `Acc@0.25=0.5915`, initial `mIoU=0.3434`,
  delta `+0.0722`.
- ScanNet VALA8 v76 consensus-gated proposal readout completed:
  split19/15/10 mIoU = `0.3782 / 0.3843 / 0.4689`. This is slightly positive
  over v76 without consensus, but still below the promoted v67 DINO-CV spatial
  row (`0.3806 / 0.3871 / 0.4711`), so it should remain diagnostic.

### Running Experiments

- GPU0: `dino_peakbg_distinctive_maskprop_ft25_b4_figwaldo_20260528`
  is running with distinctive peak/background anchors and DINO topology losses.
- GPU1: direct-coarse-aware prompt-conditioned SAM3 mask-head evaluation is
  running. The initial score-threshold q0.55 Ramen result is negative; a queued
  follow-up will re-evaluate the same heads under the direct-3D top-ratio
  protocol before launching the Ramen/Teatime distinctive DINO run.

### Current Technical Read

- P0 is producing meaningful direct-field gains on the scenes evaluated so far,
  but the full four-scene w0.30 row is still pending.
- P1 remains unstable under score-threshold direct-3D evaluation; this supports
  review1's diagnosis that direct-3D SAM refinement is prompt/support limited
  and must be judged by initial-IoU buckets and protocol-matched top-ratio
  evaluation rather than promoted directly.
- P2 now has the missing distinctive peak/background anchor implementation and
  long runs are underway.

### Additional P1 Sanity Finding

The first P1 queue used the older
`adjoint_vpr_context_querydistill_rendercons_p0_ft20_mf16` checkpoint while the
strongest direct-3D rows currently come from the support-distilled direct field
(`supportdistill_zscore_w020/w030`). That mismatch makes the initial P1
score-threshold result a diagnostic of the old field, not the final promoted
direct field. A new aligned P1 run is now queued/running with:

- Ramen: `supportdistill_zscore_w030_propcontrast_p0_ft18_mf16`
- Figurines: `supportdistill_zscore_w020_p0_ft12_mf16`
- Evaluation: OpenGaussian-style direct-3D top-ratio sweep, not the earlier
  score-threshold sweep

## 2026-05-28 Second Follow-up

### Implementation Corrections

- Added a DINO local-neighborhood affinity objective:
  `compute_radio_adaptor_local_affinity_loss`. It matches frozen-adaptor cosine
  affinities for nearby spatial offsets, targeting review1's diagnosis that
  DINO propagation is limited by local correspondence topology rather than only
  peak selection or boundary refinement.
- Exposed the objective through training config:
  `radio_adaptor_local_affinity_names`,
  `radio_adaptor_local_affinity_weight`,
  `radio_adaptor_local_affinity_downsample`, and
  `radio_adaptor_local_affinity_radius`.
- Created a separate launcher,
  `tmp/run_lerf_dino_localtopo_ft_review1_20260528.sh`, so active DINO
  distinctive runs are not modified while they are being read by running
  processes.
- Fixed teacher-cache alignment audit robustness in
  `RadioGSTrainer._load_direct_point_teacher_cache`: optional Gaussian footprint
  tensors (`scales`, `rotations`, `opacities`) are now passed to the audit only
  when the model exposes the corresponding getter. This preserves the strict
  xyz alignment check while keeping lightweight VPR summary-cache tests valid.

### Verification

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_radio_adaptor_loss.py \
  tests/test_radio_adaptor_trainer_config.py
```

Result: `28 passed in 4.94s`.

Passed:

```bash
CUDA_VISIBLE_DEVICES= bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_lerf_direct_3d_selection.py \
  tests/test_lerf_sam_dino_tasks.py \
  tests/test_prompt_conditioned_mask_head.py \
  tests/test_prompt_conditioned_mask_head_training.py \
  tests/test_prompt_conditioned_mask_refinement.py \
  tests/test_direct_head_consistency.py \
  tests/test_vpr_cache_alignment.py \
  tests/test_direct_point_supervision.py \
  tests/test_direct_point_query_logit_distill_loss.py \
  tests/test_point_summary_adapter.py \
  tests/test_radio_adaptor_loss.py \
  tests/test_radio_adaptor_trainer_config.py
```

Result: `214 passed in 25.29s`.

### New Diagnostic Results

- Aligned P1 prompt-conditioned feature-only SAM3 direct-3D evaluation remains
  below the base support-distilled direct field:
  - Ramen: best `top0p04`, `mIoU=0.3524`, `Acc@0.25=0.5634`,
    initial `mIoU=0.3434`, delta `+0.0090`, accept rate `0.085`.
  - Figurines: best `top0p02`, `mIoU=0.2456`, `Acc@0.25=0.4286`,
    initial `mIoU=0.2241`, delta `+0.0215`, accept rate `0.089`.
  This is positive over each initial coarse mask but below the unrefined
  support-distilled direct-field rows, so it should remain a boundary diagnostic
  rather than a promoted direct-3D row.
- DINO distinctive completed Figurines:
  - SAM3 point: rendered `0.4598` vs teacher `0.4614`.
  - SAM3 box: rendered `0.6787` vs teacher `0.6743`.
  - SAM3 mask propagation: rendered `0.3620` vs teacher `0.3280`.
  - DINO dense matching: rendered hit `0.6318` vs teacher `0.7020`.
  - DINO mask propagation: rendered `0.4336` vs teacher `0.4907`.
  This confirms review1's warning that DINO still needs topology-preserving
  training; the new local-affinity runs are queued to test that hypothesis.

### Running Experiments

- GPU0: distinctive DINO Figurines/Waldo is running; local-topology
  Figurines/Waldo is queued immediately after it.
- GPU1: distinctive DINO Ramen/Teatime is running; local-topology
  Ramen/Teatime is queued immediately after it.
- GPU2: Teatime w0.30 support-distill + proposal contrast watcher is queued and
  will start when the visible memory drops below the safe threshold.
- GPU4/GPU5: Waldo/Figurines w0.30 support-distill watchers remain queued.

## 2026-05-28 Third Follow-up

### New Diagnostic Results

- Distinctive-anchor DINO finished Figurines and Waldo:
  - Macro over Figurines/Waldo: SAM3 box `0.6820` vs teacher `0.6746`,
    SAM3 mask propagation `0.3670` vs `0.3348`, and SAM3 point `0.4480`
    vs `0.4312`.
  - DINO dense matching hit rate is still lower on macro (`0.6357` vs
    `0.6963`), and DINO mask propagation remains lower (`0.4203` vs
    `0.4803`).
  - This is negative evidence for "peak/background anchors alone fix DINO";
    the local-affinity topology run has now started on GPU0.
- Distinctive-anchor DINO Ramen completed:
  - Dense matching hit `0.6582` vs teacher `0.6306`, but mask propagation
    `0.4494` vs teacher `0.4819`.
  - This supports the review diagnosis: matching/peak localization can improve
    without solving propagated object support.
- Distinctive-anchor DINO also finished Teatime:
  - SAM3 point `0.4595` vs teacher `0.3637`, SAM3 box `0.7169` vs `0.7040`.
  - SAM3 mask propagation remains slightly lower (`0.3521` vs `0.3803`).
  - DINO dense hit `0.6449` vs teacher `0.7092`; DINO mask propagation
    `0.5034` vs `0.5429`.
  - Macro over Ramen/Teatime: SAM3 rows are positive overall, but DINO mask
    propagation is still below teacher (`0.4732` vs `0.5088`).
- Aligned direct-coarse prompt-SAM bucket artifact was generated:
  `paper/artifacts/direct3d_initial_iou_buckets_prompt_sam3_aligned_support_20260528.md`.
  It shows feature-only SAM helps mainly when the initial direct mask already
  has reasonable support:
  - Figurines `0.50-0.75` initial-IoU bucket: `+0.0703` mIoU and
    `+0.0736` boundary-F.
  - Ramen `0.25-0.50` bucket: `+0.0397` mIoU and `+0.0625` boundary-F.
  - Ramen `<0.25` bucket is slightly negative (`-0.0030` mIoU), confirming
    that the boundary head is not an object-recovery module.

### Running Experiments

- GPU0: `dino_localtopo_distinctive_maskprop_ft25_b4_figwaldo_20260528`
  is running with `radio_adaptor_local_affinity_weight=0.02`.
- GPU1: `dino_localtopo_distinctive_maskprop_ft25_b4_ramenteatime_20260528`
  has started with `radio_adaptor_local_affinity_weight=0.02`.
- GPU2/GPU4/GPU5: this repo's `gpu_placeholder` reports not running, but
  `nvidia-smi` currently reports non-container-visible `[Not Found]` PIDs using
  about 8.5GB each with high utilization. The support-distillation watchers
  remain queued and should start only after those GPUs actually free.
