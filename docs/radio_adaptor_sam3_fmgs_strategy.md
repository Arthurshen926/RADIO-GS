# RADIO-GS DINOv3/SAM3 Adaptor Strategy

Date: 2026-05-02

## Current Status

RADIO-GS reconstructs `C-RADIOv4-H` 1280d backbone features as the main scene
feature target. The existing open-vocabulary pipeline explicitly uses the
`siglip2-g` RADIO adaptor through frozen SigLIP2 projection/summary heads for
text grounding and ScanNet text-query evaluation.

C-RADIOv4 also exposes `dino_v3` and `sam3` adaptors. The first implementation
step keeps the output representation unchanged and adds optional frozen adaptor
consistency:

```yaml
radio_adaptor_alignment_names: dino_v3,sam3
radio_adaptor_alignment_weight: 0.05
radio_adaptor_alignment_kind: feature_projection
radio_adaptor_alignment_checkpoint: /root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar
```

When enabled, RADIO-GS projects both decoded rendered features and frozen RADIO
teacher features through the selected adaptor projections, then minimizes
cosine distance in adaptor space. The feature remains a 1280d RADIO-like map at
inference time.

## FMGS Connection

FMGS trains a 3D Gaussian/hash feature field from 2D foundation-model features
and adds DINO-driven pixel alignment to make language features follow sharper
object boundaries. In the FMGS paper, DINO features are used alongside CLIP
features, and a dot-product similarity loss transfers DINO's spatial boundary
structure to the rendered language feature field.

RADIO-GS can reuse that idea without changing the main claim:

- `siglip2-g` remains the text-aligned evaluator for LERF and ScanNet text
  queries.
- `dino_v3` acts as the boundary/detail adaptor, similar to FMGS's DINO
  regularizer.
- `sam3` acts as the region/mask adaptor, closer to SAM-guided 3DGS methods
  that use 2D masks to stabilize object regions and boundaries.

Sources:

- FMGS arXiv page: https://arxiv.org/abs/2401.01970
- RADIO repository adaptor list: https://github.com/NVlabs/RADIO
- Segment Any 3D Gaussians / SAM-style 3DGS precedent:
  https://arxiv.org/abs/2312.00860

## Training Path

Stage 1 is implemented as frozen adaptor feature consistency:

```text
decoded RADIO-GS 1280d feature -> frozen dino_v3/sam3 adaptor -> adaptor feature
teacher RADIO 1280d feature    -> frozen dino_v3/sam3 adaptor -> adaptor feature
loss = mean(1 - cosine(pred_adaptor, teacher_adaptor))
```

This is low risk because it does not alter:

- HCD codec output dimension.
- LERF/ScanNet SigLIP2 evaluation protocol.
- Checkpoint compatibility when the new weight is zero.

Stage 2 is now implemented as FMGS-style DINO relation alignment:

```text
decoded RADIO-GS 1280d feature -> frozen dino_v3 adaptor -> D_pred
teacher RADIO 1280d feature    -> frozen dino_v3 adaptor -> D_ref
S_pred(i, j) = dot(norm(D_pred_i), norm(D_pred_j))
S_ref(i, j)  = dot(norm(D_ref_i), norm(D_ref_j))
loss = mean squared error(S_pred, stopgrad(S_ref))
```

The active low-risk configuration uses deterministic token subsampling to keep
the pairwise similarity matrix bounded:

```yaml
radio_adaptor_relation_names: dino_v3
radio_adaptor_relation_weight: 0.02
radio_adaptor_relation_downsample: 2
radio_adaptor_relation_max_tokens: 384
radio_adaptor_relation_temperature: 1.0
```

## SAM3 Mask-Prior Path

SAM3 should not be treated as another text adaptor. Its best role is region
structure:

1. Extract SAM3 adaptor features during RADIO feature extraction or project
   decoded 1280d features through the frozen SAM3 adaptor during training.
2. Build optional 2D mask priors from SAM/SAM3 automatic masks or dataset masks.
3. Add mask-aware losses:
   - within-mask feature compactness,
   - between-mask feature separation,
   - boundary-aware feature sharpness near mask edges,
   - cross-view mask consistency when the same 3D Gaussians project into
     multiple views.

The current implementation adds a mask-free SAM3 soft-region loss. Teacher
SAM3 adaptor features define deterministic anchor tokens; each pixel is softly
assigned to anchors by SAM3 feature similarity, and RADIO-GS is trained to
match the same region prototypes:

```yaml
radio_adaptor_region_names: sam3
radio_adaptor_region_weight: 0.01
radio_adaptor_region_downsample: 2
radio_adaptor_region_max_tokens: 384
radio_adaptor_region_num_anchors: 24
radio_adaptor_region_temperature: 0.07
```

This is the feasible FMGS/SAM-style path in the current environment because no
`segment_anything`, `sam2`, `sam3`, or local SAM mask weights are installed.
External SAM/SAM3 mask caches remain the next step for true mask-boundary
supervision.

## Full-Experiment Status

The adaptor branches are treated as full LERF-OVS ablations, not smoke tests.
The current paper rule is conservative: a branch is only promoted when it keeps
localization accuracy and improves mIoU.

Completed full sweeps:

| Scene | Branch | LocAcc | mIoU | Decision |
|---|---|---:|---:|---|
| Figurines | baseline | 0.8214 | 0.4308 | keep |
| Figurines | DINO relation + SAM3 region | 0.8036 | 0.4399 | diagnostic: region improves, localization drops |
| Figurines | lower-weight DINO relation + SAM3 region | 0.7857 | 0.4278 | failed follow-up; do not promote |
| Figurines | DINO cross-view + spatial text heatmap | 0.8214 | 0.4343 | promote: preserves LocAcc and improves mIoU |
| Ramen | baseline | 0.9014 | 0.5862 | superseded |
| Ramen | DINO relation + SAM3 region | 0.9014 | 0.5873 | promote for adaptor ablation |
| Teatime | baseline | 0.8983 | 0.5486 | superseded |
| Teatime | DINO + SAM3 pointwise | 0.8983 | 0.5582 | superseded by relation/region |
| Teatime | DINO relation + SAM3 region | 0.8983 | 0.5592 | promote for adaptor ablation |
| Waldo Kitchen | baseline | 0.8636 | 0.4106 | keep |
| Waldo Kitchen | DINO + SAM3 pointwise | 0.8636 | 0.4103 | do not promote |
| Figurines | DINO cross-view + query text heatmap | 0.8036 | 0.4381 | diagnostic: mIoU improves, LocAcc still drops |
| Waldo Kitchen | DINO cross-view + text heatmap | 0.6818 | 0.4563 | diagnostic: high-T mIoU improves, LocAcc drops sharply |

Completed diagnostic branches:

- Ramen DINO-only: mIoU is slightly higher, but LocAcc drops from 0.9014 to
  0.8873, so it is not paper-main.
- Ramen SAM3-only: both mIoU and LocAcc trail the combined/rel-region branch.
- Waldo relation/region: does not improve the frozen baseline.

Current adaptor-enhanced selector:

- Use the DINO cross-view + spatial text-heatmap checkpoint for Figurines.
- Use DINO relation + SAM3 region checkpoints for Ramen and Teatime.
- Macro LocAcc stays at 0.8712, while macro mIoU improves from 0.4941 to
  0.4979.

The only follow-up branch that currently beats the frozen Figurines baseline
under the promotion rule is DINO cross-view with spatial text-heatmap
distillation. Plain relation/region and query-distribution text distillation
improve thresholded overlap but still move the argmax on small objects.

The ProFuse-inspired DINO cross-view branch was tested as a lightweight
optimization-time analogue of cross-view context fusion: it matches DINOv3
cross-view token affinities and adds SigLIP text-heatmap distillation to protect
text peaks. It does not replace ProFuse's full dense pre-registration and
3D-context-proposal pipeline. Empirically, it improves overlap at high
temperature in some cases but still loses the argmax localization metric, so it
remains diagnostic.

The first text-heatmap implementation matched the query distribution at each
pixel. That protects scene-category logits but does not directly preserve the
spatial peak for each query, which is what LERF LocAcc measures. The trainer now
also supports `text_heatmap_distill_mode: spatial` and `query_spatial`. The
conservative `spatial` branch is positive on Figurines: at T50 it keeps LocAcc
at 0.8214 and improves mIoU from 0.4308 to 0.4343. The stronger
`query_spatial` branch reaches 0.4383 mIoU but drops LocAcc to 0.8036, so it is
diagnostic only. Waldo remains unresolved: high-temperature overlap improves,
but LocAcc stays below baseline.

## ScanNet DINO Cross-View Diagnostics

Two ScanNet scenes were evaluated with ProFuse-inspired DINOv3 cross-view
affinity regularization after v67 teacher-balanced warm-starts:

| Scene | Branch | split19 mIoU | split15 mIoU | split10 mIoU | Decision |
|---|---|---:|---:|---:|---|
| scene0070_00 | baseline | 0.2297 | 0.2405 | 0.3238 | superseded in diagnostic |
| scene0070_00 | DINO cv, weight 0.001 | 0.2437 | 0.2466 | 0.3284 | positive diagnostic |
| scene0070_00 | DINO cv, weight 0.003 | 0.2289 | 0.2297 | 0.3226 | too strong |
| scene0645_00 | baseline | 0.2381 | 0.2458 | 0.2875 | compare |
| scene0645_00 | DINO cv, weight 0.003 | 0.2427 | 0.2500 | 0.2833 | improves 19/15, lowers 10 |

Interpretation: DINO cross-view helps ScanNet when the weight is conservative,
but the 10-class split can lose coarse-class separation if the relation signal
is too strong. The next ScanNet mainline candidate should sweep
`radio_adaptor_cross_view_weight` around `0.0005-0.001` on the full 10-scene
set before replacing the frozen v67 table.

## Downstream Adaptor Probes

The downstream DINOv3/SAM3 probes are now implemented and run on all four
LERF-OVS scenes:

- Script: `radio_gs/scripts/eval_lerf_adaptor_downstream.py`
- Aggregate report:
  `output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`
- Qualitative figure:
  `paper/figures/lerf_adaptor_downstream_qualitative.png`

Macro results:

| Adaptor | Probe | Teacher LocAcc | Teacher mIoU | Rendered LocAcc | Rendered mIoU |
|---|---|---:|---:|---:|---:|
| DINOv3 | prototype segmentation | 0.6543 | 0.0945 | 0.6277 | 0.0937 |
| DINOv3 | source-target matching | 0.5957 | 0.1032 | 0.5035 | 0.1019 |
| SAM3 | prototype segmentation | 0.8404 | 0.0757 | 0.6649 | 0.0564 |
| SAM3 | source-target matching | 0.7872 | 0.0953 | 0.7092 | 0.0687 |

Additional promptable task probes:

| Task | Teacher LocAcc/Hit | Teacher mIoU/Score | Rendered LocAcc/Hit | Rendered mIoU/Score |
|---|---:|---:|---:|---:|
| SAM3 point prompt segmentation | 1.0000 | 0.2304 | 1.0000 | 0.1210 |
| SAM3 box prompt segmentation | 0.8606 | 0.0910 | 0.7885 | 0.0614 |
| SAM3 mask prompt propagation | 0.7872 | 0.0953 | 0.7092 | 0.0687 |
| DINOv3 dense matching | 0.5723 | 0.8547 | 0.5396 | 0.9063 |

Qualitative figure:
`paper/figures/lerf_sam_dino_tasks_qualitative.png`

Interpretation:

- These probes are diagnostic, not a new claimed aggregate improvement.
- DINOv3 rendered features are close to teacher mIoU, but still lower on macro
  LocAcc.
- SAM3 shows a larger teacher-rendered gap, which is consistent with the
  current SAM3 training path being a soft-region approximation rather than true
  SAM mask supervision.
- Waldo Kitchen has useful positive cases: rendered DINOv3 matching improves
  from 0.2500 to 0.5000 LocAcc and from 0.2708 to 0.2948 mIoU; rendered SAM3
  matching improves mIoU from 0.2965 to 0.3147 at unchanged LocAcc.

Main improvement ideas:

1. Add a peak-preservation term so DINO/SAM region smoothing cannot move the
   SigLIP or adaptor heatmap argmax away from small masks.
2. Replace the current SAM3 soft-region anchors with cached SAM/SAM3 masks when
   a stable mask generator is available.
3. Use mask-aware object weighting for small objects and thin structures before
   applying relation/region losses.
4. Batch downstream prototype scoring on GPU for faster sweep iterations.

## Evaluation Path

For the paper claim that RADIO-GS reconstructed novel-view features can be more
useful than raw RADIO features on rendered RGB, use this comparison:

```text
GT RGB -> RADIO adaptor                  oracle upper bound
3DGS rendered RGB -> RADIO adaptor        render-RGB-then-encode baseline
RADIO-GS rendered feature -> adaptor      ours
```

The strongest claim is not that RADIO-GS beats the GT RGB teacher. The stronger
and fairer claim is that direct feature rendering can beat the baseline that
first renders RGB from 3DGS and then re-encodes that rendered RGB through RADIO.

Recommended tasks:

- SigLIP2: LERF text grounding and ScanNet text-query mIoU.
- DINOv3: point-query semantic transfer and feature correspondence consistency.
- SAM3: mask boundary F-score, region compactness/separation, and optional
  SAM-style prompt/automatic-mask agreement.
