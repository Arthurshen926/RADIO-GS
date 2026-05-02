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

Stage 2 should add FMGS-style DINO pixel alignment:

```text
For each pixel p and local neighborhood N(p):
S_dino(p, q) = dot(norm(dino(p)), norm(dino(q)))
S_siglip_or_radio(p, q) = dot(norm(feature(p)), norm(feature(q)))
loss = mean | S_siglip_or_radio - stopgrad(S_dino) |
```

This should be delayed until Stage 1 ablations show that adaptor consistency is
stable, because local-neighborhood losses can be expensive and sensitive to
feature resolution.

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

The initial implementation only adds the frozen SAM3 adaptor consistency loss.
That is the correct first step because it verifies whether reconstructed RADIO
features preserve SAM3-compatible information before adding external mask
generation complexity.

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

