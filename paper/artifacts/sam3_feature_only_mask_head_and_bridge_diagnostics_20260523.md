# Feature-Only SAM3 Mask-Head and Decoder-Bridge Diagnostics

Date: 2026-05-23

This audit separates three SAM3-related boundary readouts:

1. Official SAM3 RGB readout: feature-derived coarse mask prompts frozen SAM3 on
   the evaluation RGB image. This is the strong assisted upper bound.
2. CTF-GS SAM3 mask-logit head: existing rendered features produce fixed
   feature-only mask candidates through a trained `foundation_cache_projectors`
   SAM3 projector.
3. RADIO/CTF-GS to official SAM3 decoder-state bridge: reconstructed features
   are projected into the official SAM3 `backbone_out` tensors, then the frozen
   official decoder is called without RGB image features.

## Implementation Fixes

- `scripts/gpu_placeholder.sh stop` now signals the placeholder process group.
  This prevents a stale CUDA worker from keeping GPU2 memory occupied after the
  wrapper reports that the placeholder has stopped.
- `radio_gs/scripts/train_sam3_decoder_bridge.py` now calls
  `torch.cuda.set_device(args.device)` before official SAM3 construction.
  Official SAM3 creates some tensors on hard-coded `cuda`, so the current CUDA
  device must be set before model construction.
- `train_sam3_decoder_bridge.py` now normalizes dtype aliases:
  `bf16 -> bfloat16`, `fp32 -> float32`, and `none -> off`.

## SAM3 Mask-Logit Head Probe

Setting:

- Direct 3D selection uses the deployed compact direct-field score cache for
  Figurines.
- The mask refinement branch is `--mask_refinement sam3_mask_head`.
- The mask head checkpoint is
  `output/radio_gs/lerf_figurines_official_sam3_boundary_ft/checkpoints/best.pth`.
- No official SAM3 RGB readout is called during evaluation.

Strict default gate:

| Scene | Initial mIoU | Mask-head mIoU | Delta | Initial B-F | Mask-head B-F | Accepted |
|---|---:|---:|---:|---:|---:|---:|
| Figurines | 0.4205 | 0.4205 | +0.0000 | 0.5983 | 0.5983 | 0 / 56 |

Relaxed overlap gate sweep:

| Logit threshold | Min initial IoU | mIoU | Delta mIoU | Boundary-F | Delta B-F | Accepted |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.0 | 0.0315 | -0.3890 | 0.0949 | -0.5034 | 47 / 56 |
| 0.25 | 0.0 | 0.0005 | -0.4201 | 0.0152 | -0.5831 | 47 / 56 |
| 0.50 | 0.0 | 0.0000 | -0.4205 | 0.0013 | -0.5970 | 47 / 56 |

Interpretation: the current mask-logit head is not a prompt-conditioned
boundary refiner. Under the safe overlap gate it falls back to the initial mask;
when forced to accept candidates, it destroys the masks. It should not be used
as a main feature-only SAM3 result.

## Official Decoder-State Bridge Diagnostic

After fixing device and dtype aliases, the minimal teacher-feature diagnostic
runs with:

```bash
--source teacher
--scene figurines
--epochs 1
--max_train_frames 1
--max_eval_frames 1
--hidden_dim 32
--sam3_dtype float32
--sam3_amp_dtype bfloat16
```

Result:

| Source | Official RGB SAM3 mIoU | RADIO-to-SAM3 bridge mIoU | Queries |
|---|---:|---:|---:|
| Teacher RADIO feature | 0.4349 | 0.0000 | 17 |

Interpretation: this short diagnostic is not a full bridge training run, but it
does confirm the expert concern: RADIO/SAM3-adaptor features are not already on
the official SAM3 image-encoder manifold. A stronger feature-only SAM3 boundary
module should therefore be a prompt-conditioned internal mask head distilled
from official SAM3 training-view masks, not the current fixed candidate projector
or an untrained official decoder-state emulator.

