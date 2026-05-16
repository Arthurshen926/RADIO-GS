# Official SAM3 Cache Status, 2026-05-15

## Checkpoint Provenance

The official SAM3 code path is now usable with the local ModelScope mirror
checkpoint:

- checkpoint: `checkpoints/sam3_modelscope/sam3.pt`
- source: `https://www.modelscope.cn/models/facebook/sam3/files`
- SHA256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`
- config SHA256:
  `4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f`
- code: `facebookresearch/sam3`, loaded through
  `sam3.model_builder.build_sam3_image_model`

The paper wording should be:

> official SAM3 code with the public ModelScope mirror of the SAM3 checkpoint

not:

> downloaded from the official HuggingFace gated repository

unless HuggingFace access is later approved.

## Builder Updates

`radio_gs/scripts/build_sam3_foundation_cache.py` now supports:

- `--checkpoint_path`: local official checkpoint path;
- `--checkpoint_source` and automatic `--checkpoint_sha256 auto` provenance;
- `--frame_ids`: comma-separated frame stems or numeric LERF frame ids;
- `--skip_existing`: resume cache generation safely;
- `--amp_dtype auto|off|bfloat16`: use BF16 autocast for official CUDA
  inference while keeping official weights in FP32;
- `--resolution`: kept at the official 1008 by default.

The cache payload records official producer metadata and can be loaded with
`foundation_cache_require_official: true`.

## Runtime Findings

Two failed smoke tests are useful diagnostics:

- CUDA with the model forcibly cast to BF16 fails at the first convolution
  because the processor image tensor is FP32. The correct path is FP32 weights
  plus BF16 autocast around `Sam3Processor.set_image` and `set_text_prompt`.
- Reducing the processor resolution to 336 triggers a ViT RoPE shape assertion.
  The current official image model should be run at 1008 resolution unless a
  separate official low-resolution configuration is provided.

CPU execution is not a reliable fallback for full grounding. Even after BF16
autocast, the official grounding path mixes CPU and CUDA tensors internally.
The practical requirement is a GPU with enough free memory for the 1008
resolution model.

## Generated LERF Caches

Official SAM3 caches have been generated and strict-loaded for all annotated
LERF-OVS frames used by the direct-selection/grounding experiments:

| Scene | Frames | Cache root |
|---|---:|---|
| Figurines | 4 | `output/radio_gs/foundation_cache_sam3_modelscope/figurines` |
| Ramen | 7 | `output/radio_gs/foundation_cache_sam3_modelscope/ramen` |
| Teatime | 6 | `output/radio_gs/foundation_cache_sam3_modelscope/teatime` |
| Waldo Kitchen | 5 | `output/radio_gs/foundation_cache_sam3_modelscope/waldo_kitchen` |

Every cache is loadable with `require_official=True` and reports:

- backend: `facebookresearch/sam3`;
- decoder: `Sam3Image+Sam3Processor`;
- source checkpoint SHA256:
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.

## Training Integration

The official cache is now connected to training through a trainable low-res
`FoundationMaskLogitProjector`. The projector receives decoded RADIO-compatible
feature maps and predicts a fixed bank of SAM3 mask-logit channels. The
supervision helper aligns variable official SAM3 mask counts and resizes
full-resolution official logits to the feature-map resolution before applying
mask-logit and boundary-response losses.

Current fine-tune overlays:

- `radio_gs/configs/lerf_hybrid_v14_figurines_official_sam3_boundary_ft.yaml`
- `radio_gs/configs/lerf_hybrid_v14_ramen_official_sam3_boundary_ft.yaml`
- `radio_gs/configs/lerf_hybrid_v14_teatime_official_sam3_boundary_ft.yaml`
- `radio_gs/configs/lerf_hybrid_v14_waldo_kitchen_official_sam3_boundary_ft.yaml`

The active supervision block is:

```yaml
foundation_cache_root: output/radio_gs/foundation_cache_sam3_modelscope/figurines
foundation_cache_heads: sam3
foundation_cache_weight: 0.15
foundation_cache_mask_logit_weight: 0.02
foundation_cache_mask_boundary_weight: 0.08
foundation_cache_mask_projector_masks: 24
foundation_cache_require_official: true
```

## Current Runtime Blocker

After two failed warmstart attempts, GPU4 and GPU5 retain stale host-side CUDA
contexts reported by `nvidia-smi` as `[Not Found]`. GPU reset from inside this
container fails with insufficient permissions, and GPU2 still reports ~23 GB
used. Training is therefore blocked on actual GPU availability, not on SAM3
loading or cache validity.

The code-side issue that triggered the failed warmstarts has been fixed:
`load_checkpoint(..., resume=False)` now loads checkpoint tensors on CPU first,
avoiding an unnecessary GPU deserialization peak.
