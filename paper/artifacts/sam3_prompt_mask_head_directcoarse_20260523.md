# Prompt-Conditioned Internal SAM-Adaptor Mask Head

Date: 2026-05-23

This experiment implements a feature-only prompt-conditioned SAM-adaptor mask
head. Official SAM3 RGB masks are used only as training-view pseudo masks. At
evaluation time the readout uses rendered CTF-GS foundation features, SigLIP2
text embeddings, and the direct-3D coarse mask; it does not call the official
SAM3 RGB image encoder/decoder.

## Implementation

- Added `PromptConditionedMaskHead` in
  `radio_gs/models/prompt_conditioned_mask_head.py`.
- Added mapped official SAM3 cache provenance:
  `mask_query_indices` and `mask_query_ranks`.
- Added `train_prompt_conditioned_sam3_mask_head.py`.
- Added `sam3_prompt_mask_head` to LERF direct-3D evaluation.
- Best training uses direct-3D coarse masks saved from the same deployed
  compact-field protocol instead of using SAM3 target masks as the coarse prompt.

## Figurines Pilot

Protocol:

- Scene: `figurines`
- Direct-field scores:
  `output/radio_gs/score_cache/figurines_deployed_opacity_gate_only.pt`
- Selection: `score_threshold=0.35`
- Checkpoint:
  `output/radio_gs/prompt_sam3_mask_head_20260523/figurines_directcoarse_thr05/prompt_conditioned_sam3_mask_head.pth`
- Refinement: `sam3_prompt_mask_head`
- Best eval setting:
  `logit_threshold=-1.0`, `coarse_dilate=1`, geometry gate enabled,
  area ratio `[0.25, 1.8]`, minimum geometry boundary gain `0.0`.

| Setting | mIoU | Delta mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Delta Boundary-F | Trimap IoU | Delta Trimap | Accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial direct mask | 0.3746 | 0.0000 | 0.5893 | 0.4107 | 0.6109 | 0.0000 | 0.1367 | 0.0000 | 0.000 |
| Target-coarse prompt head, ungated | 0.2736 | -0.1010 | 0.4821 | 0.2143 | 0.5818 | -0.0292 | 0.0151 | -0.1216 | 0.786 |
| Direct-coarse prompt head, ungated | 0.3874 | 0.0128 | 0.6071 | 0.4464 | 0.6275 | 0.0166 | 0.1114 | -0.0254 | 0.768 |
| Direct-coarse prompt head, gated, thr=-0.5 | 0.3891 | 0.0145 | 0.5893 | 0.4464 | 0.6099 | -0.0010 | 0.1680 | 0.0313 | 0.232 |
| Direct-coarse prompt head, gated, thr=-1.0 | **0.3951** | **0.0205** | 0.5893 | **0.4643** | **0.6132** | **0.0023** | 0.1683 | 0.0315 | 0.286 |
| Direct-coarse prompt head, gated, thr=-1.5 | 0.3929 | 0.0183 | 0.5893 | 0.4464 | 0.6114 | 0.0004 | **0.1717** | **0.0350** | 0.250 |

## Conclusion

The naive target-coarse setup is invalid for the paper claim because it trains
with a perfect SAM3-derived coarse prompt and collapses under direct-3D coarse
masks. The direct-coarse setup is the correct method version: it distills
official SAM3 masks while conditioning on the same direct-field mask available
at inference.

On the Figurines pilot, the best gated direct-coarse prompt head improves mIoU
by `+0.0205`, Acc@0.50 by `+0.0536`, Boundary-F by `+0.0023`, and Trimap IoU by
`+0.0315` over the same initial direct-field mask. This is positive evidence for
an internal feature-only SAM boundary readout, but it is not yet a replacement
for the stronger RGB GrabCut promoted row. It should be reported as a feature-only
boundary module or expanded to all LERF scenes before becoming a main table row.
