# SAM3 Training-View Proposal Registration Design

## Goal

Add a label-free object-aware direct-3D readout that uses official SAM3 masks on training views as object proposals, registers 3D Gaussians to those proposals, and fuses primitive scores with proposal-pooled scores during LERF Direct 3D Object Selection.

## Non-Goals

- Do not call official SAM3 on the query/evaluation RGB image.
- Do not use LERF ground-truth masks or query labels when building proposal memory.
- Do not replace the compact direct field with VPR as the main row.

## Architecture

The feature field still produces per-Gaussian direct scores. A new proposal registration module builds soft memberships between Gaussians and training-view SAM3 proposals. Each membership is weighted by projected mask probability, SAM3 confidence, and optional visibility confidence. During direct 3D querying, primitive scores are pooled inside object proposals and fused back into Gaussian scores with a confidence gate.

This keeps the paper claim unified: one compact foundation-feature Gaussian map supports 2D rendered query and 3D primitive query. SAM3 contributes only label-free object proposals from training views.

## Data Flow

1. Load official SAM3 foundation cache files for a LERF scene from `output/radio_gs/foundation_cache_sam3_modelscope_mapped_trainviews/<scene>/frame_*.pt`.
2. For every cached training view, use the existing LERF camera pose/intrinsics to project Gaussian centers to image pixels.
3. Sample SAM3 `mask_logits` at projected pixels and convert logits to probabilities.
4. Create soft memberships for confident Gaussian/proposal pairs.
5. Pool direct text scores over proposal memberships.
6. Fuse primitive and proposal scores:

   `final = (1 - alpha * gate) * primitive + (alpha * gate) * proposal`

   where `gate` is high for low-margin or low-confidence primitive scores and zero for unassigned Gaussians.

## Protocol

The LERF Direct 3D protocol remains unchanged. Text similarity and selection happen at Gaussian/primitive level. Rendering is used only to evaluate selected primitives against annotated 2D masks.

## Testing

- Unit-test projection-independent proposal membership construction.
- Unit-test proposal score pooling and confidence-gated fusion.
- Unit-test empty/missing cache handling.
- Add a CLI parser/integration test for the new evaluator flags.
- Run existing Direct3D, ScanNet, and paper-claim validation tests.

## Experiment Plan

First validate one-scene smoke on `figurines` with fixed threshold `thr0p25`, because existing strict pad16 SAM3-box baseline is available. If positive or diagnostically useful, run all four LERF scenes under the same Direct3D protocol and record a non-promoted ablation artifact unless it improves the promoted row.
