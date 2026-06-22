# Dense/Sparse SigLIP Supervision Ablation

Date: 2026-06-15

This artifact records a Ramen short-finetune ablation for the supervision story:

- dense rendered feature learning should be organized around RADIO reconstruction plus dense structural regularization;
- sparse primitive regularization can use SigLIP/MPR as a semantic anchor;
- dense rendered SigLIP/text losses should not be presented as the core feature-reconstruction objective unless the numbers justify it.

The experiment uses 8-epoch finetunes from the same Ramen warmstart. All variants keep the same dense SAM3-style structural cache supervision. The two switches are:

- **Dense SigLIP/text**: rendered-view grounding/query, summary alignment, and text heatmap losses.
- **Sparse SigLIP/MPR**: primitive-level direct point summary adapter and query/support consistency losses from the registered summary cache.

## Results

| Variant | Dense SigLIP/text | Sparse SigLIP/MPR | Train val cosine | Direct3D best thr | Direct3D mIoU | Direct3D Acc@0.25 | Direct3D B-F | 2D LocAcc | 2D mIoU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense0_sparse0 | no | no | 0.7871 | 0.10 | 0.2389 | 0.3099 | 0.3631 | 0.9014 | 0.6389 |
| dense1_sparse0 | yes | no | 0.7834 | 0.10 | 0.2239 | 0.3239 | 0.3511 | 0.8732 | 0.6416 |
| dense0_sparse1 | no | yes | 0.7799 | 0.05 | 0.3129 | 0.4225 | 0.4998 | 0.8451 | 0.5954 |
| dense1_sparse1 | yes | yes | 0.7822 | 0.05 | 0.2913 | 0.4085 | 0.4627 | 0.8451 | 0.6142 |

Teacher RADIO on this exact Ramen rendered-view protocol is 0.8873 LocAcc / 0.5937 mIoU.

## Interpretation

The Direct3D result is the clearest signal. Turning on sparse SigLIP/MPR improves direct-3D mIoU from 0.2389 to 0.3129 and Acc@0.25 from 0.3099 to 0.4225. Dense rendered SigLIP/text supervision alone does not reproduce this gain.

The rendered-view result remains strong without dense SigLIP/text losses: dense0_sparse0 reaches 0.9014 LocAcc / 0.6389 mIoU, above frame-wise RADIO reference under the same evaluation. Adding dense rendered SigLIP/text does not improve localization in this short run.

The sparse primitive regularizer has a visible 2D tradeoff at the tested weight. The sparse-only variant gives the best Direct3D result but lowers rendered-view grounding to 0.8451 / 0.5954. Adding dense SigLIP/text recovers some 2D mIoU but still underperforms the no-sparse 2D variants and reduces the Direct3D gain.

## Paper Consequence

This supports a cleaner method story:

1. Dense rendered supervision reconstructs RADIO-compatible scene features and dense structural properties.
2. SAM/DINO-style signals belong to rendered dense structural regularization because they encode mask boundaries, local topology, and correspondence-like structure.
3. SigLIP/MPR is better described as sparse primitive semantic usability regularization: it anchors primitive-level compact features in a text-aligned summary space without making the whole compact field a task-specific SigLIP field.

The current evidence argues against presenting dense rendered SigLIP/text losses as a core method component. If kept, they should be described as optional calibration. For the main narrative, use:

> Dense rendered RADIO feature reconstruction with dense structural regularization, plus sparse primitive semantic regularization through MPR/SigLIP summary anchors.

## Limitations

This is a single-scene Ramen, 8-epoch directional ablation, not a final four-scene table. The Direct3D numbers use the best value from a fixed global threshold sweep because these short-finetune checkpoints are not calibrated for the final promoted threshold. The result is sufficient for deciding method wording and next training design, but the final paper table should still use the frozen mainline protocol and four-scene results.

## Source Paths

- Configs: `/root/RADIO-GS/radio_gs/configs/generated/siglip_dense_sparse_ablation/`
- Checkpoints: `/root/RADIO-GS/output/radio_gs/lerf_ramen_siglip_*_ft8/checkpoints/best.pth`
- Direct3D outputs: `/root/RADIO-GS/output/radio_gs/siglip_dense_sparse_ablation/direct3d_ramen_sweep/`
- LERF 2D outputs: `/root/RADIO-GS/output/radio_gs/siglip_dense_sparse_ablation/lerf2d_ramen/`
- Machine-readable artifact: `/root/RADIO-GS/paper/artifacts/siglip_dense_sparse_ablation_20260615.json`
