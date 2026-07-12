# ScanNet protocol alignment audit (2026-07-11)

This note separates the historical T3 diagnostic from two post-processing-free
Gaussian-center evaluations. It does **not** change the paper table or result
registry.

## Results on the same eight ScanNet scenes

All numbers are unweighted macros over
`scene0000_00, scene0062_00, scene0070_00, scene0097_00, scene0140_00,
scene0347_00, scene0400_00, scene0590_00`.

| Evaluation | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc | Status |
|---|---:|---:|---:|---|
| Historical T3 mesh-query diagnostic | 0.3806 / 0.6129 | 0.3871 / 0.6315 | 0.4711 / 0.7200 | Not VALA-comparable |
| Clean Gaussian-center, row-aligned labels | 0.2263 / 0.4072 | 0.2291 / 0.4150 | 0.2906 / 0.5190 | Diagnostic only |
| VALA Gaussian-domain protocol | **0.2076 / 0.3942** | **0.2107 / 0.4029** | **0.2699 / 0.5095** | Protocol-aligned result |

Sources:

- Historical T3: `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json`
- Clean row-aligned diagnostic: `output/scannet_pointcloud_eval/gaussian_index_clean_protocol_20260711/scannet_pointcloud_radio_gs_results.json`
- VALA Gaussian-domain result: `output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711/scannet_vala_gaussian_protocol_results.json`

The clean row-aligned row is reproduced inside the VALA evaluator as
`macro.opengaussian_row_unweighted`; its equality with the standalone clean run
is an implementation cross-check, not evidence that row alignment is the VALA
metric.

## Protocol differences

| Component | Historical T3 | Clean row-aligned diagnostic | VALA Gaussian-domain protocol |
|---|---|---|---|
| Prediction support | Annotated mesh/label-Ply vertices | Optimized Gaussian centers | Optimized Gaussian centers |
| Feature query | kNN-16 aggregation from 80 Gaussian candidates at each mesh vertex | Direct feature of the row-matched Gaussian | Direct feature of each Gaussian |
| Test-scene calibration | Scene-mean logit centering, alpha 0.45 | None | None |
| Spatial post-processing | Mesh kNN-12 logit smoothing, alpha 1.0 | None | None |
| Ground truth attached to | Original label-Ply row | Original label-Ply row | Optimized center via VALA anisotropic Mahalanobis-density vote (`radius_factor=5`, Euclidean candidate `k=1000`, class-balanced density, nearest fallback `k=1`) |
| Opacity/size treatment | Hard top-neighbour opacity filter at 0.1 | Hard row opacity filter at 0.1 | Continuous significance `sigmoid(opacity) * sx * sy * sz`; no hard filter |
| Metric measure | Unweighted mesh points | Unweighted row-matched Gaussians | Significance-weighted Gaussians |
| Scene aggregation | Unweighted scene macro | Unweighted scene macro | Unweighted scene macro |

The optimized centers move by roughly 2--3 cm on average relative to the input
rows. Across the eight scenes, VALA pseudo labels and raw row labels agree on
about 93.4% of rows on average, so the remaining clean-row/VALA gap is expected.
The historical T3 gap is much larger because it additionally changes the query
domain and applies test-scene calibration and spatial propagation.

## No semantic-GT label loss in the evaluated checkpoints

The training configuration does load `direct_point_pool_labels` from the label
PLY, but the evaluated v67 configurations set
`direct_point_text_loss_weight: 0.0`,
`direct_point_adapter_text_loss_weight: 0.0`, and
`direct_point_text_ce_weighting: none`. Their
`direct_point_sample_strategy: teacher_balanced` obtains sampling labels from
the RADIO/SigLIP2 teacher (`_direct_point_teacher_pseudo_labels`), while the
active pseudo-CE and distillation targets also come from teacher features.
Consequently, raw ScanNet category labels are not used by an active training
loss. The remaining alignment caveat is that training queries the row-aligned
label-Ply coordinates (`direct_point_gaussian_position_mode: label_point`),
whereas the VALA evaluation queries optimized Gaussian centers.

Relevant implementation locations:

- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0000_00.yaml`
- `radio_gs/training/feature_supervision_mixin.py` (`_subsample_direct_point_indices`, `_direct_point_teacher_pseudo_labels`, `_compute_direct_point_loss`)

## Reproduction command

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/root/RADIO-GS \
/root/miniconda3/envs/cybersim_agent/bin/python \
-m radio_gs.scripts.eval_scannet_vala_gaussian_protocol \
--scene_list 'scene0000_00,scene0062_00,scene0070_00,scene0097_00,scene0140_00,scene0347_00,scene0400_00,scene0590_00' \
--prepared_root dataset/scannet_og \
--config 'radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_{scene}.yaml' \
--checkpoint 'output/radio_gs/scannet_og_{scene}_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth' \
--output_dir output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711 \
--text_embedding_cache checkpoints/siglip2_scannet_text_embeddings_v67_knn.pt \
--prompt_templates '{query}' \
--feature_chunk_size 8192 \
--pseudo_chunk_size 512 \
--radius_factor 5 \
--candidate_k 1000 \
--fallback_k 1 \
--device cuda
```

Result JSON SHA-256:

```text
71f81b987769d62d9508b96bbbef41e0b3842796334bbd6eed17b156f2726214  output/scannet_pointcloud_eval/vala_gaussian_protocol_20260711/scannet_vala_gaussian_protocol_results.json
```

The evaluator is `radio_gs/scripts/eval_scannet_vala_gaussian_protocol.py`.
Its KD-tree candidate search was checked against VALA's released dense
`cdist + radius mask + topk` core on actual scene Gaussians, and its metric was
checked against VALA's released volume-aware metric implementation.
