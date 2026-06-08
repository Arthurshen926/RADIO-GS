# TPAMI Large-Asset Release Manifest

Date: 2026-06-01

This manifest lists the large files and external datasets needed to rerun the
paper-facing CTF-GS results. These assets are intentionally not stored under
`paper/artifacts/`, which only snapshots small result files, tables, manifests,
and provenance reports.

## Release Units

| Unit | Required for | Release status | Notes |
| --- | --- | --- | --- |
| Paper source package | manuscript/supplement build | in repository | `paper/radio_gs_tpami.tex`, `paper/radio_gs_tpami_supplement.tex`, `paper/radio_gs_refs.bib`, `paper/IEEEtran.*`, `paper/figures/` |
| Small result artifacts | claim/provenance audit | in repository | `paper/artifacts/final_rows.yaml`, `paper/artifacts/checksums.txt`, `paper/artifacts/paper_assets_manifest.json` |
| LERF-OVS labels | T1/T2 evaluation | external dataset mirror required | Current local root: `/mnt/pool/sqy/3d_understanding/lerf_ovs/label` |
| LERF CTF-GS checkpoints | T1/T2 feature rendering and direct primitive scoring | large-asset upload required | Four scene checkpoints listed below |
| Feature-only SAM3-adaptor mask heads | T1 rendered-view boundary readout | large-asset upload required | Four scene mask-head checkpoints listed below |
| ScanNet VALA/OpenGaFF-8 prepared data | T3 point-query evaluation | external dataset mirror required | Current local root: `dataset/scannet_og` |
| ScanNet CTF-GS checkpoints | T3 point-query evaluation | large-asset upload required | Eight scene checkpoints follow the pattern listed below |
| Frozen RADIO/SigLIP2 assets | T1/T2/T3 text-aligned feature scoring | redistribute if license permits, otherwise document download | Projection/head/text-cache files listed below |
| Official SAM3 checkpoint | supplementary SAM3-box controls only | optional large-asset upload or download instruction | Not required for the core compact direct-field row |
| Evaluation outputs/masks | figure regeneration and paper audit | optional large-asset upload | Useful for exact figure regeneration without rerunning evaluation |

## Core Small Artifacts

These files are part of the repository snapshot and are verified by
`sha256sum -c paper/artifacts/checksums.txt`:

- `paper/artifacts/final_rows.yaml`
- `paper/artifacts/paper_assets_manifest.json`
- `paper/artifacts/tpami_reproducibility_package_20260601.md`
- `paper/artifacts/tpami_readiness_audit_20260601.md`
- `paper/artifacts/figure_quality_audit_tpami_20260531.md`
- `paper/artifacts/lerf_rendered_grounding_feature_sam3_boundary_20260525.json`
- `paper/artifacts/lerf_direct3d_score_component_guard_20260528.json`
- `paper/artifacts/scannet_pointcloud_radio_gs_vala8_dino_cv_contextual_knn16_cand80_scene_mean_a045_spatial_smoothk12a1_results.json`

## Frozen Foundation Assets

| Asset | Current path | Use |
| --- | --- | --- |
| C-RADIOv4-H checkpoint | `/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar` | RADIO teacher and adaptor features |
| SigLIP2 feature projection | `checkpoints/siglip2_feat_projection.pth` | Map reconstructed RADIO features to text-aligned summary space |
| SigLIP2 summary head | `checkpoints/siglip2_summary_head.pth` | Text-aligned feature summary head |
| LERF text embedding cache | `checkpoints/siglip2_lerf_text_embeddings.pt` | T1 rendered-view text queries |
| LERF prompt-ensemble cache | `checkpoints/siglip2_lerf_text_embeddings_promptens_20260514.pt` and scene-specific prompt caches | T2 prompt-ensemble direct 3D selection |
| ScanNet text embedding cache | `checkpoints/siglip2_scannet_text_embeddings_v67_knn.pt` plus split-specific caches | T3 ScanNet query classes |
| Official SAM3 checkpoint | `checkpoints/sam3_modelscope/sam3.pt` | Supplementary SAM3-box boundary controls |

The current large local file sizes are approximately 3.21 GiB for
`sam3.pt`, 84 MiB for `siglip2_feat_projection.pth`, and 35 MiB for
`siglip2_summary_head.pth`.

## T1 LERF Rendered-View Assets

The promoted T1 result is registered in
`paper/artifacts/final_rows.yaml` as `ctfgs_rendered` and is sourced from
`paper/artifacts/lerf_rendered_grounding_feature_sam3_boundary_20260525.json`.

| Scene | Config | CTF-GS checkpoint | Feature-only SAM3-adaptor mask head |
| --- | --- | --- | --- |
| Figurines | `radio_gs/configs/lerf_hybrid_v14_figurines_fdh_ws240_240ep.yaml` | `output/radio_gs/lerf_figurines_v14_fdh_ws240_240ep/checkpoints/latest.pth` | `output/radio_gs/prompt_sam3_mask_head_20260523/figurines_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth` |
| Ramen | `radio_gs/configs/lerf_hybrid_v14_ramen_fdh_ws240_240ep.yaml` | `output/radio_gs/lerf_ramen_v14_fdh_ws240_240ep/checkpoints/latest.pth` | `output/radio_gs/prompt_sam3_mask_head_20260523/ramen_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth` |
| Teatime | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_teatime_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_teatime_v14_fdh_ws240_240ep_seed7/checkpoints/best.pth` | `output/radio_gs/prompt_sam3_mask_head_20260523/teatime_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth` |
| Waldo Kitchen | `radio_gs/configs/generated/seeds/lerf_hybrid_v14_waldo_kitchen_fdh_ws240_240ep_seed7.yaml` | `output/radio_gs/lerf_waldo_kitchen_v14_fdh_ws240_240ep_seed7/checkpoints/latest.pth` | `output/radio_gs/prompt_sam3_mask_head_20260523/waldo_kitchen_trainviews_lerf2dcoarse_e60_cache/prompt_conditioned_sam3_mask_head.pth` |

The exact per-scene evaluation outputs used by the registry are:

- `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/figurines_peakinit_T50/lerf_ovs_results.json`
- `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/ramen_peakinit_base_lerf2dcoarse/lerf_ovs_results.json`
- `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/teatime_peakinit_T25/lerf_ovs_results.json`
- `output/radio_gs/lerf2d_heatmap_guard_sam3_20260525/waldo_kitchen_peakinit_T25/lerf_ovs_results.json`

## T2 LERF Direct 3D Assets

The promoted T2 compact direct row is
`ctfgs_compact_prompt_ensemble_score_component_guard_thr0p55`. It uses the same
LERF CTF-GS checkpoints above, direct Gaussian-center primitive scores, a frozen
SigLIP2 prompt ensemble, and a GT-free RGB/score-component support policy. It
does not read a VPR feature cache and does not invoke an official RGB SAM
decoder at inference.

Required evaluation output root for exact figure/table regeneration:

```text
output/radio_gs/lerf_direct3d_prompt_ensemble_score_component_guard_m050_k2_lowthr_20260528
```

Optional diagnostic roots that should be uploaded only if reproducing
supplementary VPR/SAM3-box controls:

- `output/radio_gs/vpr_feature_cache/`
- `output/radio_gs/vpr_registered_feature_cache_20260516/`
- `output/radio_gs/lerf_sam3_box_*`

These optional VPR caches can be several GiB per scene and are not needed for
the core compact direct-field row.

## T3 ScanNet VALA/OpenGaFF-8 Assets

The promoted T3 result is registered as
`radio_gs_dino_cv_contextual_knn_scene_mean_support`. Required local dataset
root:

```text
dataset/scannet_og
```

Required scene subset:

```text
scene0000_00
scene0062_00
scene0070_00
scene0097_00
scene0140_00
scene0347_00
scene0400_00
scene0590_00
```

Checkpoint and config patterns:

```text
radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_{scene}.yaml
output/radio_gs/scannet_og_{scene}_v67_dino_cv001_b2_s32768_ft20/checkpoints/best.pth
```

Exact evaluation output root:

```text
output/scannet_pointcloud_eval/vala8_dino_cv_knn16_cand80_scene_mean_a045_smoothk12a1_20260524
```

## Suggested Release Layout

```text
ctfgs_tpami_release/
  paper/
  paper_artifacts/
  checkpoints/
    siglip2_feat_projection.pth
    siglip2_summary_head.pth
    siglip2_*_text_embeddings*.pt
    sam3_modelscope/sam3.pt                 # optional diagnostic
  lerf/
    labels/
    ctfgs_checkpoints/
    sam3_adaptor_mask_heads/
    eval_outputs/
  scannet_vala8/
    prepared_data/
    ctfgs_checkpoints/
    eval_outputs/
  README.md
  SHA256SUMS
```

## Release Verification Commands

After staging a large-asset release, generate checksums from the release root:

```bash
find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

Then rerun the paper-package guards from the repository root:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python radio_gs/scripts/validate_final_rows_registry.py
/root/miniconda3/envs/cybersim_agent/bin/python radio_gs/scripts/validate_paper_claims.py --root /root/RADIO-GS
sha256sum -c paper/artifacts/checksums.txt
```

## Do Not Upload As Core Assets

The following files are useful local diagnostics but should not be described as
required for the main compact-field claims:

- exploratory VPR caches under `output/radio_gs/vpr_feature_cache/`;
- scene-locked or oracle threshold sweeps;
- official RGB SAM3 box-readout masks for the core direct-3D row;
- failed external-baseline intermediate training outputs;
- GPU queue placeholders and temporary launcher logs.
