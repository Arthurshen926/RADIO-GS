# ScanNet Open-Vocabulary 3D Query Qualitative

Binary query point-cloud visualization for the VALA/OpenGaFF-style direct point-query protocol.
The baseline is `VALA`; CTF-GS panels use saved ScanNet direct point-query predictions.

Figure: `paper/figures/scannet_openvocab_3d_query_qualitative.png`

| Scene | Query | VALA IoU | CTF-GS IoU | CTF-GS source |
| --- | --- | ---: | ---: | --- |
| scene0097_00 | cabinet | 0.2504 | 0.6079 | `output/scannet_pointcloud_eval/scene0097_00_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/visualizations/scene0097_00/pred_split_19.ply` |
| scene0062_00 | door | 0.3876 | 0.7495 | `output/scannet_pointcloud_eval/scene0062_00_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/visualizations/scene0062_00/pred_split_19.ply` |
| scene0590_00 | picture | 0.0361 | 0.4601 | `output/scannet_pointcloud_eval/scene0590_00_v67_dino_cv001_b2_s32768_ft20_gidx_labelpoint/visualizations/scene0590_00/pred_split_19.ply` |
