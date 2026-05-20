# ScanNet DINO Cross-View Launch Plan

- Variant: `v67_dino_cv001_b2_s32768_ft20`
- Protocol: v67 teacher-balanced, gaussian_index, label_point, label_index
- Batch size: 2
- Direct point samples: inherited from v67 configs (`32768`)
- Cross-view adaptor: `dino_v3`, weight `0.001`

## Suggested GPU Split

- GPU4 scenes: `scene0000_00, scene0070_00, scene0140_00, scene0347_00, scene0590_00`
- GPU5 scenes: `scene0062_00, scene0097_00, scene0200_00, scene0400_00, scene0645_00`

## Configs

- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0000_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0062_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0070_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0097_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0140_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0200_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0347_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0400_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0590_00.yaml`
- `radio_gs/configs/generated/scannet_dino_cv/scannet_og_hybrid_v67_dino_cv001_b2_s32768_ft20_scene0645_00.yaml`
