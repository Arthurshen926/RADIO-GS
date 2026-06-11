# Baseline Reproduction and Qualitative Asset Audit

Date: 2026-06-10.

## LangSplatV2 for LERF rendered-view qualitative

Local code exists at `/root/baselines/LangSplatV2`, and the compatibility
evaluation directory exists at
`output/baselines/langsplatv2/lerf_compat_20260518`.

Current artifact status:

| Asset | Status |
|---|---|
| Same-protocol summary JSON | Available: `output/baselines/langsplatv2/lerf_compat_20260518/langsplatv2_lerf_summary.json` |
| Prediction mask / heatmap PNGs | Available after local export: chosen mask / heatmap / overlay PNGs for all 208 LERF query instances |
| Main qualitative use | Used in `paper/figures/lerf_2d3d_ovs_qualitative.png` as the rendered-view prior; the composer falls back to classic LangSplat only if a V2 panel is missing |

Recorded same-protocol quantitative summary from the re-exported local run:

| Scene | LocAcc | mIoU | Queries |
|---|---:|---:|---:|
| Figurines | 0.8214 | 0.5965 | 56 |
| Ramen | 0.7183 | 0.5913 | 71 |
| Teatime | 0.8814 | 0.7158 | 59 |
| Waldo Kitchen | 0.7273 | 0.5558 | 22 |
| Scene mean | 0.7871 | 0.6149 | 208 |

Decision: use LangSplatV2 as the local reproduced rendered-view qualitative
baseline. The exported visual assets live under
`output/baselines/langsplatv2/lerf_compat_20260518/eval/*/*/` and include
`chosen_*.png`, `heatmap_*.png`, and `overlay_*.png` files.

## Dr. Splat for LERF direct 3D qualitative

Local code exists at `/root/baselines/Dr-Splat`, and local LERF compatibility
outputs exist at `output/baselines/dr_splat/lerf_compat_20260519`.

Current artifact status:

| Asset | Status |
|---|---|
| LERF compatibility prediction PNGs | Available: 712 PNG files |
| LERF summary JSON | Available: `paper/artifacts/drsplat_lerf_summary.json` |
| Main qualitative use | Usable as a local reproduced prior for LERF direct 3D qualitative |

Recorded compatibility-wrapper summary: macro mIoU 0.1762, Acc@0.25 0.2561,
and Acc@0.50 0.1137 over 208 query instances. This local wrapper is much weaker
than the published Dr. Splat context, so it should be used only as qualitative
prior evidence under our exported-mask setting, not as a replacement for the
published-context quantitative row.

## Dr. Splat for ScanNet qualitative

No local Dr. Splat ScanNet prediction/evaluation directory was found under
`output/baselines`. The local Dr. Splat README also lists the official
evaluation section as `TBA`, although the repository contains ScanNet loader
code.

Decision: do not replace the ScanNet qualitative baseline with Dr. Splat.
Use the completed local VALA compatibility reproduction instead.

## Newer ScanNet qualitative baseline search

I checked the local baseline workspace for a stronger same-protocol ScanNet
qualitative replacement:

| Candidate | Local status | Decision |
|---|---|---|
| VALA | Completed a local VALA/OpenGaFF-8 compatibility run at resolution 2 for all eight scenes; prediction PLY assets are available under `output/baselines/vala/scannet_vala8_compat_20260611_res2/visualizations/*/` | Use as the ScanNet qualitative baseline |
| Semantic Gaussians | Partial ScanNet outputs exist at `output/baselines/semantic_gaussians/scannet_compat_20260520`, but only four scenes were found and the local macro mIoU is weak (`0.0280`) | Not suitable as a stronger qualitative baseline |
| OpenGaussian | Local reproduced assets are available, but VALA is newer and now has complete VALA8 qualitative assets | Replaced by VALA in the main ScanNet qualitative figure |

The completed VALA compatibility run uses all eight VALA/OpenGaFF scenes:
`scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`,
`scene0140_00`, `scene0347_00`, `scene0400_00`, and `scene0590_00`.
It is a full-scene/view qualitative reproduction, but the local compatibility
numbers are weak and should not replace the published VALA/OpenGaFF quantitative
rows:

| Split | mIoU | mAcc | Overall Acc |
|---|---:|---:|---:|
| 19 | 0.1198 | 0.2374 | 0.2381 |
| 15 | 0.1487 | 0.2815 | 0.3480 |
| 10 | 0.2272 | 0.4170 | 0.4366 |

Decision after the 2026-06-11 run: LERF rendered-view qualitative uses locally
reproduced LangSplatV2; LERF direct-3D qualitative uses the local Dr. Splat
compatibility masks as a prior; ScanNet qualitative now uses the local VALA
compatibility reproduction instead of OpenGaussian.
