# OpenGaFF Alignment Audit

Created: `2026-05-20`

## Decision

The current submission route uses OpenGaFF as the recent published-context
anchor. OpenGaFF has no public code in this workspace, so we do not attempt a
local reproduction. External comparison numbers should be cited from OpenGaFF
or the corresponding published papers, while CTF-GS numbers remain locally
audited results from our own evaluator.

## Protocol Alignment

| Track | OpenGaFF-facing protocol | CTF-GS status | Paper-safe claim |
|---|---|---|---|
| LERF rendered-view grounding | 2D open-vocabulary feature rendering / localization | `0.8712` LocAcc / `0.5243` mIoU on the frozen four-scene LERF-OVS evaluator | Strong rendered-view grounding evidence; keep separate from direct 3D object selection. |
| LERF direct 3D object selection | Query 3D primitives, render selected primitives, evaluate mIoU / Acc@0.25 on LERF-OVS masks | VPR fixed row `0.4801` / `0.6760`; direct-field + official SAM3-box fixed row `0.5705` / `0.6835` | The fixed SAM3-box row exceeds OpenGaFF's published mIoU context value `0.5436`, but not its Acc@0.25 `0.8084`; claim mIoU-competitive/stronger, not universal dominance. |
| ScanNet point-cloud understanding | Eight-scene ScanNet-v2 point-cloud understanding; one keyframe every 20 frames; Dr. Splat/LangSplatV2-style 3D evaluation with mIoU / Acc on GT semantic point clouds | VALA8 subset now uses `scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`, `scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`: Gaussian-index `0.3583/0.6006`, `0.3618/0.6152`, `0.4367/0.6998`; DINO-CV contextual kNN `0.3704/0.6017`, `0.3771/0.6198`, `0.4585/0.7032` for 19/15/10 splits | Treat as VALA/OpenGaFF split-aligned direct point-query evidence. Use OpenGaFF/VALA paper numbers for external methods rather than local reruns. |

## Comparison Policy

- Do not spend further effort reproducing non-public or low-priority external
  baselines for the main submission route.
- Do not use local wrapper failures such as the Dr. Splat LERF compatibility
  export as main evidence against the method.
- Do not include CAGS in the main published-context comparison table.
- Keep OpenGaFF as the main recent context row, with a clear note that it is an
  arXiv published-context result rather than a reproduced baseline.
- Keep all scene-locked or post-hoc selectors in appendix/diagnostic status.

## ScanNet Published Context

External rows for the ScanNet comparison are copied from OpenGaFF arXiv v2 and
kept separate from local reproduction artifacts:

| Method | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc |
|---|---:|---:|---:|
| LangSplat | 2.45 / 8.59 | 3.45 / 13.21 | 6.48 / 21.89 |
| LangSplatV2 | 14.75 / 25.47 | 17.09 / 35.68 | 22.83 / 41.52 |
| OpenGaussian | 27.73 / 42.01 | 29.67 / 46.15 | 39.93 / 57.34 |
| Dr. Splat | 29.31 / 47.68 | 33.25 / 54.33 | 44.19 / 65.19 |
| OccamLGS | 31.93 / 48.93 | 34.25 / 53.71 | 45.16 / 64.39 |
| VALA | 32.11 / 50.05 | 35.10 / 54.77 | 46.21 / 65.61 |
| OpenGaFF | 36.55 / 50.57 | 42.78 / 72.85 | 57.85 / 77.93 |
| CTF-GS Gaussian-index | 35.83 / 60.06 | 36.18 / 61.52 | 43.67 / 69.98 |
| CTF-GS contextual kNN | 36.77 / 59.97 | 37.48 / 61.81 | 45.62 / 70.08 |

## Required Wording

Use:

> Under an OpenGaFF-aligned published-context comparison, CTF-GS exceeds the
> reported LERF direct-3D mIoU while trailing the reported Acc@0.25, and it
> provides a compact dual-readout teacher-feature field with strong rendered
> grounding and VALA/OpenGaFF split-aligned ScanNet transfer evidence.

Avoid:

> CTF-GS is universally SOTA on LERF direct 3D and ScanNet.

Also avoid:

> The old 10-scene ScanNet v67 table is the paper-facing OpenGaFF/VALA ScanNet
> comparison.
