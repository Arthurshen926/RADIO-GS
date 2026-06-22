# the unpublished protocol source Alignment Audit

Created: `2026-05-20`

## Decision

The current submission route uses the public baseline rows listed by the unpublished protocol source as
published-context anchors, but does not compare against the unpublished protocol-source method row
itself. External comparison numbers should be cited from the unpublished protocol source or the
corresponding published papers, while GaussFM numbers remain locally audited
results from our own evaluator.

## Protocol Alignment

| Track | VALA-aligned protocol | GaussFM status | Paper-safe claim |
|---|---|---|---|
| LERF rendered-view grounding | 2D open-vocabulary feature rendering / localization | `0.8712` LocAcc / `0.5243` mIoU on the frozen four-scene LERF-OVS evaluator | Strong rendered-view grounding evidence; keep separate from direct 3D object selection. |
| LERF direct 3D object selection | Query 3D primitives, render selected primitives, evaluate mIoU / Acc@0.25 on LERF-OVS masks | Strict compact one-map row `0.4570` / `0.6851`; compact prompt-ensemble + RGB/score-component guard row `0.5014` / `0.7044`; frozen SAM3-box diagnostic row `0.5705` / `0.6835` | Compare against the published public baseline rows only; avoid any the unpublished protocol source head-to-head claim and keep the RGB component-guard caveat explicit. |
| ScanNet point-cloud understanding | Eight-scene ScanNet-v2 point-cloud understanding; one keyframe every 20 frames; Dr. Splat/LangSplatV2-style 3D evaluation with mIoU / Acc on GT semantic point clouds | VALA8 subset now uses `scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`, `scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`; paper-facing DINO-CV contextual kNN reaches `0.3722/0.6025`, `0.3791/0.6207`, `0.4591/0.7025` for 19/15/10 splits | Treat as VALA-aligned split-aligned direct point-query evidence. Use VALA-aligned paper numbers for external methods rather than local reruns. |

## Comparison Policy

- Do not spend further effort reproducing non-public or low-priority external
  baselines for the main submission route.
- Do not use local wrapper failures such as the Dr. Splat LERF compatibility
  export as main evidence against the method.
- Do not include CAGS in the main published-context comparison table.
- Do not include the unpublished protocol-source method row in the main comparison table; use only
  the baseline rows reported by the unpublished protocol source.
- Keep all scene-locked or post-hoc selectors in appendix/diagnostic status.

## ScanNet Published Context

External rows for the ScanNet comparison are copied from the unpublished protocol source arXiv v2 and
kept separate from local reproduction artifacts. The unpublished protocol-source method row is
omitted by policy:

| Method | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc |
|---|---:|---:|---:|
| LangSplat | 2.45 / 8.59 | 3.45 / 13.21 | 6.48 / 21.89 |
| LangSplatV2 | 14.75 / 25.47 | 17.09 / 35.68 | 22.83 / 41.52 |
| OpenGaussian | 27.73 / 42.01 | 29.67 / 46.15 | 39.93 / 57.34 |
| Dr. Splat | 29.31 / 47.68 | 33.25 / 54.33 | 44.19 / 65.19 |
| OccamLGS | 31.93 / 48.93 | 34.25 / 53.71 | 45.16 / 64.39 |
| VALA | 32.11 / 50.05 | 35.10 / 54.77 | 46.21 / 65.61 |
| GaussFM DINO-CV contextual kNN | 37.22 / 60.25 | 37.91 / 62.07 | 45.91 / 70.25 |

## Required Wording

Use:

> Using the public baseline rows reported by the unpublished protocol source, GaussFM provides a compact
> dual-readout teacher-feature field with strong rendered grounding, direct-3D
> object-selection evidence, and VALA-aligned split-aligned ScanNet transfer.

Avoid:

> GaussFM is universally SOTA on LERF direct 3D and ScanNet.

Also avoid:

> The old 10-scene ScanNet v67 table is the paper-facing VALA-aligned ScanNet
> comparison.
