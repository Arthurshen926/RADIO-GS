# Baseline Source Verification

Last updated: 2026-05-10

This sheet tracks the external methods used in the conservative submission route.
The main table now uses official-source rows instead of the deprecated
repository-carried placeholder rows.

## Identity Check

| Method | Paper | Venue | OpenAccess page | Status |
|---|---|---|---|---|
| LERF | LERF: Language Embedded Radiance Fields | ICCV 2023 | `https://openaccess.thecvf.com/content/ICCV2023/html/Kerr_LERF_Language_Embedded_Radiance_Fields_ICCV_2023_paper.html` | verified identity |
| LangSplat | LangSplat: 3D Language Gaussian Splatting | CVPR 2024 | `https://openaccess.thecvf.com/content/CVPR2024/html/Qin_LangSplat_3D_Language_Gaussian_Splatting_CVPR_2024_paper.html` | verified identity |
| LEGaussians | Language Embedded 3D Gaussians for Open-Vocabulary Scene Understanding | CVPR 2024 | `https://openaccess.thecvf.com/content/CVPR2024/html/Shi_Language_Embedded_3D_Gaussians_for_Open-Vocabulary_Scene_Understanding_CVPR_2024_paper.html` | verified identity |

## Current Main Table Values

| Method | Figurines | Ramen | Teatime | Waldo Kitchen | Macro | Source row | Freeze status |
|---|---:|---:|---:|---:|---:|---|---|
| LERF | 0.795 | 0.625 | 0.938 | 0.815 | 0.793 | ICCV 2023 main paper Table 1 | official-source row |
| LangSplat | 0.804 | 0.732 | 0.881 | 0.955 | 0.843 | CVPR 2024 main paper Table 1 | official-source row |
| LEGaussians | 0.767 | 0.737 | 0.683 | 0.523 | 0.678 | CVPR 2024 supplementary Table 5, Ours LA | official-source row with kitchen-label caveat |

The paper table reports all values as decimals. LERF and LangSplat papers report
percentages; those rows are divided by 100. The macro column is recomputed over
the four scenes used by the current RADIO-GS table rather than copied from any
paper-specific overall column.

## Deprecated Placeholder Rows

These rows were previously hardcoded in
`radio_gs/scripts/build_submission_tables.py` and are retained only to prevent
future confusion.

| Method | Figurines | Ramen | Teatime | Waldo Kitchen | Macro | Status |
|---|---:|---:|---:|---:|---:|---|
| LERF | 0.520 | 0.503 | 0.653 | 0.456 | 0.533 | deprecated unresolved secondary-source row |
| LangSplat | 0.592 | 0.659 | 0.693 | 0.600 | 0.636 | deprecated unresolved secondary-source row |
| LEGaussians | 0.631 | 0.695 | 0.745 | 0.593 | 0.666 | deprecated unresolved secondary-source row |

## Official Source Audit

| Method | Exact official row found? | Source | Exact location | Official reference | Confidence |
|---|---|---|---|---|---|
| LERF | yes | main paper | ICCV 2023 Table 1 | localization accuracy row reports `79.5 / 62.5 / 93.8 / 81.5` for `figurines / ramen / teatime / waldo kitchen` | high |
| LangSplat | yes | main paper | CVPR 2024 Table 1 | localization accuracy row reports `80.4 / 73.2 / 88.1 / 95.5` for `figurines / ramen / teatime / waldo kitchen` | high |
| LEGaussians | yes, with scene-label caveat | supplement | CVPR 2024 Supplementary Table 5 | LeRF-dataset `Ours` LA row reports `0.767 / 0.737 / 0.683 / 0.523` for `figurines / ramen / teatime / kitchen` | high |

## Cross-Paper Protocol Check

Official-source rows exist, but they are not a reproduced unified-protocol
benchmark. This is now handled as a caption and discussion caveat rather than as
an unresolved provenance blocker.

| Method / source | Figurines | Ramen | Teatime | Waldo Kitchen / Kitchen | Metric form | Notes |
|---|---:|---:|---:|---:|---|---|
| LERF paper Table 1 | 79.5 | 62.5 | 93.8 | 81.5 | percent | direct official row |
| LangSplat Table 1 re-reported LERF | 75.0 | 62.0 | 84.8 | 72.7 | percent | differs from the LERF paper row, so paper protocols are not identical |
| LangSplat Table 1 | 80.4 | 73.2 | 88.1 | 95.5 | percent | direct official row from the LangSplat paper |
| LEGaussians Supplement Table 5, LeRF row | 0.866 | 0.552 | 0.683 | 0.676 | decimal | re-reported LERF row, also differs from the LERF paper row |
| LEGaussians Supplement Table 5, Ours row | 0.767 | 0.737 | 0.683 | 0.523 | decimal | uses `kitchen`, not explicitly `waldo kitchen` |

These discrepancies imply that scene-name overlap alone is not enough to claim a
strict unified benchmark. The current submission route therefore uses the rows
as official-source context and avoids a reproduced-SOTA claim unless the
baselines are rerun under the local evaluator.

## Paper-Freeze Checklist

- [x] paper identity and venue checked against OpenAccess pages
- [x] nearest official table or supplement location recorded for each method
- [x] current repo four-scene values traced to exact official-source rows
- [x] deprecated secondary-source placeholders removed from the table generator
- [x] protocol mismatch for the candidate official rows documented
- [x] paper caption/discussion caveat records cross-paper protocol limits

## Current Verification Notes

- 2026-05-10 recheck: `radio_gs/scripts/build_submission_tables.py` now hardcodes
  the official-source values above and regenerates both markdown and LaTeX main
  tables.
- The previous unresolved placeholder rows must not be used in the paper.
- The official-source rows are acceptable for a conservative related-work
  comparison table. A strict SOTA claim still requires reproducing LERF,
  LangSplat, and LEGaussians under the same evaluator and annotations used by
  RADIO-GS.

## Conservative Route Decision

The main table remains fixed to:

- `LERF`
- `LangSplat`
- `LEGaussians`
- `RADIO-GS`

The following are not part of the current freeze scope:

- `FMGS`
- `Dr.Splat`
- `PROFUSE`

Reason: they are not integrated into the current LERF-OVS benchmark sheet,
result pipeline, or paper protocol. They can be discussed in related work or
future benchmark expansion, but should not dilute the current submission table.
