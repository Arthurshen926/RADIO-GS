# RADIO-GS Benchmarking Plan

This note freezes which published methods should be treated as the main paper
comparisons and which methods should only appear in related work or in a
supplementary section.

## Main-table benchmark set

The current paper should anchor its LERF-OVS grounding comparison around these
published methods:

| Method | Venue | Why it must be included | Current role |
|---|---|---|---|
| LERF | ICCV 2023 | Foundational open-vocabulary 3D querying baseline | Historical lower bound |
| LangSplat | CVPR 2024 | Strong 3DGS language-field baseline with clear efficiency claim | Modern competitive baseline |
| LEGaussians | CVPR 2024 | Best currently aligned published open-vocabulary 3D Gaussian baseline in the repository's framing | Primary SOTA target |
| RADIO-GS | This repository | Current method | Ours |

## Supplementary benchmark set

These are published and relevant, but they should not replace the main-table trio
above unless the evaluation protocol is re-aligned more carefully.

| Method | Venue | Best use |
|---|---|---|
| Gaussian Grouping | ECCV 2024 | Open-world 3D segmentation and editing discussion |
| 3D Gaussian Splatting | SIGGRAPH 2023 | Geometry and rendering efficiency context |

## Methods that should not be sold as published external baselines

- `Feature3DGS-style` in the current repository is an **internal reproduced baseline label**.
- Room_0 geometric or fused depth numbers are **internal task-utility evidence**, not external paper-to-paper grounding baselines.

## Current repository-carried draft rows

The following four-scene grounding values are the rows currently carried by the
internal reports.

`RADIO-GS` comes from the current internal evaluation pipeline. The three
external rows are still unresolved borrowed comparison values, not exact
paper-anchored numbers, so this table is useful for internal drafting but not
submission-safe as written.

| Method | Figurines | Ramen | Teatime | Waldo Kitchen | Macro |
|---|---:|---:|---:|---:|---:|
| LERF | 0.520 | 0.503 | 0.653 | 0.456 | 0.533 |
| LangSplat | 0.592 | 0.659 | 0.693 | 0.600 | 0.636 |
| LEGaussians | 0.631 | 0.695 | 0.745 | 0.593 | 0.666 |
| RADIO-GS | 0.821 | 0.901 | 0.881 | 0.864 | 0.867 |

Source of the current internal table layout:

- [output/radio_gs/reports/sota_comparison_table.md](../output/radio_gs/reports/sota_comparison_table.md)
- [output/radio_gs/reports/comprehensive_results.md](../output/radio_gs/reports/comprehensive_results.md)
- Exact provenance status: [output/radio_gs/reports/baseline_source_verification.md](../output/radio_gs/reports/baseline_source_verification.md)

## Benchmark rules for the paper draft

1. Every external method in the main table must point to a published paper URL.
2. Every external number must be traceable to a paper table, supplementary table, or official project page with paper backing.
3. Internal reproduced baselines must be labeled as reproduced or internal, never as published SOTA.
4. If a method targets a different task, move it to related work or supplementary analysis instead of forcing it into the main table.

## Next benchmark steps

1. Re-check exact table locations for LERF, LangSplat, and LEGaussians before writing the camera-ready main table.
2. Add one cross-domain benchmark, ideally ScanNet, to reduce the risk that the paper looks over-specialized to LERF-OVS.
3. Add a clean efficiency comparison section so the paper does not read as accuracy-only.
