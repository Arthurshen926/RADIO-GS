# TPAMI Readiness Audit

Date: 2026-06-01

This audit records the current journal-submission state for the CTF-GS paper.
It is intentionally stricter than a compile check: the goal is to verify whether
the main manuscript is coherent, protocol-bounded, and backed by reproducible
artifacts.

## Main-Paper State

| Area | Status | Evidence |
| --- | --- | --- |
| Target venue format | Covered | `paper/radio_gs_tpami.tex`, `paper/IEEEtran.cls`, `paper/IEEEtran.bst`, compiled `paper/radio_gs_tpami.pdf` |
| Supplementary material | Covered | `paper/radio_gs_tpami_supplement.tex` compiles to `paper/radio_gs_tpami_supplement.pdf` with calibration tables, protocol controls, and additional qualitative figures. |
| Central thesis | Covered | The paper consistently frames CTF-GS as one compact foundation-feature Gaussian map with rendered-view, direct-primitive, and direct-point readouts. |
| Figure 1 | Covered | `paper/figures/radio_gs_framework.pdf` is a vector-first journal figure with offline supervision, a central stored compact map, global readout heads, and protocol evidence. |
| Main qualitative figures | Covered | Main paper uses LERF 2D/3D OVS, ScanNet binary query, and Direct3D support-calibration ablation; weaker DINO/SAM/multiview-registration visuals are kept for supplement. The LERF 2D/3D figure uses high-quality cases (`old camera`, `green apple`, `pumpkin`, `tea in a glass`, `apple`, `bag of cookies`). |
| Reproducibility package | Covered | `paper/artifacts/tpami_reproducibility_package_20260601.md` lists paper integrity checks, canonical rows, table/figure regeneration commands, and data-dependent evaluation templates. |
| Large-asset release plan | Covered | `paper/artifacts/tpami_large_asset_release_manifest_20260601.md` lists required checkpoints, datasets, evaluation outputs, optional diagnostics, and release verification commands. |
| LERF rendered-view result | Covered | Table reports 0.8598 LocAcc / 0.5889 mIoU with feature-only SAM3-adaptor boundary readout and no official RGB SAM decoder. |
| LERF direct 3D result | Covered with protocol caveat | The deployed compact readout reports 0.5014 mIoU / 0.7044 Acc@0.25 without a multiview-registration cache or official SAM readout; label-free color-edge support calibration is disclosed. |
| ScanNet direct point query | Covered with scope caveat | VALA/OpenGaFF-8 rows report split-19/15/10 results and are framed as direct point-query feature probes, not a full ScanNet segmentation leaderboard. |
| Frame-wise foundation-feature comparison | Covered but bounded | Main table and SAM/DINO/SigLIP probes support selected downstream usability over frame-wise RADIO; text avoids universal superiority language. |
| Storage/efficiency | Covered | Storage footprint and efficiency/cost tables are included and separated from optional multiview-registration caches and runtime buffers. |
| Failure analysis | Covered | LERF small-object failures and ScanNet category instability are included as main-paper analysis. |
| External baselines | Bounded | External rows are source-anchored context numbers; the paper avoids strict same-evaluator SOTA claims. |

## Verification Run

Latest verification commands:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_tpami.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_tpami_supplement.tex
rg -F "undefined" paper/radio_gs_tpami.log
rg -F "undefined" paper/radio_gs_tpami_supplement.log
rg -F "Overfull \\hbox" paper/radio_gs_tpami.log
rg -F "Overfull \\hbox" paper/radio_gs_tpami_supplement.log
rg -F "pdfTeX warning" paper/radio_gs_tpami.log
rg -F "pdfTeX warning" paper/radio_gs_tpami_supplement.log
/root/miniconda3/envs/cybersim_agent/bin/python radio_gs/scripts/validate_paper_claims.py --root /root/RADIO-GS
/root/miniconda3/envs/cybersim_agent/bin/python radio_gs/scripts/validate_final_rows_registry.py
/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile radio_gs/scripts/build_paper_assets_manifest.py radio_gs/scripts/validate_paper_claims.py radio_gs/scripts/draw_framework_figure.py radio_gs/scripts/validate_final_rows_registry.py radio_gs/scripts/compose_lerf_main_qualitative.py radio_gs/scripts/compose_scannet_openvocab_query_qualitative.py radio_gs/scripts/compose_lerf_direct3d_ablation_qualitative.py
sha256sum -c paper/artifacts/checksums.txt
git diff --check
```

Observed state after the 2026-06-01 polish pass:

- TPAMI PDF compiles.
- Supplementary PDF compiles.
- No undefined references/citations remain in the final log.
- No overfull hbox or pdfTeX warnings remain in the final log.
- Paper-claim validator passes.
- Final-row registry validator passes.
- Artifact checksum manifest passes.
- Python helper scripts compile.
- Git whitespace check passes.

## Remaining Non-Blocking Work

These items are not blockers for a manuscript draft, but they would improve the
submission package before an actual journal upload:

1. Decide whether to rerun any external baselines locally. This is only required
   if the paper wants to claim a strict same-evaluator leaderboard.
2. Do one final human read-through for language, citation ordering, and
   compressed table readability.
3. Stage the large-asset release package using
   `paper/artifacts/tpami_large_asset_release_manifest_20260601.md` if the
   journal or code-release venue requires downloadable checkpoints and caches at
   submission time.
