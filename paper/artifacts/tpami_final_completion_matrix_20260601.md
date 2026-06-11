# TPAMI Final Completion Matrix

Date: 2026-06-01

This matrix audits whether the repository currently contains a journal-ready
CTF-GS manuscript package. It separates manuscript/content readiness from
human-only upload metadata such as author names, conflicts, funding, and portal
forms.

## Requirement-Level Audit

| Requirement | Evidence | Status |
| --- | --- | --- |
| Top-journal venue selected | `paper/artifacts/tpami_submission_strategy.md`, `docs/submission_status.md` | Covered: TPAMI is the target venue. |
| Correct journal-style template | `paper/radio_gs_tpami.tex` uses `IEEEtran` journal mode; `paper/IEEEtran.cls` and `paper/IEEEtran.bst` are included. | Covered. |
| Main manuscript exists and compiles | `paper/radio_gs_tpami.tex` -> `paper/radio_gs_tpami.pdf` | Covered. |
| Supplementary material exists and compiles | `paper/radio_gs_tpami_supplement.tex` -> `paper/radio_gs_tpami_supplement.pdf` | Covered. |
| Central logic chain is explicit | Abstract, Introduction, Method, Experiments, Discussion, Limitations, and Figure 1 all frame one compact foundation-feature Gaussian map with rendered-view, direct-primitive, and point-query readouts. | Covered. |
| Method claims are protocol-bounded | `radio_gs/scripts/validate_paper_claims.py`; main text distinguishes Multiview Primitive Registration as a training bridge, label-free color-edge support calibration, official SAM3 control, and external published-context results. | Covered. |
| Three benchmark tracks are represented | LERF rendered-view OVS, LERF direct 3D OVS, and VALA/OpenGaFF-8 ScanNet point query appear in main tables and `paper/artifacts/final_rows.yaml`. | Covered. |
| Frame-wise foundation-feature comparisons are represented | Main Table `tab:teacher_vs_rendered`, `tab:sam_dino_tasks`, and `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.{md,json}`. | Covered with bounded wording. |
| Component and quantitative ablations are represented | `paper/lerf_component_ablation_table.tex`, `paper/quantitative_ablation_summary_table.tex`, and supplementary protocol controls. | Covered. |
| Storage and efficiency are represented | `paper/storage_footprint_table.tex`, `paper/efficiency_cost_table.tex`, and supporting artifacts. | Covered. |
| Failure analysis is represented | `paper/lerf_failure_analysis_table.tex`, `paper/scannet_category_stability_table.tex`, and supporting artifacts. | Covered. |
| Main qualitative figures are curated | `paper/figures/radio_gs_framework.pdf`, `lerf_2d3d_ovs_qualitative.png`, `scannet_openvocab_3d_query_qualitative.png`, and `lerf_direct3d_support_policy_ablation_qualitative.png`; `paper/artifacts/figure_quality_audit_tpami_20260531.md`. | Covered. |
| Weaker diagnostic visuals are not overused in the main paper | SAM/DINO, multiview-registration, SAM3-box, and alpha/depth visuals are kept in the supplementary material. | Covered. |
| Reproducibility entry point exists | `paper/artifacts/tpami_reproducibility_package_20260601.md`. | Covered. |
| Large-asset release plan exists | `paper/artifacts/tpami_large_asset_release_manifest_20260601.md`. | Covered. |
| Upload-facing materials exist | `paper/tpami_cover_letter_draft.md`, `paper/tpami_submission_checklist.md`, and `paper/tpami_submission_mode_guide.md`. | Covered except human-specific fields. |
| Artifact and claim provenance are checkable | `paper/artifacts/checksums.txt`, `paper/artifacts/paper_assets_manifest.json`, `validate_paper_claims.py`, and `validate_final_rows_registry.py`. | Covered. |

## Fresh Verification Evidence

The latest verification pass ran:

```bash
/root/miniconda3/envs/cybersim_agent/bin/python -m py_compile \
  radio_gs/scripts/build_paper_assets_manifest.py \
  radio_gs/scripts/validate_paper_claims.py \
  radio_gs/scripts/draw_framework_figure.py \
  radio_gs/scripts/validate_final_rows_registry.py

sha256sum -c paper/artifacts/checksums.txt

/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/validate_paper_claims.py --root /root/RADIO-GS

/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/validate_final_rows_registry.py

git diff --check
```

Observed output:

- Python helper scripts compile.
- `sha256sum -c paper/artifacts/checksums.txt` reports `OK` for all tracked
  artifacts.
- `validate_paper_claims.py` prints `paper claims ok`.
- `validate_final_rows_registry.py` prints `final_rows registry ok`.
- `git diff --check` reports no whitespace errors.

The preceding LaTeX verification pass also built:

- `paper/radio_gs_tpami.pdf` from `paper/radio_gs_tpami.tex`.
- `paper/radio_gs_tpami_supplement.pdf` from
  `paper/radio_gs_tpami_supplement.tex`.

Final LaTeX log scans found no undefined references/citations, no warning lines,
and no overfull hbox reports.

## Human-Only Items Before Portal Upload

These do not change the technical manuscript content but must be completed by
the submitting authors:

1. Confirm in the TPAMI ScholarOne portal whether the selected route expects
   anonymous or author-visible review PDFs.
2. Fill author names, affiliations, corresponding author, ORCIDs, conflicts,
   funding, acknowledgements, and required AI/tool disclosures.
3. Decide whether large assets are staged at submission time or prepared for
   release after acceptance.
4. Do a final human proofread of names, metadata, and the generated PDF pages
   after author-specific edits.

## Completion Judgment

The repository now contains a submission-ready anonymous TPAMI manuscript
package from the technical-content perspective: main paper, supplement, figures,
tables, evidence registry, reproducibility entry point, release manifest, cover
letter draft, and upload checklist are present and verified. The only remaining
items require author-specific human input or portal choices and cannot be
filled from the repository alone.
