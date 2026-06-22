# TPAMI Submission Checklist

This checklist tracks the final non-experimental items before uploading the
GaussFM manuscript package.

## Manuscript Files

- [x] Main manuscript: `paper/radio_gs_tpami.tex`
- [x] Main PDF: `paper/radio_gs_tpami.pdf`
- [x] Supplement: `paper/radio_gs_tpami_supplement.tex`
- [x] Supplement PDF: `paper/radio_gs_tpami_supplement.pdf`
- [x] Bibliography: `paper/radio_gs_refs.bib`
- [x] IEEEtran class/style files: `paper/IEEEtran.cls`, `paper/IEEEtran.bst`
- [x] Legacy CVPR draft moved out of the active root:
  `paper/archive/radio_gs_draft_legacy_cvpr.tex` is traceability-only and must
  not be uploaded as an active source file.
- [x] Main figures: `paper/figures/figure1_overall_framework.pdf`,
  `paper/figures/figure2_method_details.pdf`,
  `paper/figures/lerf_2d3d_ovs_qualitative.png`,
  `paper/figures/scannet_openvocab_3d_query_qualitative.png`,
  `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png`

## Evidence And Reproducibility

- [x] Canonical result registry: `paper/artifacts/final_rows.yaml`
- [x] Claim validator: `radio_gs/scripts/validate_paper_claims.py`
- [x] Final-row registry validator:
  `radio_gs/scripts/validate_final_rows_registry.py`
- [x] Artifact checksum manifest: `paper/artifacts/checksums.txt`
- [x] Reproducibility entry point:
  `paper/artifacts/tpami_reproducibility_package_20260601.md`
- [x] Optional benchmark provenance audit:
  `paper/benchmark_provenance_table.tex` (retained outside the main manuscript)
- [x] Large-asset release manifest:
  `paper/artifacts/tpami_large_asset_release_manifest_20260601.md`
- [x] Readiness audit: `paper/artifacts/tpami_readiness_audit_20260601.md`
- [x] Submission mode guide: `paper/tpami_submission_mode_guide.md`

## Commands To Run Before Upload

```bash
cd /root/RADIO-GS

(
  cd paper
  latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_tpami.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error radio_gs_tpami_supplement.tex
)

rg -n -F "undefined" paper/radio_gs_tpami.log paper/radio_gs_tpami_supplement.log
rg -n -F "Overfull \\hbox" paper/radio_gs_tpami.log paper/radio_gs_tpami_supplement.log
rg -n "LaTeX Warning|Package .* Warning|Warning:" paper/radio_gs_tpami.log paper/radio_gs_tpami_supplement.log

test ! -f paper/radio_gs_draft.tex

/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/validate_paper_claims.py --root /root/RADIO-GS

/root/miniconda3/envs/cybersim_agent/bin/python \
  radio_gs/scripts/validate_final_rows_registry.py

sha256sum -c paper/artifacts/checksums.txt
git diff --check
```

Expected result: the LaTeX commands exit successfully; the three `rg` checks
print no matches; validators print `paper claims ok` and `final_rows registry
ok`; checksum and whitespace checks pass.

## Human-Only Upload Items

- [ ] Replace `Anonymous Authors` with the final author list if the selected
  submission route is not anonymous.
- [ ] Confirm whether the TPAMI ScholarOne route requests double-anonymous or
  single-blind review PDFs, then follow `paper/tpami_submission_mode_guide.md`.
- [ ] Fill cover letter fields in `paper/tpami_cover_letter_draft.md`.
- [ ] Add funding, acknowledgements, and required AI/tool disclosures.
- [ ] Confirm all third-party dataset, code, pretrained-model, and figure
  licenses are acceptable for submission.
- [ ] Decide whether the large-asset package is staged at submission time or
  prepared for release after acceptance.
