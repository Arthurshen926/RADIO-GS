# Frame-wise RADIO vs GaussFM 2D Usability Design

## Goal

Add a reproducible, paper-facing 2D feature-usability artifact that compares
frame-wise RADIO features against rendered GaussFM features under the
same frozen downstream heads. This supports the broader claim that the compact
Gaussian foundation-feature field can be more useful than raw frame-wise teacher
features in selected novel-view downstream tasks, without claiming universal
superiority.

## Scope

The first implementation is artifact consolidation, not a new training run. It
parses existing controlled LERF evidence and formal SAM3/DINOv3 task sweeps,
writes a JSON manifest, a Markdown report, and a compact LaTeX table snippet,
and updates the paper-facing documentation to reference those outputs.

## Inputs

- `paper/artifacts/controlled_evidence_table.json` for the rendered-view
  SigLIP2/text grounding and feature-memory controls.
- `output/lerf_sam_dino_tasks/formal_v9_dino_topk_area200_bg110_peak_20260514/lerf_sam_dino_task_aggregate.json`
  for the current formal SAM3/DINOv3 frozen-head task sweep.
- `output/lerf_sam_dino_tasks/formal_v8_mutual_homography_ransac_all_20260514/lerf_sam_dino_task_aggregate.json`
  for the optional homography-RANSAC DINO diagnostic row.
- `output/lerf_adaptor_downstream/mainline/lerf_adaptor_downstream_aggregate.json`
  for appendix-only prototype adaptor diagnostics.

## Outputs

- `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.json`
- `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md`
- `paper/tables/teacher_vs_ctfgs_2d_usability_20260525.tex`

## Claim Policy

The report must separate three levels of evidence:

1. Main text grounding/feature-memory evidence: GaussFM rendered features beat
   frame-wise RADIO reference, nearest-view cache, and explicit 1280-D Gaussian
   memory on LERF rendered-view localization and mIoU.
2. Frozen-head task evidence: GaussFM rendered features beat the teacher on
   SAM3-adaptor mIoU for point, box, and mask propagation tasks, and on DINOv3
   dense-match similarity plus DINOv3 mask-propagation localization accuracy.
3. Caveat evidence: the teacher remains stronger on DINOv3 dense-match hit rate
   and DINOv3 mask-propagation mIoU under the same readout. The report must say
   "selected downstream tasks" instead of "universally better."

## Non-Goals

- Do not tune thresholds or select rows using test GT masks.
- Do not claim official SAM3 instance segmentation from the SAM3-adaptor tasks.
- Do not merge rendered-view LERF mIoU and direct-3D object-selection mIoU into
  a single comparable metric.
- Do not add per-scene special cases.

## Design

Create a small CPU-only script in `radio_gs/scripts` with pure JSON parsing and
formatting helpers. The script exposes `build_report(...)`, `write_report(...)`,
and a CLI. Tests use temporary JSON fixtures to verify metric extraction,
delta/sign handling, conservative claim-summary generation, and file outputs.

The Markdown report is the source for human review. The JSON manifest is the
auditable source for paper tables. The LaTeX snippet is intentionally compact so
the CVPR draft can reference a short table while moving detailed rows to the
appendix.

## Validation

- Unit tests cover controlled evidence parsing and SAM3/DINO aggregate parsing.
- The real build command must complete without CUDA.
- Paper claim validation remains green after the docs are updated.
