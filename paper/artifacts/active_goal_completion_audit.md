# Active Goal Completion Audit

Current objective: follow `ChatGPT-RADIO模型多视角重建优化 (1).md` by
reframing the paper around three tasks, reproducing open-source methods under
the same protocols where code/artifacts are available, keeping available GPUs
busy during the reproduction phase, and producing a guarded paper-facing
evidence package.

Status: **complete for the actionable local reproduction/evidence package**.
The final registry and claim guards now cover the three paper tasks, the
open-source comparison rows that can be reproduced locally have been promoted
from queued/in-flight artifacts to completed summaries, and the remaining gaps
are explicit caveats rather than hidden pending work. This does not establish an
unqualified global SOTA or ScanNet leaderboard claim.

## Prompt-To-Artifact Checklist

| Requirement | Current evidence | Status |
|---|---|---|
| Define the three paper tasks clearly | `paper/artifacts/final_rows.yaml` has T1 LERF rendered-view OVS, T2 LERF direct 3D selection, and T3 ScanNet point-cloud segmentation/probe; `paper/artifacts/README.md` indexes artifacts by T1/T2/T3 | Covered |
| Use a canonical paper-facing registry | `paper/artifacts/final_rows.yaml`; guarded by `radio_gs/scripts/validate_final_rows_registry.py` | Covered |
| Prevent overclaiming global SOTA or ScanNet leaderboard status | `radio_gs/scripts/validate_paper_claims.py`; final consistency/claim constraints in `paper/artifacts/` | Covered |
| Public artifact snapshot rather than private `output/` symlink only | `paper/artifacts/README.md`, `checksums.txt`, source JSONs, reports, manifests, audits, and per-method summaries | Covered |
| T1 RADIO-GS/CTF-GS main evidence | LERF rendered grounding threshold sweeps, controlled evidence table, cache and explicit-memory baselines | Covered |
| T2 direct-3D evidence | `lerf_direct_3d_selection.md`, per-scene direct-3D JSONs, query breakdowns, query audit, VPR protocol card, and SAM3-box geometry sweep | Covered for CTF-GS rows |
| T3 ScanNet evidence | OpenGaussian and RADIO-GS ScanNet JSON snapshots, contextual kNN/probe rows, and Semantic Gaussians compatibility eval | Covered as contextual/probe evidence, not leaderboard SOTA |
| OpenGaussian reproduction | ScanNet local reproduction and all-four-scene LERF compatibility readout; caveated in `external_baseline_audit.{md,json}` and registry | Covered |
| OccamLGS reproduction | All-four-scene LERF compatibility readout documented in `external_baseline_audit.{md,json}` and registry | Covered |
| LangSplatV2 reproduction | `langsplatv2_lerf_summary.{md,json}` reports all four compatibility scenes; scene-mean LocAcc 0.6176 / mIoU 0.4601, object-weighted LocAcc 0.6010 / mIoU 0.4487 over 208 queries | Covered |
| GAGS reproduction | `gags_lerf_summary.{md,json}` reports all four scenes; scene-mean LocAcc 0.7273 / mIoU 0.4893, object-weighted LocAcc 0.7308 / mIoU 0.4935 over 208 queries | Covered |
| Dr. Splat reproduction | `drsplat_lerf_summary.{md,json}` reports all four local nested-mask eval scenes; macro mIoU 0.1762, Acc@0.25 0.2561, Acc@0.5 0.1137, 208 objects, 0 missing masks | Covered |
| Classic LangSplat reproduction | `langsplat_classic_lerf_summary.{md,json}` reports all four compatibility scenes; scene-mean LocAcc 0.7335 / mIoU 0.4433, object-weighted LocAcc 0.7356 / mIoU 0.4613 over 208 queries | Covered |
| LEGaussians reproduction | Official quantize/train/render-mask local compatibility chain completed for all four LERF scenes; `legaussians_lerf_summary.{md,json}` reports scene-mean mIoU 0.2694, Acc@0.25 0.3974, Acc@0.5 0.2312, 208 objects, 0 missing masks | Covered |
| CAGS reproduction | `cags_lerf_summary.{md,json}` reports all four local compatibility scenes; scene-mean mIoU 0.2627 / Acc@0.25 0.3997, caveated by missing rendered masks | Covered with caveat |
| Semantic Gaussians reproduction | `semantic_gaussians_eval_metrics.{json,csv}` reports four ScanNet label-PLY compatibility scenes; mean IoU 0.0280 | Covered with compatibility caveat |
| LaGa reproduction | Affinity, descriptor, mask export, and nested-mask eval completed for all four LERF scenes; `laga_lerf_summary.{md,json}` reports macro mIoU 0.2337, Acc@0.25 0.3660, Acc@0.5 0.1535, 208 objects, 0 missing masks | Covered with descriptor-setting caveat |
| OpenGaFF | No public code/checkpoints found; tracked only as upstream-blocked published-context row | Blocked upstream |
| Parallel GPU utilization | Long-running reproduction chains used the available GPUs across GAGS, Dr. Splat, Semantic Gaussians, LEGaussians, and LaGa; completion artifacts and logs are under `output/baselines/**/logs/` | Covered |
| Improve CTF-GS/RADIO-GS evidence package | Main-task metrics, external rows, provenance, checksums, mechanism/failure analysis, and validators are present | Covered for paper package |
| Top-journal-ready claim discipline | The package is stronger and auditable, but global SOTA wording remains disallowed unless future strict leaderboard evidence supports it | Covered by caveat |

## Completed Baseline Rows

- GAGS completed all four LERF compatibility scenes from Occam RGB starts:
  `paper/artifacts/gags_lerf_summary.{json,md}`.
- Dr. Splat completed all four LERF nested-mask compatibility scenes:
  `paper/artifacts/drsplat_lerf_summary.{json,md}`.
- LEGaussians completed all four LERF official quantize/train/render-mask
  compatibility scenes:
  `paper/artifacts/legaussians_lerf_summary.{json,md}`.
- LaGa completed all four LERF affinity, descriptor, export, and eval stages:
  `paper/artifacts/laga_lerf_summary.{json,md}` and
  `paper/artifacts/laga_lerf_readiness_audit.{json,md}`.
- Semantic Gaussians completed the four-scene ScanNet label-PLY compatibility
  evaluation:
  `output/baselines/semantic_gaussians/scannet_compat_20260520/semantic_gaussians_eval_metrics.{json,csv}`
  plus `paper/artifacts/semantic_gaussians_readiness_audit.{json,md}`.
- LangSplatV2, classic LangSplat, CAGS, OpenGaussian, and OccamLGS are already
  represented in the synchronized final registry and external baseline audit.

## Residual Caveats

1. Several promoted rows are local compatibility reruns rather than strict
   released-checkpoint leaderboard submissions. The registry and external
   audits label these rows accordingly.
2. OpenGaFF remains upstream-blocked because no public code/checkpoints were
   available during this run.
3. LaGa descriptor construction was made tractable with the compatibility
   settings `max_views=32` and `num_per_cluster_features=5`; the exported
   masks and metrics are internally consistent with that setting.
4. Semantic Gaussians uses the available ScanNet label-PLY path because the
   extracted scenes provide `*_vh_clean_2.labels.ply` rather than per-view
   `label-filt` PNGs.
5. The evidence package supports guarded paper claims, but it does not support
   an unqualified global SOTA or ScanNet leaderboard statement.

## Verification Evidence

The final verification set was run after synchronizing completed summaries into
the paper registry:

```bash
bash radio_gs/scripts/run_repo_python.sh -m pytest -q \
  tests/test_eval_drsplat_lerf_masks.py \
  tests/test_audit_laga_lerf_readiness.py \
  tests/test_export_laga_lerf_masks.py \
  tests/test_build_laga_lerf_descriptors.py \
  tests/test_audit_semantic_gaussians_readiness.py \
  tests/test_validate_final_rows_registry.py \
  tests/test_sync_external_reproduction_summaries.py \
  tests/test_audit_external_baselines.py
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/validate_final_rows_registry.py
bash radio_gs/scripts/run_repo_python.sh radio_gs/scripts/validate_paper_claims.py
sha256sum -c paper/artifacts/checksums.txt
git diff --check
```

Observed status before this audit-file refresh: 52 targeted tests passed, the
final-rows registry validator passed, the paper-claims validator passed,
artifact checksums for the promoted results passed, and `git diff --check`
returned cleanly. After this file update, `checksums.txt` must be refreshed and
the same final verification commands rerun before the goal is closed.
