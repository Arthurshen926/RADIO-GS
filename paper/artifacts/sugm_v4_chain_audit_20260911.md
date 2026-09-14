# v4 chain audit — 2026-09-11

Scope: latest user attachment `4b8c75c4-87e5-4b0e-8454-65a41d1addc5/pasted-text.txt`.
This is a partial implementation audit, not a claim that all bugs are fixed or a new benchmark result.

## Repairs and checks completed

- Permanently quarantine `lerf_development_pipeline.run`, including its deprecated opt-in. That path applied a SigLIP2 summary head to spatial/pooled tokens without summary supervision. Historical helper code remains for inspection; its run entry cannot produce new results.
- Official crop summary wrapper now supports tuple, mapping and named outputs and rejects absent, zero, nonfinite and wrong-shaped summaries. Shape checks cannot establish semantic provenance by themselves.
- Actual pinned C-RADIO checkpoint uses per-teacher class tokens: SigLIP2 slot 0, DINOv3 slot 1. `OfficialCropSummaryRuntime` reencodes RGB crops and selects the genuine SigLIP2 summary. Existing crop caches are not invalid merely because they use a summary head.
- Object hypothesis metadata now reports the configured cannot-link and evidence-pooling policies instead of constant values. This does not change historical object payloads.
- Training receipts no longer describe synthetic SAM-like perturbations as real SAM noise. Explicitly record the local RADIO descriptor contract and unvalidated deployment domain.

Regression: `tests/v4`: 253 passed, 1 skipped. `tests/test_posefree_image_query_cache.py`: 12 passed, 1 skipped (including seven new summary cases). These are code checks, not quality measurements.

## Existing language cache integrity

Audited `/mnt/pool/sqy/results/RADIO-GS/output/v4_1_fragment_language_20260904` using four parallel CPU workers. For every frame, verified output, source RGB and SAM payload SHA256 against the manifest. Checked both descriptor matrices for proposal count, 1536 dimensions, finite/nonzero rows and normalization.

| Scene | Frames | Descriptor rows (masked + context) | Max norm error | Errors |
| --- | ---: | ---: | ---: | ---: |
| figurines | 32 | 1158 | 0.00017643 | 0 |
| ramen | 32 | 970 | 0.00024510 | 0 |
| teatime | 32 | 1342 | 0.00024248 | 0 |
| waldo_kitchen | 32 | 818 | 0.00019432 | 0 |

This verifies integrity, not source-crop/text retrieval accuracy or fresh reencoding equivalence. Neither was measured this round. Manifest authority and checkpoint hashes were not rehashed in this cache scan.

## Unclosed links and next experiments

1. Identity: establish native crop versus paired official text ranking on fixed source-only examples before changing learned summaries. Compare genuine summary against historical pooled proxy only as a labeled diagnostic. No held-out GT for model selection.
2. Association: training uses mean local RADIO F71 columns 4:68; deployment uses SigLIP2 masked/context summaries. Equal scalar feature layouts do not establish equal feature distributions. First align descriptors or train a geometry-only ablation. Current set-conditioned edge network still ends in greedy grouping; it is not a soft partition solution.
3. Object/part hierarchy: disjoint masks in one view are not sufficient proof of separate objects. Separate proven conflict from part compatibility. Preserve local query capability; do not force material/part queries into whole-object outputs.
4. Completion: adapter now marks overlapping memberships unknown instead of choosing an arbitrary winner, but the categorical checkpoint still cannot consume full soft memberships. This repair prevents false certainty; it does not solve information loss or noisy-fragment training mismatch.
5. Query posterior: explicitly test multiple targets, no-match and part prompts; verify 2D rendering and 3D selection originate from the same element scores. Upper bounds are diagnostics, not public performance.
6. Public protocol: correct the earlier claim that LERF3D necessarily requires element-level GT. OpenGaussian selects 3D Gaussians, renders them, and evaluates resulting 2D masks. Match official frames, image resolution, mask conversion, missing predictions and aggregation before reporting comparable results. Current coarse-grid/category-macro diagnostic numbers are not automatically this protocol.

Primary references inspected: https://github.com/NVlabs/RADIO ; https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/README.md ; https://raw.githubusercontent.com/yanmin-wu/OpenGaussian/main/scripts/compute_lerf_iou.py . The latter uses image threshold >10, missing predictions as zero, observation-wise mean IoU and strict >0.25 / >0.5 accuracies.

No new mIoU or GPU training result in this audit. At inspection all six GPUs showed 100% utilization and about 2.8 GB free each, occupied by existing tasks. Do not kill or compete blindly with them. Prior September 4 results remain historical diagnostics and must not be attributed to these repairs.
