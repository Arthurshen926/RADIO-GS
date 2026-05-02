# RADIO-GS Paper Writing Skeleton

This note turns the current repository assets into a first-pass manuscript scaffold.
It is intentionally conservative: any quantitative statement used in the paper
should be frozen against the generated audit report before it enters the final draft.

For the current 2026-05-02 frozen numbers and manuscript text, use
[`paper_draft_current.md`](paper_draft_current.md) as the active draft. This file
is retained as the older writing scaffold.

## Working title

Foundation Feature Reconstruction in 3D Gaussian Scenes for Open-Vocabulary Scene Understanding

## Abstract draft

We study how to reconstruct foundation-model spatial features inside a 3D Gaussian scene representation so that a single trained scene can support open-vocabulary grounding, depth prediction, and semantic transfer at novel views. Our method distills frozen RADIO teacher features into a hybrid Gaussian feature field that combines per-Gaussian latent storage with a coarse spatial branch, and supervises the reconstruction through a feature codec and frozen downstream heads instead of task-specific finetuning. This design turns the 3D scene itself into a reusable feature memory rather than a scene-specific classifier. Across LERF-OVS scenes, the current system already shows strong open-vocabulary grounding performance, while auxiliary experiments on Replica indicate that the reconstructed field also preserves geometric and semantic utility. The main remaining challenge before submission is not core method capability, but result freezing: broader public comparisons, cross-domain evidence, and exact provenance for every reported number.

## Introduction skeleton

Paragraph 1:
Large vision-language models expose rich dense features, but most 3D pipelines still optimize either RGB appearance or narrowly task-specific scene embeddings. This leaves a gap between 2D foundation features and reusable 3D scene representations.

Paragraph 2:
The paper targets feature reconstruction rather than direct task fitting. The core question is whether a 3D Gaussian scene can preserve high-dimensional teacher features well enough that novel rendered views remain useful for open-vocabulary grounding and other downstream probes.

Paragraph 3:
The technical challenge is that dense foundation features are high-dimensional, view-dependent, and expensive to store directly on every Gaussian. RADIO-GS addresses this with a hybrid feature field, an HCD-style bottleneck codec, screen-space refinement, and frozen depth-head supervision.

Paragraph 4:
The current empirical story should be framed around LERF-OVS as the main benchmark, with Replica depth and segmentation as supporting evidence that the reconstructed field retains task utility beyond language grounding.

## Contributions draft

1. We formulate 3D Gaussian feature reconstruction as a scene-level distillation problem from frozen RADIO teacher features, instead of learning scene-specific open-vocabulary heads from scratch.
2. We introduce a hybrid Gaussian feature representation with compressed latent decoding and auxiliary frozen-head supervision, which improves the usability of reconstructed features for downstream tasks.
3. We show that one reconstructed feature field can support multiple outputs, with LERF-OVS grounding as the primary benchmark and Replica depth and segmentation as supporting evidence.

## Method section outline

1. Problem setup: frozen teacher feature extraction, Gaussian scene representation, and novel-view feature rendering.
2. Hybrid feature field: per-Gaussian latent branch, coarse spatial branch, feature fusion.
3. HCD codec and rendering path: bottleneck compression, decoding, screen-space refiner.
4. Supervision: decoded-feature loss, reconstruction loss, FDH auxiliary loss, optional semantic heads.
5. Training schedule: no-FDH warm start, ws240 transition, FDH refinement.

## Experiments section outline

### Main benchmark

Use LERF-OVS as the main-table benchmark with four scenes and published baselines LERF, LangSplat, and LEGaussians.

### Supporting tasks

Use Replica room_0 depth and semantic visualizations as evidence that the reconstructed feature field supports geometry-aware and semantic downstream use.

### Required ablations

1. No-FDH vs FDH.
2. ws240 warm start vs direct FDH training.
3. Temperature and heatmap upsampling sensitivity.
4. Small-object analysis on figurines.
5. Efficiency and memory trade-offs.

## Figure and table checklist

1. Main grounding table: output/radio_gs/reports/paper_submission_main_table.tex
2. Result provenance audit: output/radio_gs/reports/paper_submission_result_audit.md
3. Benchmark target sheet: output/radio_gs/reports/paper_benchmark_targets.md
4. Frozen figure shortlist: output/radio_gs/reports/figure_shortlist.md
5. Replica qualitative figure: output/radio_gs/paper_figures/fig_room0_pipeline_comparison.png
6. LERF-OVS qualitative grids: output/radio_gs/paper_figures/*_grounding_vis/

## Writing rules before freeze

1. Only use exact numbers that have a clear rendered-JSON source or a verified reproduced evaluation log.
2. If an internal report number cannot be tied to a concrete run, treat it as provisional and keep it out of the paper body.
3. Keep the paper framed as feature reconstruction and scene understanding, not as a pure depth or rendering paper.
4. Separate published baselines from internal reproduced baselines in every table.
