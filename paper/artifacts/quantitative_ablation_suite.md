# Unified Quantitative Ablation Suite

This report consolidates the paper-facing ablations into one contribution ranking. Rows are grouped by protocol status so diagnostic readouts are not mistaken for core compact-field evidence.

Score = mean positive delta over the listed metrics. Negative secondary deltas remain visible in the metric details.

## Contribution Ranking

| Rank | Contribution | Task | Reference -> Variant | Score | Metric deltas | Status |
| ---: | --- | --- | --- | ---: | --- | --- |
| 1 | VPR-to-field primitive feature registration | LERF Direct3D | raw Gaussian-center score -> VPR-to-field compact primitive score | 0.4415 | mIoU: 0.0460->0.4120 (+0.3660)<br>Acc@0.25: 0.0710->0.5880 (+0.5170) | core direct-field evidence |
| 2 | Foundation-space reconstruction codec | LERF rendered architecture | w/o CTR -> full GaussFM seed-7 model | 0.2760 | LocAcc: 0.5310->0.8580 (+0.3270)<br>mIoU: 0.2600->0.4850 (+0.2250) | core architecture |
| 3 | Final rendered-view readout vs frame-wise RADIO | 2D frame-wise-RADIO-vs-field feature usability | frame-wise RADIO reference -> GaussFM rendered field + feature-only boundary readout | 0.0934 | LocAcc: 0.7985->0.8598 (+0.0613)<br>mIoU: 0.4634->0.5889 (+0.1255) | main claim evidence |
| 4 | FGC/FDH geometry-aware warm-start | LERF rendered architecture | w/o FGC warm-start -> full GaussFM seed-7 model | 0.0585 | LocAcc: 0.8020->0.8580 (+0.0560)<br>mIoU: 0.4240->0.4850 (+0.0610) | core architecture |
| 5 | Official SAM3 box boundary diagnostic | LERF Direct3D | score-component guarded compact readout -> frozen official SAM3 box readout | 0.0346 | mIoU: 0.5014->0.5705 (+0.0691)<br>Acc@0.25: 0.7044->0.6835 (-0.0209) | diagnostic, not core method |
| 6 | Direct3D prompt ensemble + component support policy | LERF Direct3D | strict single-prompt pure one-map -> score-component guarded compact readout | 0.0342 | mIoU: 0.4489->0.5014 (+0.0525)<br>Acc@0.25: 0.6724->0.7044 (+0.0321)<br>Boundary-F: 0.6124->0.6305 (+0.0181) | main compact direct readout |
| 7 | DINOv3 dense matching feature readout | 2D frozen-head downstream | frame-wise RADIO reference -> GaussFM rendered field | 0.0251 | Mean score: 0.8547->0.9048 (+0.0501)<br>HitRate: 0.5723->0.5396 (-0.0327) | downstream readout evidence with caveat |
| 8 | SAM3 point-prompt feature readout | 2D frozen-head downstream | frame-wise RADIO reference -> GaussFM rendered field | 0.0237 | mIoU: 0.3700->0.4173 (+0.0473)<br>LocAcc: 1.0000->1.0000 (+0.0000) | downstream readout evidence |
| 9 | Peak-component rendered mask readout | LERF rendered-view grounding | threshold 0.60 mask -> threshold 0.60 + peak component | 0.0232 | mIoU: 0.5243->0.5707 (+0.0464)<br>LocAcc: 0.8712->0.8598 (-0.0114) | readout policy |
| 10 | ScanNet contextual kNN + spatial logit propagation | ScanNet VALA8 direct point-query | DINO-CV contextual kNN alpha=0.5 -> k16/cand80 + scene alpha 0.45 + spatial k12/a1 | 0.0121 | split19 mIoU: 0.3704->0.3806 (+0.0102)<br>split19 mAcc: 0.6017->0.6129 (+0.0112)<br>split15 mIoU: 0.3771->0.3871 (+0.0100)<br>split15 mAcc: 0.6198->0.6315 (+0.0117)<br>split10 mIoU: 0.4585->0.4711 (+0.0126)<br>split10 mAcc: 0.7032->0.7200 (+0.0168) | supporting ScanNet diagnostic readout |
| 11 | VFA view-space aligner | LERF rendered architecture | w/o VFA -> full GaussFM seed-7 model | 0.0115 | LocAcc: 0.8400->0.8580 (+0.0180)<br>mIoU: 0.4800->0.4850 (+0.0050) | core architecture |
| 12 | Hybrid Gaussian code field | LERF rendered architecture | w/o HGCF -> full GaussFM seed-7 model | 0.0095 | LocAcc: 0.8390->0.8580 (+0.0190)<br>mIoU: 0.5070->0.4850 (-0.0220) | core architecture with tradeoff |
| 13 | Feature-only SAM3 boundary readout | LERF rendered-view grounding | peak-component core -> feature-only SAM3 boundary readout | 0.0091 | mIoU: 0.5707->0.5889 (+0.0182)<br>LocAcc: 0.8598->0.8598 (+0.0000) | readout policy |

## Interpretation

- **VPR-to-field primitive feature registration:** Largest direct-3D jump; primitive features need multiview registered support before selection is meaningful. Source: `paper/lerf_vpr_field_consistency_table.tex`.
- **Foundation-space reconstruction codec:** Dominant architectural dependency; direct 1x1 projection cannot recover RADIO-compatible features. Source: `paper/lerf_component_ablation_table.tex`.
- **Final rendered-view readout vs frame-wise RADIO:** Shows the reconstructed scene field improves text-grounding usability over frame-wise RADIO features under the same evaluator. Source: `paper/artifacts/final_rows.yaml`.
- **FGC/FDH geometry-aware warm-start:** Second largest controlled architecture contribution after CTR/HCD. Source: `paper/lerf_component_ablation_table.tex`.
- **Official SAM3 box boundary diagnostic:** Shows remaining boundary headroom, but it uses an external RGB SAM3 decoder and should not be counted as compact-field evidence. Source: `paper/lerf_direct_3d_selection_table.tex`.
- **Direct3D prompt ensemble + component support policy:** Primary Waldo/small-object recovery mechanism for the compact direct row. Source: `paper/artifacts/lerf_direct3d_compact_readout_ablation_20260528.md`.
- **DINOv3 dense matching feature readout:** Feature similarity improves, but hit-rate caveat shows DINO topology remains a separate concern. Source: `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md`.
- **SAM3 point-prompt feature readout:** Strongest SAM3-adaptor primary-metric improvement with no point-prompt localization loss. Source: `paper/artifacts/teacher_vs_ctfgs_2d_usability_20260525.md`.
- **Peak-component rendered mask readout:** Large boundary/support gain; it trades a small heatmap-peak drop for much better connected mask support. Source: `paper/artifacts/final_rows.yaml`.
- **ScanNet contextual kNN + spatial logit propagation:** Small but consistent VALA8 gains across all reported ScanNet splits. Source: `docs/experiments/2026-05-24-direct-field-joint2d3d-optimization.md`.
- **VFA view-space aligner:** Moderate localization gain; smaller region-overlap effect than CTR or FGC. Source: `paper/lerf_component_ablation_table.tex`.
- **Hybrid Gaussian code field:** Improves peak stability but not raw mIoU in this controlled table; keep as a tradeoff rather than a universal gain. Source: `paper/lerf_component_ablation_table.tex`.
- **Feature-only SAM3 boundary readout:** Boundary refinement improves region overlap without changing heatmap localization. Source: `paper/artifacts/final_rows.yaml`.

## Missing Same-Protocol Follow-Ups

- **P0** optional full 2x2 Direct3D factorial for no-prompt + RGB/score guard: the strict no-prompt/no-RGB cell is now filled; the remaining optional cell would isolate whether RGB/score support still helps without the prompt ensemble.
- **P1** ScanNet module-removal training ablation on all VALA8 scenes: readout ablations are complete enough for the paper, but full training-time removals for DINO-CV/context losses are not all positive or promoted.
- **P1** multi-head DINO/SAM/SigLIP2 removal under the 2D frozen-head benchmark: frame-wise-RADIO-vs-field downstream wins are recorded, but per-head removal deltas are not yet a clean single-table factorial.
- **P2** LERF final-row architecture ablation after feature-only SAM3 boundary readout: core architecture table uses controlled seed-7 rendered features; final boundary readout is measured separately.
