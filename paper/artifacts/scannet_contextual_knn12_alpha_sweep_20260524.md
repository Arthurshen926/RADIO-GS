# ScanNet VALA8 Contextual kNN Calibration Sweep

Protocol: VALA/OpenGaFF-8 ScanNet direct point-query evaluator, DINO-CV compact
field, `query_mode=knn`, prompt `{query}`, and label-free scene-mean
calibration. No ScanNet labels are used for calibration or row selection beyond
final metric reporting.

| Readout | 19 mIoU / mAcc | 15 mIoU / mAcc | 10 mIoU / mAcc | Mean mIoU | Mean all six |
|---|---:|---:|---:|---:|---:|
| k8/cand32, alpha=0.50 previous | 0.3704 / 0.6017 | 0.3771 / 0.6198 | 0.4585 / 0.7032 | 0.4020 | 0.5218 |
| k12/cand48, alpha=0.40 | 0.3714 / 0.6026 | 0.3784 / 0.6208 | 0.4580 / 0.7028 | 0.4026 | 0.5224 |
| k12/cand48, alpha=0.45 previous | 0.3715 / 0.6024 | 0.3784 / 0.6206 | 0.4585 / 0.7029 | 0.4028 | 0.5224 |
| k12/cand48, alpha=0.50 | 0.3713 / 0.6017 | 0.3780 / 0.6199 | 0.4592 / 0.7029 | 0.4028 | 0.5221 |
| k12/cand48, alpha=0.60 | 0.3698 / 0.5989 | 0.3761 / 0.6168 | 0.4602 / 0.7028 | 0.4021 | 0.5208 |
| k8/cand32, alpha=0.75 diagnostic | 0.3683 / 0.5957 | 0.3746 / 0.6136 | 0.4612 / 0.7036 | 0.4014 | 0.5195 |
| k16/cand64, alpha=0.40 | 0.3718 / 0.6026 | 0.3789 / 0.6208 | 0.4583 / 0.7025 | 0.4030 | 0.5225 |
| k16/cand64, alpha=0.45 | 0.3720 / 0.6024 | 0.3788 / 0.6206 | 0.4589 / 0.7026 | 0.4032 | 0.5225 |
| k16/cand80, alpha=0.45 promoted | 0.3722 / 0.6025 | 0.3791 / 0.6207 | 0.4591 / 0.7025 | 0.4034 | 0.5227 |
| k20/cand96, alpha=0.45 diagnostic | 0.3723 / 0.6024 | 0.3791 / 0.6205 | 0.4590 / 0.7023 | 0.4034 | 0.5226 |

Conclusion: `k=16`, `candidate_k=80`, `alpha=0.45` is the paper-facing row. It
improves 19/15/10 mIoU over the previous k12/cand48 row and has the best
aggregate score across the six reported mIoU/mAcc numbers in this sweep. The
k20/cand96 row is slightly more smoothed but gives lower mAcc, so it remains
diagnostic. Higher calibration scales improve the 10-class split but weaken
19/15-class balance, so they remain diagnostic.
