# VPR-to-field consistency and confidence selector audit, 2026-05-15

## Goal

Push two paper-level optimization directions for LERF-OVS direct 3D object selection:

1. label-free confidence-based primitive selection, intended to improve mask boundaries without using GT masks;
2. VPR-to-field consistency, intended to compress registered multiview VPR evidence back into the compact direct Gaussian-center field.

Both directions keep the existing GaussFM framework and avoid per-scene object-mask tuning.

## Implementation changes

- `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
  - added `score_margin`, `score_ratio`, and `entropy_score` confidence selectors;
  - added registered VPR summary-feature caches;
  - added point summary adaptor blending and valid-mask gating for direct Gaussian-center evaluation;
  - made score-cache metadata reject mismatched point-summary-adaptor settings.
- `radio_gs/scripts/train_scannet_point_summary_adapter.py`
  - supports registered summary-feature teacher caches;
  - supports LERF text categories from teacher-cache metadata;
  - supports field-level fine-tuning via `--train_field`.

## Results

All direct-3D rows below use the same softmax-threshold selector family, 0.5% selection floor, 1.8% selection cap, and GT-free RGB boundary snap unless noted. Rows marked as diagnostic best choose the best threshold per scene from the sweep; the fixed `thr0p25` cache row is the paper-facing registered VPR reference.

| Readout / selector | Figurines | Ramen | Teatime | Waldo | Macro mIoU | Macro Acc@0.25 | Macro Acc@0.50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw Gaussian-center, diagnostic best | 0.0014 | 0.0059 | 0.1575 | 0.0185 | 0.0458 | 0.0707 | 0.0551 |
| VPR-to-field consistency, direct field, blend 0.5, diagnostic best | 0.4877 | 0.4381 | 0.5103 | 0.2115 | 0.4119 | 0.5876 | 0.4621 |
| Registered VPR readout, fixed `thr0p25` cache eval | 0.5307 | 0.5796 | 0.5659 | 0.2433 | 0.4799 | 0.6760 | 0.5480 |
| Registered VPR readout, per-scene diagnostic best | 0.5327 | 0.5822 | 0.5662 | 0.2463 | 0.4819 | 0.6715 | 0.5428 |
| Confidence-margin selector, per-scene diagnostic best | 0.5123 | 0.5659 | 0.4728 | 0.2338 | 0.4462 | 0.6282 | 0.5002 |

## Interpretation

- Confidence selectors are negative as a main protocol choice. The best margin sweep improves over weak selectors but remains below the fixed global threshold row.
- VPR-to-field consistency is positive and method-relevant. It lifts the direct Gaussian-center readout from 0.0458/0.0707 to 0.4119/0.5876, showing that registered multiview evidence can be compressed back into the student field.
- The streamed registered VPR readout remains stronger than the compressed direct-field readout, especially on Ramen and Waldo Kitchen. The paper should therefore use registered VPR as the main direct-3D row and VPR-to-field consistency as evidence for student field usability, not as a replacement for the registered readout.
- Teatime field-level training with batch size 65536 hit a CUDA illegal-memory-access failure. The same setting completed with batch size 32768, and the resulting metrics are included above. This looks like kernel/batch-size stability rather than a formulation issue, because the same code path completed on the other scenes.

## Paper-facing conclusion

The strongest defensible claim is:

> GaussFM supports a strong registered primitive-level direct-3D readout via VPR, and a label-free VPR-to-field consistency stage substantially improves the direct student field itself. This strengthens the reusable foundation-feature-field claim, while the remaining gap to streamed VPR and the Waldo failure mode argue against claiming universal primitive-level SOTA without same-evaluator baseline reruns.
