# SAM3 Decoder Bridge and Direct-Field Audit, 2026-05-22

This audit records the outcome of the 2026-05-22 method-level checks for three
paper-facing risks: official SAM3 decoder usage, compact direct-field strength,
and ScanNet VALA-aligned ScanNet-8 protocol completeness.

## Official SAM3 Decoder Bridge

Implemented code paths:

- `radio_gs/models/sam3_decoder_bridge.py`
- `radio_gs/scripts/train_sam3_decoder_bridge.py`
- `tests/test_sam3_decoder_bridge.py`

The bridge maps RADIO/CTF feature maps into the official SAM3 backbone output
schema (`vision_features` and three-level `backbone_fpn`) and then calls the
frozen official SAM3 processor/decoder. This is distinct from the promoted
`sam3_box` boundary readout, which runs official SAM3 on the RGB evaluation
image with a box prompt derived from the 3D selection mask.

Smoke results on `figurines`:

| Run | Source | Train frames | Eval frames | Official RGB SAM3 mIoU | Bridge mIoU |
|---|---|---:|---:|---:|---:|
| `figurines_teacher_smoke` | frame-wise RADIO reference | 4 | 1 | 0.4349 | 0.0000 |
| `figurines_teacher_thr0_train8_e4` | frame-wise RADIO reference | 8 | 1 | 0.7489 | 0.0010 |

Conclusion: the official decoder bridge is executable and tested, but the
current simple backbone-output regression is a negative result. It should not be
promoted as evidence that reconstructed RADIO/CTF features can directly drive
the official SAM3 decoder. The paper-facing SAM3 claim should remain scoped to
`SAM3-adaptor/cache supervision` and `frozen RGB SAM3 boundary readout`.

## Compact Direct-Field Update

Implemented training support:

- Direct-point view-count weighting:
  `direct_point_view_count_weighting={none,log,clipped_log}`.
- Weighted direct-point feature, summary, adapter, and text-distillation losses.
- Optional teacher-text supervised contrast:
  `direct_point_text_contrast_weight`.

Validation tests:

- `tests/test_direct_point_supervision.py`

Additional VPR-to-field adapter experiments:

| Scene | Run | Best tag | mIoU | Acc@0.25 | Boundary F | Trimap IoU | Promotion |
|---|---|---:|---:|---:|---:|---:|---|
| figurines | `vpr_field_20260522_fig_prompt_rank_nowt` | `thr0p2` | 0.0532 | 0.0893 | 0.0708 | 0.0306 | no |
| waldo_kitchen | `vpr_field_20260522_waldo_clipped_rank` | `thr0p4` | 0.2606 | 0.5000 | 0.3173 | 0.1556 | no |

Relevant previous compact-field references:

| Scene | Run | Best tag | mIoU | Acc@0.25 | Boundary F | Trimap IoU |
|---|---|---:|---:|---:|---:|---:|
| figurines | `lerf_direct3d_fig_vpr_field_consistency_p5recompute_20260516` | `thr0p1` | 0.5151 | 0.6607 | 0.6712 | 0.2671 |
| waldo_kitchen | `lerf_direct3d_vpr_field_consistency_weighted_promptens_scene_a05_max05_20260515` | `thr0p35` | 0.2560 | 0.5000 | 0.3442 | 0.1838 |

Conclusion: the new `waldo_kitchen` run gives a small mIoU gain over the previous
compact-field reference, but the `figurines` prompt/rank/no-weighting variant is
strongly negative. The paper should keep the existing stronger compact-field
evidence and should not demote VPR from the main direct-3D readout based on this
round alone.

## ScanNet VALA-aligned ScanNet-8 Protocol Check

The paper-facing ScanNet results cover the requested eight scenes:

`scene0000_00`, `scene0062_00`, `scene0070_00`, `scene0097_00`,
`scene0140_00`, `scene0347_00`, `scene0400_00`, `scene0590_00`.

Current local macro results. Only the DINO-CV contextual kNN row is
paper-facing; the earlier Gaussian-index and non-DINO contextual rows are
legacy diagnostics:

| Row | Split 19 mIoU/mAcc | Split 15 mIoU/mAcc | Split 10 mIoU/mAcc |
|---|---:|---:|---:|
| Gaussian-index direct point-query (legacy) | 0.3583 / 0.6006 | 0.3618 / 0.6152 | 0.4367 / 0.6998 |
| Contextual kNN, scene-mean alpha=0.5 (legacy) | 0.3677 / 0.5997 | 0.3748 / 0.6181 | 0.4562 / 0.7008 |
| DINO-CV contextual kNN, scene-mean alpha=0.5 | 0.3704 / 0.6017 | 0.3771 / 0.6198 | 0.4585 / 0.7032 |

Protocol hardening added:

- `build_scannet_vala8_report.py` checks exact scene set and expected source
  arguments.
- `validate_final_rows_registry.py` checks exact VALA8 scene list, source
  protocol arguments, and recomputed macro metrics for paper-facing rows.

Fresh verification:

```text
86 passed in 8.01s
final_rows registry ok
paper claims ok
```
