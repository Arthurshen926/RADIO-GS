# Rendered Boundary and Direct3D Support Interaction Audit

Date: 2026-06-10.

## Question

Do rendered-view boundary calibration and Direct3D support calibration improve
each other through shared compact-field training?

## Code-level status

The training code contains both families of supervision hooks:

| Mechanism | Code support | Meaning |
|---|---|---|
| Direct primitive support distillation | `radio_gs/scripts/train_feature_field.py` uses `direct_point_query_support_distill_weight` and related config fields | Can train compact primitive scores/features with query-support pseudo targets |
| Foundation-cache mask/boundary loss | `radio_gs/scripts/train_feature_field.py` uses `foundation_cache_mask_boundary_weight` | Can train rendered features with cached mask-logit/boundary signals when the cache is available |
| Prompt-conditioned SAM3-adaptor boundary head | `radio_gs/scripts/train_prompt_conditioned_sam3_mask_head.py` | Trains a feature-only mask head/readout, not necessarily the compact Gaussian field itself |

## Promoted-row status

The current promoted paper-facing rows should be described carefully:

| Row | Current role | Backpropagates into compact field in promoted evaluation? |
|---|---|---|
| LERF rendered feature-only SAM3 boundary readout | Inference/readout refinement using the trained prompt-conditioned mask head | No; it refines masks from reconstructed features at evaluation |
| LERF Direct3D support-calibrated compact readout | Query-time support policy using compact primitive scores, prompt ensemble, and score/color component support calibration | No; the support policy is a fixed GT-free selector at evaluation |
| Existing direct support distillation variants | Training-side evidence that support targets can supervise the compact field | Yes, when enabled in training configs |
| Existing foundation-cache boundary variants | Training-side evidence that boundary/mask-logit caches can supervise the compact field | Yes, when enabled in training configs |

Therefore, the current evidence supports this wording:

> Rendered boundary calibration and Direct3D support calibration are unified as
> object-support formation mechanisms around the compact foundation-feature
> field and improve their corresponding protocols.

It does not yet justify this stronger wording:

> Rendered boundary calibration and Direct3D support calibration have been
> proven to mutually improve each other through shared compact-field training.

## Existing positive evidence

| Evidence | Artifact | Result |
|---|---|---|
| Rendered boundary readout improves LERF 2D grounding | `paper/artifacts/lerf_rendered_grounding_feature_sam3_boundary_20260525.json` | Main rendered row reaches 0.8598 LocAcc / 0.5889 mIoU |
| Direct3D support policy improves compact direct selection | `paper/artifacts/lerf_direct3d_score_component_guard_20260528.md` | Score-component guard reaches 0.5014 mIoU / 0.7044 Acc@0.25 |
| Boundary qualitative examples | `paper/figures/lerf_rendered_boundary_calibration_qualitative.png` | Ramen examples show +0.11 to +0.20 per-query IoU from coarse support to feature-only boundary |
| Direct3D support qualitative examples | `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png` | Small/fragmented object support is visibly more stable than the base compact direct mask |

## Missing causal experiment

To prove mutual training benefit, run the following fixed 2x2 experiment with
the same seed, training schedule, and evaluation scripts:

| Training variant | LERF rendered LocAcc/mIoU | LERF Direct3D mIoU/Acc@0.25 | Interpretation |
|---|---:|---:|---|
| Base compact field | -- | -- | Control |
| + rendered boundary supervision only | -- | -- | Tests whether 2D boundary training helps Direct3D |
| + Direct3D support distillation only | -- | -- | Tests whether primitive support training helps rendered-view grounding |
| + both | -- | -- | Tests complementarity |

Until this table is run, paper text should claim unified support formation and
protocol-specific gains, not proven cross-task causal improvement.

## 2026-06-10 implementation audit

The first boundary-only run exposed two implementation bugs that prevented the
foundation-cache branch from being a valid 2D boundary-supervision factor:

| Issue | Root cause | Fix |
|---|---|---|
| SAM3 foundation cache was configured but the logged `foundation_cache` loss stayed zero | The resolver did not try the local five-digit cache naming pattern (`frame_00006.pt`) | `radio_gs/training/feature_training_utils.py` now checks `%05d` variants such as `frame_{frame_int:05d}.pt` |
| Once cache paths resolved, training crashed when reading projectors | `nn.ModuleDict` does not implement `.get(...)` | `radio_gs/models/foundation_cache.py` now uses explicit membership lookup before indexing |

Fresh smoke verification after the fixes loaded
`output/radio_gs/foundation_cache_sam3_modelscope/ramen/frame_00006.pt`, found
the `sam3` head and mask logits, and returned a nonzero foundation-cache loss
(`0.0240797`) with no skipped heads.

## 2x2 training-side result

All four Ramen 12-epoch fine-tunes use the same warm start,
`direct_point_teacher_cache`, seed, schedule, and train/val split. The support
and boundary factors differ only in the intended supervision switches.

| Variant | Boundary supervision | Direct3D support supervision | Best val cosine | Active loss evidence |
|---|---:|---:|---:|---|
| `00_none_ft12` | no | no | 0.7932 | `foundation_cache=0`, `support=0`, `render_consistency=0` |
| `01_support_ft12` | no | yes | 0.7913 | `support=0.3874`, `render_consistency=0.0948`, `proposal=0.0015` |
| `10_boundary_ft12_cachefix2` | yes | no | 0.7947 | `foundation_cache=0.000051`, support losses zero |
| `11_both_ft12_cachefix2` | yes | yes | 0.7927 | `foundation_cache=0.000051`, `support=0.3896`, `render_consistency=0.0921`, `proposal=0.0015` |

Training-side conclusion: the two supervision families now both attach to the
compact-field training graph. Boundary-only gives a small positive validation
cosine gain on this Ramen smoke run, while support-only and both slightly reduce
the feature-cosine proxy. Because the paper claim is about downstream support
formation rather than only feature cosine, fixed-protocol LERF Ramen Direct3D
and rendered-grounding evaluations are required before claiming mutual
cross-task improvement.

## Fixed-protocol downstream checks launched

After the training runs completed, I launched a same-protocol Ramen Direct3D
evaluation for all four checkpoints:

- evaluator: `radio_gs/scripts/eval_lerf_direct_3d_selection.py`
- scene: `ramen`
- selection: `score_threshold` with `0.60,0.65`
- scoring: `softmax_scene`
- direct head: point-summary adapter enabled, opacity valid mask
- postprocess: `rgb_grabcut_component_guard`
- output root:
  `output/radio_gs/mutual_training_eval/direct3d_ramen_20260610/`

I then ran the rendered-view grounding counterpart with the same four
checkpoints, same Ramen scene, same prompt ensemble, `softmax_scene`, and fixed
`iou_threshold=0.6`.

## Fixed-protocol downstream results

The Ramen Direct3D batch completed under the same compact direct-field protocol
used for the paper-facing guarded row. For each checkpoint, the table reports
the better of the two fixed thresholds (`thr0p60`, `thr0p65`) by mIoU.

| Training variant | Direct3D best threshold | Direct3D mIoU | Acc@0.25 | Acc@0.50 | Boundary-F | Trimap IoU |
|---|---|---:|---:|---:|---:|---:|
| `00_none_ft12` | `thr0p60` | 0.4368 | 0.6479 | 0.5493 | 0.6189 | 0.2113 |
| `01_support_ft12` | `thr0p60` | 0.3950 | 0.5352 | 0.4507 | 0.5545 | 0.2009 |
| `10_boundary_ft12_cachefix2` | `thr0p60` | 0.4445 | 0.6620 | 0.5775 | 0.6213 | 0.2168 |
| `11_both_ft12_cachefix2` | `thr0p60` | 0.3889 | 0.5352 | 0.4507 | 0.5416 | 0.1970 |

The rendered-view Ramen grounding counterpart used the same prompt ensemble,
`softmax_scene`, and fixed `iou_threshold=0.6`, without mask refinement:

| Training variant | Rendered LocAcc | Rendered mIoU | Frame-wise RADIO LocAcc | Frame-wise RADIO mIoU |
|---|---:|---:|---:|---:|
| `00_none_ft12` | 0.8873 | 0.5934 | 0.8873 | 0.5450 |
| `01_support_ft12` | 0.9014 | 0.5965 | 0.8873 | 0.5450 |
| `10_boundary_ft12_cachefix2` | 0.8873 | 0.5963 | 0.8873 | 0.5450 |
| `11_both_ft12_cachefix2` | 0.9014 | 0.5938 | 0.8873 | 0.5450 |

## Interpretation

This 2x2 smoke test does not support a strong claim that rendered boundary
calibration and Direct3D support calibration mutually improve each other through
shared training.

What it does support:

- The SAM3 foundation-cache boundary branch is now genuinely active in compact
  field training.
- Boundary supervision gives a small positive Ramen Direct3D gain over the
  matched base (`0.4368 -> 0.4445` mIoU and `0.6479 -> 0.6620` Acc@0.25).
- Direct3D support supervision gives a small positive Ramen rendered-view gain
  (`0.5934 -> 0.5965` mIoU and `0.8873 -> 0.9014` LocAcc).

What it does not support:

- Support-only training does not improve Ramen Direct3D in this short schedule.
- Combining both losses does not produce additive gains; it regresses the
  Direct3D smoke result.
- Therefore, paper text should not claim proven mutual training synergy.

Recommended paper wording:

> Boundary and support calibration are implemented as compatible supervision
> and inference mechanisms around the same compact feature field. In a Ramen
> 2x2 smoke study, boundary supervision improves direct 3D selection, while
> support supervision slightly improves rendered grounding. We keep the final
> promoted rows protocol-specific and leave full multi-scene joint-training
> synergy as future work.
