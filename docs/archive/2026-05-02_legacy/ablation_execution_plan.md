# Ablation Execution Plan

This document freezes the current ablation queue into an executable checklist.

## Primary Ablations

1. `No-FDH vs FDH`
   Goal: quantify the gain from frozen depth supervision under the same warm-start and epoch budget.
   Artifacts: `report_room0_v14_baseline.md`, `report_room0_v14_frozen_dh.md`, matching eval logs.

2. `Pure-Frozen Depth-Only`
   Goal: test whether removing the cross-domain frozen segmentation teacher recovers geometry while preserving semantics.
   Target runs:
   - `replica_hybrid_v14_room_0_pure_frozen_depth_only.yaml`
   - `lerf_hybrid_v14_{figurines,ramen,teatime,waldo_kitchen}_pure_frozen_depth_only.yaml`

3. `ws240 Warm-Start`
   Goal: isolate the benefit of warm-starting from the 240-epoch no-FDH solution before frozen-head refinement.
   Comparison: `*_nofdh_240ep` vs `*_fdh_ws240*`.

4. `Temperature / Heatmap Sensitivity`
   Goal: show that the reported LERF numbers are stable to modest evaluation-time temperature changes.
   Tooling: `auto_eval_lerf_sweep.py`.

5. `Small-Object Analysis`
   Goal: analyze why `figurines` is sensitive to supervision choice and feature resolution.
   Comparison: `figurines_best`, `figurines_pure_frozen_depth_only`, `figurines 2x`.

6. `Efficiency / Memory`
   Goal: report the runtime and memory cost of training/evaluation for representative settings.
   Tooling: `profile_command.sh`, `build_profile_report.py`.

## Reporting Template

For each ablation, record:

1. config path
2. checkpoint path
3. exact eval command
4. rendered metrics
5. qualitative failure/success mode
6. paper-facing takeaway

## Freeze Rule

Any number intended for the paper must be backed by a concrete artifact under `output/` and linked from a report.
