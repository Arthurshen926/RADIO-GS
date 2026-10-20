# ADR0035: Complete signed query evidence before instance transport

2026-10-18. N12 fixed after N11 results, before N12 predictions and scoring.

N11 cannot distinguish missing query scope from genuinely irrelevant regions
without identity evidence. Test an upstream intervention: construct a fuller
reference response using the original positive AND negative scribbles, then
transport its instance identity with frozen SAM3 memory. This changes query
response construction, not teacher fields or geometry. Completeness is a
hypothesis, not established by scribble agreement.

Three reference seeds share frozen models and identical two-frame transport:
1. The original v10 positive-only SAM reference mask (matched memory control).
2. Frozen DINOv2 signed nearest-cosine affinity from reference to itself, decoded
   by SAM3 with the existing dense response prompt. Unknown patches carry no sign.
3. Same semantic response, plus 16 deterministic farthest-spread points per
   original scribble sign in SAM3. After decoding, impose all original signed
   scribble pixels exactly and leave unknown pixels untouched (primary).

Source-only LOO scale remains N09's scale; no target descriptor, target threshold
selection, task-specific points, scene-name gate or target annotation. All
reference masks use 0.5. Report reference size, signed-query violations BEFORE
clamping, and changes to the inherited reference mask; these are not GT scores.
Each seed enters the exact frozen two-frame reference-to-target memory operation.
The target uses observed RGB. The original static target prompt is replayed as
control. Actual native-ray 3D fitting, 120 Huber steps, confidence abs(2p-1), v10
initialization/observed domain, geometry and 0.5 readout remain fixed.

Nine arms: v10; static_image/static_native; legacy_memory_image/native;
semantic_memory_image/native; constrained_memory_image/native (primary).
All72 predictions precede either target GT convention. Report all-eight IoU/AP,
independent confusion counts, original memory/static replay, source-teacher
residuals, paired task effects and full-cohort spatial errors. Compare N11 as an
explicit earlier candidate; do not stitch task-wise winners. Other four tasks
remain hash-bound references, not fresh inference. Opened development cohort,
view-conditioned outputs, no default promotion or predeclared SOTA claim.

Storage continuation is explicit: do not alter the almost-full 2GiB study.
Create output/tmm_response_20261018/study with an immutable 8MiB total,
2MiB/run, 1MiB/file and 1GiB free reserve; reserve at most 1MiB for N12.
Inherit old sources/inputs by hash and snapshot new runtime sources. Prediction
RAM <=256MiB, per-task worker temporary files <=64MiB, spatial links <=128MiB.
Release all owned dense arrays after scoring; no new weights. Preserve failures.
