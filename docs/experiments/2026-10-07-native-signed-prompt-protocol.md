# Real signed prompts through the common field: fixed development protocol

This round keeps the accepted physical member energy and all training settings
unchanged. The failed native-local gate is not relaxed. No benchmark-specific
decoder, prompt-derived SAM mask, source tracking, native bypass, extra 3D labels,
query-time fitting or oracle scale is introduced.

The designated query-free NVOS source provider has 14 fern frames and no other
scene manifests. Use all 14 cached authorized frames; hold out the sorted roster's
quarter and three-quarter positions for teacher diagnostics and train on the
remaining 12. This availability-selected one-case pilot is not representative
of the full eight-case benchmark. Target frame 1 is excluded from construction;
the existing frozen geometry is reused, so the teacher holdout is feature-only.
Encode fresh native SigLIP crops with the established LERF recipe and fresh DINO
pixels, then use the existing 96x128, 32-channel, 2000-step source subset-alpha fit.

The only new input operation pools official foreground/background scribble pixels
through registered Gaussian rays. Original positive/negative masks must be
nonempty, aligned and disjoint. Area resampling to 192x256 retains thin evidence;
each ray is normalized before pixel weights are pooled into the existing Support.
Both signs may share uncertain Gaussian contributors. No hard 2D constraints or
nearest-visible fallback are allowed. The native-resolution reference RGB is not
an inference input. All source/cache files are construction-only.

Before reading the target mask, freeze all eight predictions: untrained/trained
field, each with positive-only and signed scribbles and two fixed scope policies.
The positive control uses
the raw official strokes, not v10's SAM-derived reference mask. Both arms use the
same member threshold. Scope policies are the unchanged default and the larger of
that default and twice the spatial standard deviation of positive ray centres,
in physical metres. The factor two uses the existing source-mask scale convention;
within-ray depth uncertainty is excluded. This scope amendment was fixed while
native source encoding was running, before fitting, prediction or target scoring.
It addresses information lost when multiple positive pixels are treated like one
point; no benchmark masks or negative strokes determine scale. It remains an
explicit adapter experiment, not a promoted method based on one case.
Re-render selected Gaussian opacity
at 192x256, bilinearly resize continuous alpha to 756x1008, then threshold at 0.5.
Report foreground IoU, pixel accuracy, precision/recall, selected size, both sign
coverage and deployment cost. A fresh process replay with copied field, camera
fixture and strokes must deny access to original output and NVOS dataset paths.
All arms and failures count; no arm/scale/threshold selection after scoring.

This extends validation and input coverage; it is not a new model promotion gate.
It cannot establish five-task equivalence, innovation by itself, or closure of the
v10 accuracy gap. Any discovered failure should motivate a separately fixed common
method experiment with text/image/point checks, rather than a fern-specific fix.

Pre-scoring implementation correction: the first eight predictions reached cold
replay, which caught last-bit member differences from CUDA atomic raster pooling
(selected set and projected alpha were exact). No target scoring occurred.
Preserve those aborted outputs separately and use ordered CPU float64 accumulation
for transient input registration. Re-run all eight predictions and cold checks.
Also retain the v10 full-scene observed-support-normalized member response at
756x1008 with its >=0.5 threshold and AP as a separate readout control. This is
needed for a readout-aligned v10 comparison; actual subset alpha >0.5 remains a
distinct metric. It uses the same stored membership, never another model.
These corrections were fixed before any target labels were opened. Prompt access,
field tensors and hyperparameters are unchanged. The positive-only controls read
negative strokes for validation/agreement reporting but do not use them in
membership or scale. This is an inference influence control, not negative-file
access isolation.
