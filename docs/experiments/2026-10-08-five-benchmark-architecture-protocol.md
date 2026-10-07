# Complete the five-benchmark architecture baseline before local optimization

The user's 2026-10-08 instruction changes the work order: first run the whole
new architecture on all five existing benchmark cohorts, then diagnose its gap
to saved v10. Earlier local promotion gates remain historical experiment results;
they no longer delay this full-cohort integration. No new negative-rule, support
mixture, compression, speed or threshold study is part of this run.

Freeze the accepted physical field at fb3c768: native SigLIP/DINO coordinates,
32-dimensional scale-conditioned group features, single-support source fitting,
physical member energy, training-eligible inferred anchors, eight candidates,
product signed exclusion, threshold0.5 and2000 updates. These settings are shared
across scenes. Inference uses the exported Gaussian field and the current input.
No old label-trained UQIS member model, MaskAdapter prior or source-query solver
is imported. This is the high-capacity architecture baseline; compression waits.

Complete roster: four LERF scenes, both retained 2D semantic and actual Gaussian
subset projection metrics; ten ScanNet OVS rooms; eight NVOS tasks; nine UQIS
rooms with all67 targets and268 independent text/image/2D/3D queries. Missing
registered support produces a recorded empty prediction and remains in the
denominator. Infrastructure errors must be repaired, not treated as low scores.

Retain the already built four LERF fields and fern field. Construct the remaining
fields with20 uniformly spaced legal mapping observations, quartile positions5/15
held out from feature fitting (18 training views). Fern retains its established
12-view source budget. Source-frame budgets differ from v10 and must be reported.
Horns left/centre share one field because their geometry and source authority
are identical; only the authorized current prompt differs.

NVOS keeps saved geometry/cameras and raw signed prompts, using default scale.
New construction RGB is resized to longest side1008; UQIS mapping RGB to width640.
Generate query-free SAM3 grid12 masks using the available frozen checkpoint;
retain all masks after existing quality/deduplication, then the same .002-.6
native crop eligibility. Reuse bound ScanNet query-free SAM/native crop caches
under that eligibility rule. DINO and Gaussian assignments are compiled afresh.
No benchmark reference frame or withheld neighbour enters UQIS feature building.

ScanNet and UQIS use their declared unlabelled official mesh geometry track.
ScanNet retains its saved surface Gaussians. UQIS converts the public mesh to
surface Gaussians using the existing area-derived scale/normal rule, with no
instance labels. Use explicit OpenCV camera tables and world units; no artificial
MoGe calibration is needed for registered metric cameras. Evaluation transfers
Gaussian predictions to mesh vertices by a fixed nearest-centre correspondence.
Report geometry transport and unobserved coverage rather than dropping points.

ScanNet category predictions compete across the complete declared menu: run the
same text-union member query per category and argmax its membership scores within
each19/15/10 menu. Native semantic argmax is a separately reported control. No
per-class thresholds, fitted native/member mixture or choice based on scores.
This first categorical member readout has no separately trained category model.

UQIS keeps original expressions, RGB crops, registered pixel/camera inputs and
world points. All four call the same member core with instance scope. Persist
membership scores and actual selected elements, then use the retained official
mesh evaluator with the original target pairing hidden from inference. Point3D
registration retains the existing0.1m tolerance; point2D uses its supplied raster.
No privileged target size, reference pose for image inputs, or extra label access.

Persist predictions for the complete cohort before new target scoring. Reuse
sealed LERF results with hashes and verify reused fern. Produce one five-task
table against frozen v10 with complete scene/query denominators and explicit
input/teacher/geometry/readout differences. Do not pick per-scene checkpoints.
Only after this baseline is complete, diagnose coverage, native observation
fusion, input identity, member extent, and final rendering/transport separately.
This is historically opened development, not an independent final evaluation.
