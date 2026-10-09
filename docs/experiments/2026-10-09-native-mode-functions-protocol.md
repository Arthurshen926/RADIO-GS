# Fixed native mode and current-input function transfer

Freeze three arms before construction/prediction/scoring:
1. mean_native: complete Gaussian-supported mode candidates, local mean,
   native-size query patches with fractional padding mass.
2. measure_resized: same candidates and semantic directions, full local measure,
   previous448-width query resampling with unit patch weights.
3. measure_native: same field/measure, native-size query and padding mass.

Use all9 UQIS scenes and all268 inputs per arm. The mean/measure contrast at native
input isolates the local function; the resized/native contrast at full measure
isolates the current-query preprocessing function (resolution plus padding).
Against the prior full_supervised, modes/candidate coverage/semantic binding and
the geometric2D input also change: not an isolated local-patch comparison.

All arms retain full legal1089 source frames and original18 development exclusions,
the previous joint member tensors, protected native channels and geometry. Existing
22-room instance supervision/3-room calibration remains explicit. No new training,
teacher, label-based model selection, benchmark-conditioned source construction or
threshold change. Uncompressed appearance atoms are additional field state, not a
compression claim; their cost is measured. No per-view patch raster/offsets or old
source solver in runtime. Every mode binds to an existing Gaussian support column.

All source candidates with nonempty carrier support and positive local measure are
retained. Its median-nearest Gaussian is a query-independent estimated anchor.
Text uses native mode semantic cosine; image equal semantic/local percentile ranks.
The same member function predicts extent for both; inferred anchors never clamped.
Registered3D unchanged;2D uses the already fixed first-surface adapter in all arms.
The source-only mode descriptor construction does not erase the inherited member
supervision. Text/source scope is not claimed identical to every v10 text solver.

Before public inputs,8 fixed source-derived query measures per scene compare every
mode against the original per-view match_regions operator, tolerance2e-6. Record
all removed unbound modes/atoms and median-anchor changes from the3 missing rows
in each of two scenes. Test CPU/CUDA weighted functions, ragged chunk tails, padding
mass, the same-mean/different-function counterexample, standalone export and parent
registered-query regression. Construct all18 exports before predictions; all804
predictions before any scoring. Encode current queries separately from construction.

Report all4 modalities/AP/IoU and paired9-scene intervals. Post-score labels may
diagnose candidate/identity/scope failures, never select an inference mode. Preserve
failures and raw copied-export replay; no full-five-task, final-test or default
promotion claim. A positive result verifies a capability reference, not the final
compact novel method. Retain previous strict bitwise failures as recorded.
