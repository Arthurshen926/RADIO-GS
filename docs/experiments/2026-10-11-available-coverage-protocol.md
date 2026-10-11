# Actual member-domain correction, fixed before construction

Correct only the geometric domain of sampled coverage from observed to
observed OR semantic_measure.supported, matching the native member readout.
Keep source pseudo-label construction on observed; semantic support is not
foreground supervision. Same parent fields, 72 source views, 16x16 fixed samples,
scope atoms, alpha conventions and tolerances as the previous construction.
No benchmark queries, predictions or target labels enter compilation.

Keep seven previous independent member conditions and add semantic-only and
unavailable-only conditions before the continuous condition. All nine compare
against relaxed rendering with the explicit available domain. Additionally all
eight binary conditions compare directly against selected_alpha on the actual
subset (no observed-domain masking inside this renderer). Both comparisons
require maximum absolute error <=1e-4 and exact silhouette decisions at10.5/255.
Continuous all-row mean-opacity gradient error <=2e-4. Do not relax gates.

Fresh-process, Python-source-denied reload replays all nine conditions and
checks every parent tensor unchanged. This remains sampled, view-grouped,
uncompressed functional state. It does not protect unsampled positive extent.

Subsequent member evaluation must hold native semantic initialization, native
region acceptance, ordering, one-pass policy and source budget fixed, comparing
old/new domain for both frozen-weight and joint-opacity controls. All four scenes
must predict before any target scoring. No per-query fitting or benchmark answer
compilation. A construction pass alone is not performance recovery, promotion,
five-benchmark completion, or paper closure. Prior semantic initialization is
42.9498%, old observed-domain joint43.5621%, fixed-weight43.9940% in four-scene
development evaluation; these are historical comparisons, not new results.
