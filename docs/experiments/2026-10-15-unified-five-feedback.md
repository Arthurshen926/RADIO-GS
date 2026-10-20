# Unified compact field: complete five-benchmark feedback

Branch: `codex/unified-compact-multiteacher`.

The 30-scene compact baseline and both transient-hypothesis controls completed
prediction and scoring on the entire declared cohort. Every scene uses one 96D
Gaussian latent, supervised by SigLIP2, DINOv2 and SAM under the same recipe.
Weights are fitted separately per scene. This is a complete development-cohort
evaluation, not an independent final test or a recovery of v10 performance.

## Measured results

Values below are percentages. `peak` and `pooled` reuse the exact baseline fields.
Both discover the same eight-or-fewer transient member hypotheses. They differ
only in ranking by anchor identity versus membership-weighted region identity.
Instance requests select one hypothesis; union requests combine admitted sets.

| Task / fixed readout | Baseline | Peak | Pooled |
|---|---:|---:|---:|
| LERF2D member IoU | 19.49 | 19.61 | 16.72 |
| LERF2D member AP | 69.83 | 73.95 | 55.16 |
| LERF3D local selected-set projection IoU | 11.27 | 10.98 | 10.00 |
| NVOS retained-response IoU | 62.25 | 62.25 | 62.25 |
| ScanNet native mIoU, 19 / 15 / 10 classes | 36.57 / 37.63 / 45.85 | identical | identical |
| UQIS text AP / actual-set IoU | 19.91 / 9.78 | identical | 14.00 / 7.16 |
| UQIS image AP / actual-set IoU | 31.41 / 18.43 | identical | 30.99 / 18.17 |
| UQIS 2D-point AP / actual-set IoU | 58.81 / 30.62 | identical | identical |
| UQIS 3D-point AP / actual-set IoU | 58.34 / 30.55 | identical | identical |

LERF has four scenes and 208 query-view observations; NVOS has eight cases over
seven fields; ScanNet has ten rooms; UQIS has nine rooms and 268 queries over 67
targets. Six unsupported 2D-point and eight unsupported 3D-point UQIS queries
remain in the denominator. LERF3D here is local projection of the selected set,
not a true 3D-ground-truth benchmark. ScanNet's native complete-menu partition
was declared primary before baseline scoring; raw member argmax is diagnostic.

The peak control's LERF AP change is +4.12 pp, with a paired four-scene bootstrap
95% interval of [-2.00, 11.61] pp. Its selected-set projection changes -0.28 pp.
This does not establish a robust improvement. Pooled versus peak reduces LERF
member IoU on all four scenes and AP by 18.79 pp; their AP interval is
[-26.13, -11.74] pp. These small-cohort development intervals do not establish
generalization. No arm is promoted, and no per-task winner is assembled.

## Attribution supported by interventions

1. **Feature compression is not a universal explanation.** Compared with the
   older same-cohort high-capacity common system, native ScanNet results are
   close and compact UQIS image/point AP is higher. Those older comparisons also
   change objectives/readouts, so they cannot isolate compression causally.
   The substantial v10 gap remains, with different source/supervision budgets
   for some v10 tasks; see the attribution JSON rather than treating v10 as a
   matched ablation.
2. **Candidate availability and candidate selection are separate.** On UQIS,
   the eight-candidate set contains a target anchor for 32/67 text and 41/67 image
   queries. Peak selects a correct anchor for 14 and 26 respectively; pooled
   selects only 7 and 23. More proposals have not delivered usable identity
   transfer, and region averaging over these predicted extents is inadequate.
3. **Correct identity does not guarantee correct extent.** Every supported
   UQIS point query starts on the correct target, yet 18/61 valid 2D-point and
   16/59 valid 3D-point queries expand to a set whose dominant instance is wrong.
   Perfect identity and precision within the observed carrier has a loose
   GT-oracle IoU ceiling of 87.43%, versus about 30.6% measured point-set IoU.
   This ceiling is diagnostic only, not an achievable predictor or training
   target. Coverage loss alone cannot explain the result.
4. **The explicit/inferred flag is not the main source-domain failure.** Across
   all 30 fields, held-out partial-source IoU using real positive-ray support
   versus retrieved support is 89.67/62.49 (LERF), 92.98/83.58 (NVOS),
   81.66/43.15 (ScanNet), 79.24/35.63 (UQIS). Swapping only the flag has a smaller
   effect. These are partial source constraints, not whole-instance scores.

The pooled control does not disprove contextual identity reasoning in general:
its context comes from an inaccurate member decoder, and its averaging rule is
only one estimator. Conversely, retrieval recall does not prove that the needed
extent function is present in the compact latent.

## Consequence for the next method change

The next priority is a matched **member-function transfer** experiment over the
same complete cohort. Keep one jointly supervised compact latent and a common
decision contract. Establish a source-only teacher advantage for legal support
queries first, then transfer actual support-to-set and set-to-visible-projection
behavior, including positive, negative and unknown domains. The validation must
exercise legal point/ray support as well as inferred identity support; a fall in
balanced sampled-ray BCE alone is insufficient evidence.

Two concrete implementation deficiencies need separate controls: retrieved
training supports are assigned the source region's target without verifying
their association, and member training only checks sampled known rays at a fixed
scale rather than validating complete visible extent. Source SAM region scales
currently affect negative-region selection but are not used as the member
training scale. These observations are hypotheses about causal mechanisms,
not yet measured benefits of a repair. Invalid association, hierarchical scope
ambiguity, incomplete source coverage and optical set rendering must not be
collapsed into a single generic “more training” diagnosis.

Keep native identity preservation checks active throughout, use the same recipe
on all scene families, seal all five prediction sets before feedback, and retain
failures/unsupported queries. Teacher/source observations remain construction
inputs; they are not additional query-time feature fields or an answer bank.

## Reproducibility and storage

Study: `/mnt/pool/sqy/results/RADIO-GS/output/unified_five_20261015/study`.
The two controls sealed every prediction before either new score was opened.
Matched candidate statistics, unchanged native readouts and explicit-input
outputs passed exact replay checks. NVOS predictions were reused with explicit
provenance and rescored; the other families were predicted by each control.
Five mechanism/cold-load tests pass, including querying with source file access
denied. The controls add no model weights or teacher copies. Model payloads
remain 4,675,112,831 bytes for all 30 fields. At comparison summary time the study
is 10,583,840,773 bytes (9.86 GiB), within its unchanged 16 GiB hard budget.
This excludes pre-existing teacher caches and query encoders, and is not the
whole repository's disk usage.

Evidence:
- [Complete comparisons, paired intervals and identity counts](2026-10-15-unified-hypothesis-results.json)
- [Baseline, v10 references and support/coverage attribution](2026-10-15-unified-five-attribution.json)
- `STUDY/hypothesis_comparison/{design,prediction_barrier,summary,identity_audit}.json`
- `STUDY/hypothesis_comparison/source_snapshot.zip` and `validation.json`
- `STUDY/support_diagnostic/` (source-only flag interventions and post-scoring GT diagnostics)
