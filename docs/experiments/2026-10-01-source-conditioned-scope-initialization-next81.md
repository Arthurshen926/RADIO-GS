# Source-conditioned scope initialization (next81)

## Mechanism

Next80 allowed native-region tracking hypotheses to supervise membership, but
`mode=seeded` only initialized elements for queries newly admitted by **detector**
corroboration. Regional tracks were already active before that admission step,
so they did not enter the initializer. The optimizer still started from old
semantic memberships, including near-zero missing concepts and extraneous
positive elements. Source supervision and initialization were disconnected.

This experiment initializes membership from the *same* authorized observations:

- For each active view, project its existing known-pixel weights back through
  the frozen visible Gaussian operator.
- Divide projected soft positive mass by projected known mass.
- Replace membership only where that known mass exists; retain the semantic
  prior elsewhere. Unknown pixels are not counted as negative evidence.
- Fit the same subset-alpha objective with the same 240/max-view update budget,
  teachers, tracks, geometry, source views, encoders and evaluation threshold.

This is an initializer for a nonconvex selected-set solver, not a new semantic
field, a new teacher or an asserted independently verified identity model.
First run: the same 13 regional queries as next80; remaining 54 outputs retained.
Second run: apply this one initialization rule to all 67 text queries, using the
54 existing next79 tracks plus 13 next80 regional tracks. Regional positive
admission remains restricted to the 13 region-seeded queries as in next80.
No per-query winner selection is performed.

## Initial 13-query intervention

Macro selected projection IoU rises 56.9226 to 58.1688%. Scene values:
figurines62.8787→67.1621, ramen44.9314→48.3236,
teatime62.1447→62.0486, waldo_kitchen57.7357→55.1407.
This is a tradeoff, not uniform improvement.

Strong changes: waldo30.53→83.31%, Pikachu77.51→92.15%,
Tesla handle21.84→47.96%, kamaboko13.50→46.31%.
Failures: pour-over vessel52.01→3.70%, Ottolenghi83.69→67.80%,
plastic ladle still0%; onion segments only0→1.31%.

The post-evaluation visualization `output/next81/selected_comparison.jpg`
shows GT (left), next80 (middle), source initialization (right): green true
positive, red false positive, blue false negative. It was produced only after
prediction construction and is never used by training or inference. Waldo's
improvement includes substantial false-positive removal, so the mechanism is
not merely turning on previously empty elements. The wrong pour-over seed
becomes a wrong reconstructed set more decisively: lower source fitting loss
cannot certify the query's referent.

## Reproduction and boundaries

`run_source_scope_next81.py --scene SCENE --gpu GPU` runs the 13-query control;
`--all-queries` runs the unified rule. `summarize_source_scope_next81.py` records
all changed-query outcomes, initial/final selected counts and equality of retained
semantic scores. Machine reports live in `output/next81/`.

No new unit tests, threshold search or target-GT supervision. The score is the
frozen selected-Gaussian 2D projection metric, not independent 3D ground truth.
2D semantic scores are frozen; NVOS/ScanNet/UQIS remain unchanged. A single region
seed still cannot express exhaustive text-instance union. Independent referent
validation remains unresolved, as do compact distillation and same-map four-query
publication evidence.

## Completed all-query control and decision

All eight scene evaluations finished. The all-query policy attempted all67
queries;66had active supervision. All semantic-score tensors remain exactly
identical. The13-query intervention also retains the other54selections exactly.

| Scene | next80 | Initialize13regional queries | Initialize every active query |
|---|---:|---:|---:|
| figurines | 62.8787 | 67.1621 | 66.7718 |
| ramen | 44.9314 | 48.3236 | 45.4648 |
| teatime | 62.1447 | 62.0486 | 62.7399 |
| waldo_kitchen | 57.7357 | 55.1407 | 53.1891 |
| Macro | 56.9226 | 58.1688 | 57.0414 |

**No mainline promotion. Retain next80.** The all-query rule adds only0.1188pp
macro and worsens waldo_kitchen by4.5466pp. The13-query rule improves macro1.2462pp
but worsens two scenes and collapses the wrong pour-over hypothesis. Neither is
a robust replacement. Do not stitch successful queries or scene policies using
these outcomes. The source initializer remains an explicit experimental option;
its default is the unchanged semantic initialization.

The all-query control improves pirate hat42.25→75.89, yellow desk56.49→72.92,
and three cookies45.10→61.49, but regresses rubber duck with buoy48.05→15.06,
wavy noodles62.35→40.35 and pot93.16→68.47. Even existing detector-seeded tracks
are not uniformly compatible with the requested whole/part extent.

This establishes a useful separation:

1. Some source hypotheses already contain useful information which the old
   membership start suppresses or entangles with extraneous elements. Source
   initialization demonstrates substantial recoverable headroom on those queries.
2. Some hypotheses point to the wrong object or incompatible scope. Fitting them
   better cannot repair semantic identity; it can replace a partly correct map
   response with a more faithfully reconstructed wrong object.
3. A global initialization switch cannot distinguish these cases. The next method
   must explicitly maintain semantic referent evidence separately from extent
   evidence, including alternative whole/part hypotheses and evidence that can
   contradict the initial source-region choice. Tracker/geometric agreement alone
   is not independent semantic verification.

Prioritize this evidence distinction over more initialization blends, longer
training or post-hoc per-query score selection. Source-stage visual review is
useful for diagnosing the distinction but must not silently become an extra
manual-label advantage in benchmark comparisons. The newly recorded initial and
final element counts permit future scope failures to be distinguished from
missing activation without treating low fitting loss as success.
