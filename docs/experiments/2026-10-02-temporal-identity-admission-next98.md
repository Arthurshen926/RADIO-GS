# Temporal identity drift and source-component admission (next98)

2026-10-02. Complete12scene evaluations across three source-admission recipes.
The category-compatibility recipe reaches62.430% selected projection IoU versus
next97 61.820%; retain as an optional higher-mean LERF candidate. Defaults and
stable reference are unchanged. Teacher identity reliability remains unresolved.

## Concrete failure found before the intervention

The sake-cup temporal track repeatedly moves from the cup to nearby white paper,
and sometimes labels cup and paper simultaneously. A deterministic20-view source
contact sheet (even spacing among nonempty present nonseed views) makes this
visible: source16,22,45,67 include the wrong nearby material. Source1,7,29,40 show
the cup. This is a source-image observation, not a target-GT diagnosis or a complete
manual annotation of every frame. See output/next98/sake_track.jpg and its JSON.

The source-only detector/track union check finds NO detector instances for sake
cup. Therefore old detector union cannot explain this cup contamination. Of these
13queries, only pikachu(1) and spatula(2) have detector instances, none with positive
native foreground margin. That separate admission concern remains; it is not the
cup explanation. Results are in output/next98/source_union.json.

Next97's stronger positive loss reconstructs some wrong temporal positives more
faithfully. Scope supervision must distinguish correct tracking from merely a
tracker-present flag. Geometry changes and extra optimization cannot turn paper
labelled as cup into correct target supervision.

## Three bounded source-only controls

All start from the SAME next95 high temporal caches and query hypotheses, next97
balanced loss, footprint prior0.1, frozen geometry/native scores, high/interleaved
raster, source initialization, source frames and update budget. Refit the same13
columns, retain other54 from next97. No target RGB/labels used by admission.

Use8-connected components of each present source mask. Process every component
containing at least one positive-core pixel under the existing3x3erosion; no source
frame cap or area threshold. A component too thin to have a positive core already
cannot supply positive supervision. The final recipes judge1477components across
13queries; all14original seed hypotheses remain unchanged, including the admitted
spatula secondary. Reference crops come from their selected seed masks.

1. **Direct:** local Qwen2.5-VL-3B compares outlined source references with the
   outlined candidate component, without query text. A=same physical object,
   B=different/mixed, C=unknown. It admits1069/1477components. It incorrectly admits
   obvious paper for sake cup at source16/22/67, admitting103/108cup components.
   Thus this is not verified instance identity even where metrics improve.
2. **Description:** independently describe references and each candidate without
   query text or reference context for the candidate. Then compare descriptions
   for compatible category/material/appearance. Captions are generated in batches
   of8with greedy64-token decoding; text comparison is A/B/C. It admits350/1477.
   It rejects paper but also rejects real cups because one caption says metal,
   another says ceramic. Other captions hallucinate candy/meat/bread for fishcake.
3. **Category:** reuse EXACT same captions; text comparison ignores guessed material
   and appearance differences, allowing compatible broad categories/visible parts.
   Admit A; both contradictory B and insufficient C become unknown supervision.
   It admits689, rejects133 and abstains655. This is a gross-category drift filter,
   NOT proof of physical identity, exact attributes or whole/part completeness.

The latter two controls were motivated by SOURCE observations of earlier verifier
failures in this turn. This is sequential development exploration, not preregistered
or independent validation. The category prompt contains the source-derived
cup/material and paper/cup examples, applies uniformly to every query, and uses no
target GT. Prompts, model revision, raw decisions and captions are saved per query.

## How admission enters the pipeline

Keep the original probability maps. Positive permission is the union of accepted
component masks. Rejected/uncertain components are NOT zeroed into new background
labels. The original negative construction and unknown boundary band remain.
A frame without any accepted component has zero scope reliability and supplies
no active fit view; a query with no active views retains next97's selection.

There are material consequences beyond deleting positive pixels: active views
also determine initialization and footprint-cost camera weights, and fixed update
counts revisit the remaining eligible views more often. Hence these are end-to-end
admission controls, not an isolated label-cleaning-only effect. Because positive
permission is limited to kept temporal components, detector-only positives in
processed frames are also excluded; only2of13queries have such detections.

Source geometry, native semantic encoder/field and evaluator are not changed.
No post-evaluation stitching or per-query winner selection. The mask permissions
use the existing decoder path; none of the verifier's answers is treated as GT.

## Results

Selected Gaussian silhouette projection IoU (%):

| Scene | next97 | Direct | Strict descriptions | Category compatibility |
|---|---:|---:|---:|---:|
| figurines | 67.159 | 67.731 | 67.817 | 67.415 |
| ramen | 54.740 | 55.315 | 55.894 | 56.308 |
| teatime | 63.809 | 63.818 | 63.661 | 63.693 |
| waldo_kitchen | 61.572 | 61.660 | 59.152 | 62.302 |
| Macro | 61.820 | 62.131 | 61.631 | **62.430** |

Category gains+0.610points versus next97. Three scenes improve; teatime regresses.
It improves the mean versus direct, not every query. Strict descriptions are not
promoted. Keep all comparisons, including the material and category failures.

| Query | next97 | Category |
|---|---:|---:|
| pikachu | 91.69 | 94.80 |
| tesla door handle | 71.49 | 68.92 |
| waldo | 67.88 | 72.30 |
| kamaboko | 81.33 | 86.10 |
| onion segments | 0.00 | 0.00 |
| sake cup | 56.94 | 69.92 |
| dall-e brand | 50.43 | 49.96 |
| paper napkin | 81.13 | 79.95 |
| ketchup | 73.25 | 78.60 |
| ottolenghi | 69.28 | 71.21 |
| plastic ladle | 0.00 | 0.00 |
| pour-over vessel | 89.51 | 89.08 |
| spatula | 40.17 | 49.36 |

Key posthoc precision/recall:
- Sake cup: precision67.1→80.8%, recall69.4→71.9%, IoU56.94→69.92.
- Spatula: precision43.0→53.8%, recall85.8→85.6%, IoU40.17→49.36.
- Waldo: precision97.0→97.2%, recall69.4→73.9%, IoU67.88→72.30.
- Pour-over: precision97.3→98.6%, recall91.7→90.2%, IoU89.51→89.08.

Strict description admission leaves only2/60pour-over components and changes the
available source constraints; its IoU collapses to36.32 with precision38.1%, recall
88.8%. This is NOT simply a smaller prediction or successful rejection of background.
Over-pruning source evidence can leave the selected set underconstrained and
increase false extent. Category admission retains36/60components and recovers89.08.

Source examples clarify what the category rule does: cup frames7/29, incorrectly
rejected for material mismatch, are accepted; paper frames16/22/45/67 are rejected.
Frame62's real cup is described as a knob and still rejected. These are qualitative
source checks, not a full verifier accuracy estimate. Wrong seeds, same-category
distractors, teacher hallucinations and incomplete scope remain unsolved.

All12outputs preserve semantic scores and the other54selection columns bit-exactly.
Semantic IoU66.930/AP84.146 remain fixed. No new NVOS/ScanNet/UQIS, unified four-query,
compact-student, independent3DGT, independent-test or SOTA result. Additional source
VLM calls have a cost; no inference-speed benefit is claimed. This remains the
optional source-assisted transient extent path, not a replacement for native query
matching or evidence of cross-scene generalization.

## Decision and next focused work

Keep category admission as the optional higher-mean candidate, with next97 rollback
and next80 stable reference. Do not promote strict descriptions. Direct is retained
as a diagnostic alternative, not selected per query. No defaults changed.

The core unresolved issue is now concrete: temporal masks may change referent,
and neither repeated geometric support nor an uncalibrated VLM answer certifies
identity. Category checks remove some obvious drift but cannot validate exact
instances, parts or reference seeds. Next strengthen source-side identity support
across independent views, with explicit unknown handling and protection against
removing all useful views. Do not relabel unknown as background or pursue another
unbounded loss-weight sweep. Onion/ladle remain0; Waldo still below the earlier
next93 78.22. Those failures are not repaired by this result.

## Reproduction and artifacts

- inspect_scope_union_next98.py; view_sake_scope_next98.py.
- verify_track_identity_next98.py --scene SCENE --identity-mode direct|description.
- check_track_category_next98.py --scene SCENE (reuses completed blind captions).
- run_identity_scope_next98.py --scene SCENE --gpu GPU
  --identity-mode direct|description|category.
- summarize_identity_next98.py; summarize_identity_admission_next98.py;
  measure_identity_errors_next98.py; measure_identity_sets_next98.py.
- output/next98/{summary,identity_admission,source_union,error_decomposition,
  selected_sets}.json and sake_track.jpg.
- output/identity_tracks_next98; output/batched_description_identity_tracks_next98;
  output/category_identity_tracks_next98 contain final evidence and judgments.
- output/{identity_scope_next98,description_identity_scope_next98,
  category_identity_scope_next98}/high_footprint/<key>/ contain final evaluations.

Operational notes: tiny crops initially violated the processor's28-pixel minimum;
neutral padding now satisfies that input contract. Initial serial description runs
were stopped for batch throughput; their partial outputs under
output/description_identity_tracks_next98 are NOT consumed by final recipes.
