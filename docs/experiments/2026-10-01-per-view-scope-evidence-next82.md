# Per-view scope evidence versus temporal propagation (next82)

The previous experiments established both recoverable initialization headroom and
harmful reconstruction of wrong source hypotheses. This iteration asks a narrower
question: does the initial geometric check actually cover the subsequent track,
and can per-view scope disagreement identify unreliable supervision?

## Source-stage findings

The equivalence connected component of each next80 seed spans only2–5source
views. Mean tracker/region IoU in these components is0.86–0.995. This is local
corroboration, not validation of a100+frame track. Components use only frozen
mutual-equivalence edges; containment is not interpreted as identity. They were
saved for diagnosis, not used as sparse substitutes for full-source supervision.

Compare each present tracked mask against all separately generated automatic
regions in its source frame, selecting the highest mask IoU. This requires no
query reranking, reference annotation or target image. Full-source agreement is
much lower for some queries: waldo0.363, onion segments0.342, DALL-E label0.370,
ketchup0.238. Paper napkin0.957 and sake cup0.906 remain high. Low agreement does
not distinguish a drifting tracker from missing/incomplete automatic proposals;
high agreement does not prove semantic identity (the sake candidate is a bottle).

## Method control

Keep seed identity, all available source frames, tracking probabilities, Gaussian
geometry, native semantic field, query encoders, optimizer budget and evaluator.
Change only the source evidence consumed by scope fitting:

- Positive pixels must also lie inside an eroded, matching automatic region.
- Negative pixels must also lie outside its dilated region.
- Disagreement is unknown. No missing region is asserted to be background.
- Multiply view confidence by mask IoU. Zero intersection supplies no fit weight.
- Apply the same restriction to subsequent detector-corroborated positives.

The automatic region is chosen by scope overlap, not claimed as a new semantic
identity teacher. Both automatic regions and the tracker use SAM3; independently
computed observations do not imply independent model errors. A matching region
may have wrong whole/part granularity, which this experiment cannot assume away.

Two complete four-scene controls use semantic initialization (matched next80)
and source initialization (matched13-query next81). Only the same13previously
seedless queries are refit; the other54selections are retained. No per-query or
per-scene winner assembly is used. The0.5overlap count in diagnostics is descriptive,
not an admission threshold or a tuned benchmark decision.

## Artifacts

- `build_correspondence_scope_next82.py`: seed-component coverage and agreement.
- `verify_track_regions_next82.py`: full-source regions, uncertainty masks,
  continuous overlap confidence and per-frame diagnostics.
- `run_verified_scope_grid_next82.py SCENE GPU`: both matched reconstruction arms.
- `summarize_verified_scope_next82.py`: full scene and changed-query comparisons.

No target RGB or GT enters the method. The evaluator still reports selected-set
2D projection IoU, not independent3DGT. Semantic response scores are frozen.
NVOS/ScanNet/UQIS and compact-student/four-query deployment are unchanged.

## Completed results and decision

All8scene evaluations complete. Selected-Gaussian projection IoU(%):

| Scene | next80 | Verified / semantic init | next81 regional source init | Verified / source init |
|---|---:|---:|---:|---:|
| figurines | 62.8787 | 62.7580 | 67.1621 | 67.0736 |
| ramen | 44.9314 | 44.9253 | 48.3236 | 48.4486 |
| teatime | 62.1447 | 62.1581 | 62.0486 | 62.0312 |
| waldo_kitchen | 57.7357 | 57.7164 | 55.1407 | 55.5826 |
| Macro | 56.9226 | 56.8895 | 58.1688 | 58.2840 |

**No promotion; retain next80.** The semantic-init arm loses0.0331pp. The
source-init arm improves its matched parent by0.1152pp, but remains a tradeoff:
it does not repair the large wrong-referent regressions and is not an accepted
replacement of the retained four-scene recipe. No per-query result stitching.

The source-init arm recovers pour-over vessel3.70→11.69%, still far below
next80's52.01%; onion segments1.31→2.90% is still poor; plastic ladle stays0%.
Waldo regresses83.31→81.43 despite its low automatic-region agreement. This
rejects the assumption that low overlap necessarily means an erroneous track.
Fine labels and small objects may simply lack an adequate automatic region.
Sake cup's high source-mask agreement also does not correct its wrong bottle
hypothesis. Independent mask generation is not independent semantic evidence.

The source evidence narrows the next priority: validate the *referent* and
requested whole/part scope before allowing a region to initialize an element
set. Repeating geometric correspondence, mask agreement or confidence-weighting
variants cannot supply that missing identity information. Future work should
compare competing source-region identity hypotheses using content evidence,
with generic category/attribute distinctions and source-only validation, while
preserving verbatim runtime-query semantics and multimodal encoder compatibility.
Do not add benchmark-specific aliases or manually discard failed queries.

Existing local CLIP foreground/context controls from next58 were also checked:
they were near-neutral and do not establish a ready-made independent verifier.
This round does not relabel a repeated CLIP confidence swap as a new solution.
No large improvement, SOTA status, independent3DGT result or new other-benchmark
score is claimed. All diagnostic components and optional source evidence remain
experimental; unverified default track artifacts are untouched.
