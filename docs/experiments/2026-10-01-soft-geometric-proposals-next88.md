# Soft geometric proposal association (next88)

## Mechanism

The same native source bank contained relevant-looking thin-object regions that
were absent from the strict equivalence-pair shortlist. Instead of treating the
three-view equivalence graph as an identity admission gate, use geometry to
propose candidate correspondences:

1. Rank ALL authorized source-region descriptors with the unchanged native query.
2. Take128regions as an explicit bounded proposal shortlist (not a source-frame cap).
3. For each region, find its highest support-Jaccard partner in another source
   frame within that shortlist. Require positive overlap, without a hard
   equivalence or third-view confirmation requirement.
4. Rank pairs by minimum native cosine, suppress support-Jaccard>0.8 near
   duplicates, and keep at most16pairs for the unchanged two-view Qwen classifier.
5. The same source-initialized membership solver consumes the selected seed's
   full-source SAM3 track. Geometry, field, encoders and scoring are frozen.

The provider still may choose a wrong identity. Positive overlap is a proposal,
not verified correspondence. This changes the candidate search graph and its
budget, not just one numerical threshold. Target cameras, RGB and GT never
participate in proposal selection; camera coverage from next86 was diagnostic.

## Complete matched result

Selected Gaussian 2D silhouette projection IoU (%):

| Scene | next85 | Soft proposal association |
|---|---:|---:|
| figurines | 67.099 | 67.225 |
| ramen | 49.594 | 49.552 |
| teatime | 62.033 | 62.034 |
| waldo_kitchen | 57.586 | 58.093 |
| Macro | 59.078 | 59.226 |

The +0.148percentage-point gain is modest, not a breakthrough. Kitchen improves
57.586→58.093, including pour-over58.82→65.72 and ketchup46.85→49.92. Handle
46.60→49.85; ramen slightly regresses. Onion remains0, ladle is only0.02, spatula
11.17 remains unchanged under single-scope selection. No per-query stitching.

Keep as an optional higher-mean LERF research candidate; stable next80 remains.
The result does not establish SOTA, independently annotated3D IoU, compact student
improvement or complete four-query retrieval. Semantic IoU/AP66.930/84.146 stay
unchanged; no new NVOS/ScanNet/UQIS result.

## Remaining issue and paired experiment

Source crops now include the red spatula among the candidate pool, but the
single winning scope still selects a wooden spatula. Next89 applies the same
bounded two-hypothesis union to this changed pool. The previously defined joint
comparison rule is also repeated on the six spatially distinct hypotheses,
without adapting its labels or admission criterion. It again admits none;
retaining the single-scope output under that rule is an exact fallback, not
an observed improvement or proof of correct rejection. The raw union comparison
will show whether recovered scope evidence improves actual selected elements
and whether false identities still outweigh its benefit.

## Artifacts

- output/next88/summary.json and source_choices.jpg
- output/soft_candidates_next88/lerf_*/contract.json
- output/grounded_seeds_next88/lerf_*/contract.json
- output/grounded_scope_next88/source/lerf_*/typed_readout.pt
- output/next89/lerf_*/joint_identity.json
