# Typed regional relations and native semantic propagation, next79

## Implemented change

Use next78's unmerged, all-source region bank to distinguish equivalent-support
hypotheses from directional containment. Restore next75's native feature-writing
and readout order for this comparison; none of next78's failed late-pooling
policies is used. Same frozen Gaussian geometry, native encoder, source cohort,
192x256 writing,96x128 source calibration, refinement and evaluation.

For each source A and its camera-selected neighbors B/C from next77:

1. Compute A/B regional overlap on their common Gaussian visibility. Select
   maximum-Dice correspondence in BOTH directions.
2. Admit an equivalence edge only for mutual matches whose two directional
   coverages are each at least0.7. Require third-view support of at least0.7,
   measured by both pair members agreeing with the SAME region in C. C does
   not select the initial A/B match, but does gate final edge admission.
3. Weight an admitted edge by pair Dice × third-view support × squared positive
   native descriptor cosine. Deduplicate undirected pairs; reciprocal camera
   choices are not independent extra votes.
4. Separately record directional containment when one coverage is at least0.8
   and the reverse below0.5. These are geometric containment hypotheses, not
   semantic part labels. They NEVER become equivalence edges or propagate
   descriptors in this experiment.
5. For each region with admitted neighbors, take one normalized native-vector
   update: original descriptor plus its weighted neighbor mean, then normalize.
   Isolated observations retain their normalized native descriptor. Original
   descriptors remain in the immutable source bank. No random adapter, query
   names, target pixels, aliases, or learned temperature are introduced.

All thresholds and the single update are fixed before retrieval evaluation;
there is no threshold sweep or per-query routing. The local third-view gate is
not an independent test-set evaluation, and common visibility cannot prove that
whole physical-object scopes are identical when parts are occluded. The native
feature coordinates remain fixed, but propagated vectors do change their query
cosines; this is not a mathematical guarantee of preserved native accuracy.

## Source relation coverage

| Scene | Region observations | Unique equivalence edges | Regions with neighbors | Directed containment records |
|---|---:|---:|---:|---:|
| figurines | 24,486 | 5,168 | 7,906 | 21,382 |
| ramen | 8,719 | 2,252 | 3,412 | 13,322 |

These are source construction counts, not correspondence precision/recall. No
instance GT was used to certify identity. Containment records may repeat across
camera-pair traversal; unlike equivalence, they are not used as weighted votes.

## Evaluation and artifacts

First freeze next75 selected memberships to isolate semantic changes. Then run
the unchanged selected-subset decoder when evaluating final extent. Retain
response and selected projection as separate metrics. No independent3D-GT claim.

- `scripts/research/build_region_relations_next79.py` builds typed edges and
  optional propagated descriptors from the existing sparse bank.
- `scripts/research/rebuild_expanded_semantic_next72.py` accepts optional regional
  descriptors; its default path is unchanged.
- `run_equivalent_region_next79.py`, `decode_equivalent_region_next79.py`, and
  `summarize_next79.py` reproduce the controlled retrieval pipeline.
- `output/region_relations_next79/{scene}/relations.pt`, `descriptors.pt` and
  `report.json` retain source hypotheses and provenance.
- `output/next79/summary.json` records completed development evaluations.

Containment-aware query-scope inference is still OPEN: recording an edge does
not implement a query-dependent choice between whole and part. No compact
multi-teacher rebuild, four-query deployment, NVOS/ScanNet/UQIS or SOTA result
is established by this text-retrieval experiment.

## Completed semantic pilots

| Scene | Baseline IoU/AP | Equivalence propagation IoU/AP |
|---|---:|---:|
| figurines | 68.517/87.626 | 68.521/87.619 |
| ramen | 62.609/78.300 | 62.441/78.202 |

Both selected metrics are fixed by construction in these two controls. The
semantic results do not support promotion, so no selected decoder or remaining
scene expansion is performed for this descriptor-propagation arm. Typed edge
construction is implemented; effectiveness and instance correctness are not
established. Retain next75. The independent full-source temporal controls in
`2026-10-01-full-source-text-tracking-next79.md` are not combined with this arm.
