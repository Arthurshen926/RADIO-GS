# Region granularity before semantic writing, next76

## Method question

Next75 improved source projection, but every pixel still received the normalized
mean of all covering SAM-region descriptors. This confounds native descriptors
from differently sized supports before they enter the shared Gaussian map.
This iteration changes that observation representation, keeping the frozen
next66 carrier, all legal source views,192x256 native writing,96x128 source
calibration, source scope refinement, confidence transport and evaluation fixed.
No new teacher, query alias, target RGB or annotation supervision is introduced.

Two predeclared controls use figurines and ramen as the first pilots:

- `leaf`: for each source pixel, write the native descriptor of its smallest
  covering region instead of averaging all covering regions. One1536D field.
- `hierarchy`: reconstruct three native1536D slots on the SAME Gaussians:
  original mixture, smallest covering region, and largest covering region.
  Normalize each slot independently and take the maximum query cosine BEFORE
  the common source-trained scalar calibration. No GT-selected scale and no
  per-query variant switching. This has3x native feature storage and greater
  reconstruction cost; it is not an equal-capacity compact-student comparison.

Area ordering is only a granularity proxy. These slots do not establish
containment, cross-view identity or genuine whole/part relations. In particular,
`parent` in the prototype names the largest covering observation, not a proven
parent instance. Camera changes can alter slot membership. The source cache's
intermediate regions remain represented by the mixture slot, not independently
preserved region instances. These limits matter when interpreting a failure.

The first evaluations freeze next75 selected membership exactly. Any subsequent
joint-extent run uses the unchanged240-update source decoder, with all other
controls fixed. No scene/query-wise winner assembly is allowed.

## Source-only mechanism observation

`measure_region_overlap_next76.py` measures the old authorized source cohort,
not evaluation images. Among source pixels covered by any cached teacher region:

| Scene | Source views | Multiple-region coverage | Coarse/fine area ratio | Native cosine, fine vs coarse |
|---|---:|---:|---:|---:|
| figurines | 20 | 87.766% | 3.590 | 0.860 |
| ramen | 25 | 89.846% | 4.805 | 0.867 |

Overlapping supports are common, but these statistics do not prove incorrect
labels or harmful mixing. Differently sized regions can supply useful context.
The actual retrieval interventions are needed to establish utility.

Source statistics: `output/next76/source_overlap.json`.
Evaluation summary: `output/next76/summary.json`.
Implementation: `rebuild_region_slots_next76.py`, `run_region_slots_next76.py`,
`summarize_next76.py` under `scripts/research`.

## Scope

All scores are development results. Selected-Gaussian projection is not an
independent3D-GT metric. No new NVOS/ScanNet/UQIS evaluation or compact
multi-teacher training is claimed. Native encoders remain unchanged, but same-map
four-query deployment and generic unseen-query calibration remain outstanding.

## Completed results and decision

| Scene | Recipe | 2D IoU | AP | Selected projection IoU |
|---|---|---:|---:|---:|
| figurines | baseline | 68.517 | 87.626 | 59.432 |
| figurines | leaf | 68.020 | 85.660 | 59.432 |
| figurines | hierarchy | 68.196 | 85.938 | 59.432 |
| figurines | hierarchy_joint | 68.196 | 85.938 | 58.175 |
| ramen | baseline | 62.609 | 78.300 | 44.018 |
| ramen | leaf | 61.594 | 77.628 | 44.018 |
| ramen | hierarchy | 62.291 | 79.324 | 44.018 |
| ramen | hierarchy_joint | 62.291 | 79.324 | 44.671 |

All six planned pilot evaluations completed. Both alternatives fail replacement;
no extension to the remaining two scenes, no per-query assembly, and no default
change. Retain the complete next75 LERF candidate66.930IoU/84.146AP/54.292selected
projection. NVOS86.489IoU/98.014AP remains the separate next74 research candidate;
it was not rerun here.

The leaf control loses IoU/AP in both scenes. Its figurines query changes include
pumpkin−32.17pp and pirate hat−6.24pp, despite green toy chair+9.95pp and miffy
+4.39pp. These are post-evaluation diagnostics, not routing rules. Removing
context is not a general fix. The three-slot maximum improves ramen AP and its
redecoded selected projection, but both scene IoUs fall; figurines AP and selected
projection fall too. Extra native capacity alone does not solve this problem.

The experiment rejects these particular area-based policies. It does NOT show
that genuine region identity, containment or cross-view factorization is useless.
A largest covering region can contain several objects or merely overlap a small
region. Maximum cosine is insufficient evidence for transferring its label to
the element. The next substantive work remains preserving region observations
and their support relationships, then selecting compatible query evidence using
independent source views. Whole/part labels must not be inferred solely from area.
This requires an explicit correspondence/containment mechanism, not promoting
these three area slots as a solved hierarchy.

One prototype save-stage variable collision (`parent` path vs region index) was
fixed and the affected construction jobs rerun before obtaining the results.
No failed/partial jobs enter the table. No unit-test scaffolding or scorer changes
were introduced; validation consists of the completed real retrieval experiments.
