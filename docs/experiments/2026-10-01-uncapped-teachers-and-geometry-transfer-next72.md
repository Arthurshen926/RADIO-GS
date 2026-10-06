# Uncapped source teachers and matched geometry transfer (next72)

Status: active research execution, not a promoted recipe. This supersedes the old
source32/cache-intersection constraint for the new source expansion experiments.
The requested four-benchmark SOTA goal is not completed.

## Decisions and completed changes

- Legal registered RGB/cameras and frozen source/target splits determine teacher
  eligibility. Existing RADIO, MoGe prediction or SAM cache availability does not
  remove otherwise legal camera views. Frozen next66 geometry remains the LERF
  carrier. Counts: figurines293, ramen122, teatime169, waldo_kitchen180; the two
  reserved source-validation views in each scene remain excluded.
- Each scene has original, coverage64 and all cohorts. Coverage64 includes the
  original cohort, then fills camera-position/orientation gaps with a mild RGB
  sharpness preference. This is **pose diversity**, not an occlusion-aware surface
  coverage score. All has no maximum-image setting.
- Query-free multiscale SAM evidence is stored at original RGB resolution per
  frame. SigLIP native1536 masked-region descriptors are also frame-sharded.
  Source mask resolution and fit resolution are distinct. The first semantic
  comparisons explicitly retain96x128 fitting to isolate source input expansion;
  no high-resolution fit improvement is claimed.
- Detector entrypoint now accepts an explicit legal-source authority and a runtime
  query JSON list; original-resolution probability storage is optional. Identity
  qualification accepts the same authority and explicit query encodings. This
  removes the fixed teacher-frame loop from those entrypoints, but does not yet
  make the historical confidence prior reproducible for arbitrary new queries.
- Extent decoder reprojects the old semantic reference on new source cameras.
  Old reference invisibility is unknown, not a certified negative. Current carrier
  visibility defines detector support. Leave-one-view-out sums are accumulated
  rather than storing two dense element/query matrices for every view. Steps are
  at least the number of source views, so every active source view is consumed.
- Compact student retraining, four-query UQIS validation and default promotion
  remain deferred. Image/prompt routes are not replaced by a text-only task.

## ScanNet: completed eight-scene point-domain diagnostic

Same frozen next44 soft-identity predictions, same19/15/10 class splits. Transfer
from nearest Gaussian centre to public label-PLY vertices, equal point weights,
absent GT classes excluded from per-scene class mean, scene-macro aggregation.
Unobserved nearest elements predict void and their points remain in scoring.
No GT-to-Gaussian pseudo-label assignment or Gaussian-volume weighting.

| Frozen recipe | 19-class | 15-class | 10-class |
|---|---:|---:|---:|
| next38 parent, point domain | 50.123 | 51.322 | 64.432 |
| next44 soft identity, point domain | 53.938 | 54.503 | 67.680 |
| next44 soft identity, previous Gaussian domain | 59.247 | 59.485 | 73.219 |

The final row is a different metric domain, not a competing model. The point
adapter has not been certified equivalent to every named paper's full protocol;
these numbers are not advertised as an official SOTA comparison. These carriers
are provided-mesh based: point-to-centre distances at p50/p90/p99 are zero for
all eight scenes. Observed-nearest fraction is93.60–98.59%, with uncovered points
counted. The next44 improvement survives point scoring, but the larger old
Gaussian numbers must not be compared to point-domain papers.

Artifacts: `output/scannet_points_next72/{scene}/report.json`,
`output/scannet_points_next72/summary.json`, and parent counterpart.

## NVOS: completed initial geometry/video separation

Reconstruct source RGB+calibrated MoGe,3000updates, first300color-only, then joint
Gaussian fitting and densification; original materialized depth grid320retained.
No target RGB, mask or scribble enters geometry reconstruction. Recompute every
source projection and observed support on each geometry. Use identical frozen
next43 reference-seeded video evidence for old/new arms. Fit a full-scene visible
response with unknown query evidence abstaining to zero at output; no isolated
selected-alpha objective.120source-mask fitting iterations in both arms.

| Scene | Old geometry IoU/AP | New geometry IoU/AP |
|---|---:|---:|
| orchids | 66.266 /93.522 | 65.911 /97.816 |
| leaves | 74.238 /97.007 | 91.805 /99.684 |
| flower | 78.259 /96.664 | 84.473 /99.152 |

These are matched **mask-transfer mechanism controls**, without the retained
semantic prior or structural completion module, not full-mainline eight-scene
results. Geometry can materially improve transfer, but does not repair every
scene. Leaves has only4usable tracked source views, orchids17. The initial
prototype incorrectly left no-evidence elements at0.5, which the>=0.5evaluator
selects; this was corrected to output abstention and both old/new controls were
rerun. The discarded prototype figures are not final evidence or an old-mainline
bug claim.

Reserved-source PSNR after reconstruction: orchids19.47, leaves19.97dB. Before
values used gray initialization and are **not** a historical photometric baseline.
MoGe disagreement changes.0861→.0616 and.0670→.0863 respectively; depth is not GT.
RGB quality alone cannot certify better geometry. Full-eight results are reported below.

## In progress and not yet established

1. Two LERF coverage64 semantic/extent factorial pilots are complete; results below.
   Ramen all122semantic result is61.659IoU/77.485AP/43.722selected, versus
   coverage64 61.876/77.598/43.682. All-source/remaining-scene native-field jobs
   continue; the inconsistent expanded-extent rule is not extended blindly.
2. NVOS same-video old/new geometry controls are complete for all8queries.
   Integration into the retained native/structural map and external baseline comparison remain open.
3. Expanded source evidence still needs verified entity/whole-part completeness;
   same-query geometric agreement alone does not certify referent correctness.
4. Arbitrary-new-query confidence prior, external official protocol alignment,
   independent-view identity verification and multimodal regression execution
   are still open. They are not silently marked complete by teacher expansion.

Execution scripts: `run_source_coverage_next72.py`,
`run_expanded_semantic_next72.py`, `run_nvos_geometry_next72.py`,
`run_expanded_extent_next72.py` under `scripts/research/`. Logs and per-stage exit
codes live beside artifacts. No new test/audit framework or threshold sweep.

## Public comparison correction after reading primary sources

The [LESV paper, section4.3](https://arxiv.org/html/2604.01388v1#S4.SS3) specifies K64 Gaussian candidates with Euclidean-distance/opacity weighting and softmax class probabilities, not the single-neighbour diagnostic above. An additional frozen-prediction adapter implements that described transfer, yielding eight-scene19/15/10 57.665/58.214/72.437%. This is a transfer/protocol change, **not a learned-model gain**. Missing-teacher candidates have zero class probability; all-unknown neighbourhoods predict void. It remains an eight-scene result, not a certified paper comparison.

The [official LUDVIG repository](https://github.com/naver/ludvig#open-vocabulary-semantic-segmentation) lists10ScanNet scenes and an opacity filter. Current mainline8excludes scene0200_00 and scene0645_00. Both datasets are present locally; new-mainline reconstructions are not. They must be completed before a full-cohort public claim. The provided-mesh/point-cloud input track must also remain explicit.

## Completed interim results

All eight NVOS old/new geometry controls completed. Identical sparse video evidence:
meanIoU79.562→83.244, AP94.704→96.993. The retained full mainline remains83.484IoU;
therefore this mechanism control is **not promoted as its replacement**. Its gains
need integration with rebuilt native/structural features, especially orchids/trex.

Two-scene LERF coverage64 semantics with original extent evidence completed:

| Scene | Retained next68 IoU/AP/selected | Fresh old-cohort control | Coverage64 |
|---|---|---|---|
| figurines | 66.711 /84.722 /56.263 | 66.546 /84.571 /54.304 | 67.403 /84.880 /56.947 |
| ramen | 61.477 /76.117 /43.079 | 61.378 /76.315 /43.232 | 61.876 /77.598 /43.682 |

Fresh-old is not a bitwise replay: query-free SAM was regenerated, region eligibility
is calculated on native masks, and expanded decoder support uses current carrier
visibility with old-unknown exclusion abstention. Therefore both the matched new
control and retained mainline are reported. Selected mean improves~0.64pp versus
retained, below the desired~1pp gate. No promotion; all-view native teacher and
expanded extent pilots remain in progress. No student/four-query closure claim.

Existing full-source NVOS video (no new tracker training) crossed with both carriers:
orchids66.187→65.859, leaves77.338→92.552IoU. This reinforces geometry interaction
for leaves but does not repair orchids. The original sparse-video counterparts
remain separate; no per-scene winning recipe is stitched together.

An extraction resume issue was corrected:64/all cohorts now use distinct manifest
filenames over the same immutable per-frame cache. The old producer rejected the
new cohort at final manifest writing despite having generated all122ramen frames.
No frame budget was reduced to work around this; saved per-frame data were reused.

Post-evaluation visual diagnosis: `output/next72/nvos_geometry_controls.png` (GT is
used only for display). `output/next72/summary.json` lists completed controls.

Missing public ScanNet scenes have now been prepared without label access:
scene0200_00 uses all46provided training RGB/cameras and83,291mesh elements;
scene0645_00 uses all226views and352,477elements. These are geometry/source
preparations, not trained semantic maps or completed benchmark scores.

Execution at handoff: source extraction runner uses GPUs0–3 (ramen complete);
native reconstruction runner queues remaining cohort/scene jobs on GPUs4–5,
with one stage owner per GPU. Completed evaluation cohorts are reused. The
64extent pilot runner and all8NVOS geometry controls have finished. The recipe
is recorded in `configs/recipes/next72_research.json`; no default promotion.
