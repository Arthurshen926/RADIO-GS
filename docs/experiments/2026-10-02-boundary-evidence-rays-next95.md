# Boundary-preserving source evidence with interleaved rays (next95)

Date: 2026-10-02. Complete: 14 full-source tracker runs (13 primary, one admitted
secondary), paired low/high caches, eight scene fits and evaluations. No promotion;
retain next93 optional research candidate and next80 stable reference/default.

## Mechanism tested

The temporal model processes source images at width1008, but the original cache
interpolates logits directly to96x128 before sigmoid/storage. Training erodes one
pixel on that raster. This discards a disproportionate fraction of thin targets.
This is a lossy design choice, not evidence that the complete pipeline was broken.

Preserve192x256 teacher probability maps directly from the same model logits.
Also save96x128 from those exact logits in the same tracking pass. All14 low
tracks match their prior counterparts exactly for every frame/probability/presence.
No new identity candidates or source frames:293/122/169/180 authorized source views.
Keep the same admitted red-spatula secondary; no new union or identity decisions.

The decoder projects frozen geometry at192x256 and takes one interleaved pixel
phase per source view, cycling four phases by source-list index. Store96x128 rays
per view without pooling adjacent rays. The subset-alpha fit renders192x256 and
samples the identical pixels. Masks are eroded/dilated on the high raster BEFORE
sampling. Existing detector masks are resized from their source evidence; only
selected temporal tracks are regenerated at high resolution. Query columns not
being refit are retained from next93.

This changes evidence resolution, ray locations, and the physical width of the
one-pixel unknown boundary together. It is not an isolated test of teacher
resolution alone and not full dense192x256 supervision. It keeps the same number
of stored samples and optimizer steps, but high-resolution rasterization costs
more computation. Every authorized source view is still considered; no frame cap.
Teacher/network, geometry/opacity, native features, confidence, identities,
negative corroboration rule and evaluation remain fixed. No target RGB/GT enters
tracking, sampling or training. Sampling is independent of target labels.

## Source information actually recovered

Fraction of teacher-positive pixels remaining after erosion:

| Source scope | Low | High |
|---|---:|---:|
| waldo (primary) | 31.4% | 59.4% |
| tesla door handle (primary) | 51.3% | 73.1% |
| spatula (secondary) | 8.8% | 33.5% |
| plastic ladle (primary) | 18.0% | 41.7% |

These are source teacher-mask measurements, not true-object coverage or teacher
accuracy. The thin positive support loss is real; recovering it does not validate
the identity or completeness of the underlying mask.

## Frozen-protocol results

Selected Gaussian silhouette projection IoU (%):

| Scene | next93 | Fresh low | High/interleaved |
|---|---:|---:|---:|
| figurines | 67.065 | 67.065 | 67.199 |
| ramen | 54.234 | 54.234 | 53.588 |
| teatime | 62.098 | 62.098 | 62.231 |
| waldo_kitchen | 59.054 | 59.054 | 58.006 |
| Macro | 60.613 | 60.613 | 60.256 |

Fresh low reproduces all query IoU results exactly, but fitted soft membership
arrays are not bit-exact; do not confuse metric reproduction with weight equality.
All eight outputs preserve semantic scores and the other54 selection columns
bit-exactly. Semantic IoU66.930/AP84.146 are unchanged by construction.

High variant improves pour-over69.31→75.09, handle51.82→53.41, pikachu93.20→94.33,
napkin58.92→60.54 and waldo78.22→78.91. It regresses kamaboko72.80→67.40,
ketchup49.92→32.69 and spatula28.71→11.15. Onion remains0; ladle0.02→1.48
remains a failure. No scene/query-specific winner assembly.

Posthoc precision/recall makes the main distinction clear:
- Spatula: recall86.3→97.8%, precision30.1→11.2%.
- Waldo: recall82.8→84.8%, precision94.2→92.5%.
- Kamaboko: recall96.8% unchanged, precision74.7→69.0%.
- Pour-over: precision77.5→81.1%, recall86.8→91.1%.

Thus boundary evidence can improve some targets, but recovering target support
can also inflate the selected set. These results do not isolate whether residual
leakage is caused by incorrect source masks, mixed primitive footprints,
underconstrained membership, or optimization; the frozen-carrier trial cannot
establish an irreducible geometry bound.

## Decision and next method step

Do not promote the high/interleaved recipe. Retain next93 results. Keep the paired
teacher caches as reusable source evidence, avoiding repeated expensive tracking.
Stop expanding positive cores as a standalone remedy: next92 and next95 both show
that coverage gains can be accompanied by a much larger precision loss.

Next isolate selected-set expressiveness from supervision completeness:
1. With fixed paired source evidence, inspect whether the source constraints admit
   unnecessarily broad selected subsets. A compact-explanation prior must be
   explicit about its bias against unseen portions, not relabel unknown as GT
   background.
2. Compare source-only boundary-aware primitive responsibility/refinement to the
   frozen-carrier control if mixed footprints prevent fitting both sides. Any
   geometry refinement needs explicit feature/selection transfer and a matched
   carrier evaluation; no query-dependent or GT-dependent geometry edits.
3. Independently improve identity for onion/ladle. Their scores do not establish
   a boundary failure, and more raster detail has not supplied correct identity.

No new NVOS/ScanNet/UQIS result; no independent3DGT, unified four-query, generalizable
student-field, held-out or SOTA claim. This remains the source-assisted transient
scope path, compatible with but not validating the broader multimodal framework.

## Artifacts and reproduction

- track_boundary_next95.py --scene SCENE --gpu GPU (plus kitchen --secondary).
- run_boundary_scope_next95.py --scene SCENE --gpu GPU --arms low high.
- summarize_boundary_next95.py; measure_boundary_errors_next95.py;
  measure_boundary_evidence_next95.py.
- output/next95/{summary,source_evidence,error_decomposition,retained_outputs}.json.
- output/boundary_tracks_next95/{low,high}_{primary,secondary,merged}/.
- output/boundary_scope_next95/{low,high}/lerf_*/ contains fit and evaluation.

Decoder/tracker --raster-scale default remains1; paired low output is opt-in.
