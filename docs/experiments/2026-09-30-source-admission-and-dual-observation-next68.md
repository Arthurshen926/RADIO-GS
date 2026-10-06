# Next68: source supervision admission and cold-start membership

## Outcome

Four complete four-scene controls separate two hypotheses. Adding a full-scene visible-ownership objective does **not** improve the current selected-set decoder. Other-view geometric corroboration can admit missing source positives, but a previously empty membership also needs an evidence-based initialization. The resulting candidate improves selected projection **51.414 → 51.835**; semantic2D IoU/AP/localization remain **64.882/81.021/87.328**, unchanged by construction. This is a modest gain, not a major performance breakthrough or complete scope solution.

All16 scene fits and16 evaluations finished. The experiments use the same expanded-RGB next66 geometry, original source labels/frames,240steps/query, frozen semantic response, and next67 exposed update domain. No test suite/audit framework was added. The development evaluation and rendering thresholds remain unchanged. No new other-benchmark or independent-test result is claimed.

## 1. Test whether fitting visible ownership helps isolate the object

The retained loss fits the alpha silhouette of the selected subset against partial source masks. A second observation of membership is its visible contribution under full-scene transmittance, computed with the existing normalized sparse contribution operator. We tested BCE+Dice on that visible membership alone and the sum of visible and subset objectives with unit weights. Both use the same known-pixel masks, confidence, membership initialization,240updates and temperature schedule.

| Recipe | Selected projection IoU (%) |
|---|---:|
| Next67 exposed, subset objective | **51.414** |
| Visible membership objective only | 33.505 |
| Visible + subset objectives | 49.665 |

Reject these replacements. A normalized full-scene membership response and an isolated subset silhouette are different consumed quantities. Under the current noisy/partial labels and carrier, optimizing the two together does not yield a better selected set. This experiment does not establish that every weighting, supervision source or surface-ownership model must fail. It provides no basis for retaining another loss merely because it sounds physically plausible.

## 2. Distinguish absent source detections from rejected source detections

A source-only inventory found14 of67queries with no eligible range-training view in next67. Of these:

- **11 have no nonempty detector candidate** on the original source cohort: figurines tesla door handle/waldo; ramen kamaboko/onion segments/sake cup; teatime dall-e brand/paper napkin; waldo_kitchen ketchup/ottolenghi/plastic ladle/pour-over vessel.
- **3 have candidates but no eligible view**: pikachu1candidate-bearing view, hand2, corn14. A candidate count is not a verified correct detection count. Native qualification, erosion and source visibility affect eligibility.

The absence of eligible source supervision is different from a trained query with wrong identity/scope: miffy and bear nose each have9eligible views yet remain poor. Current source evidence therefore has both coverage and correctness limitations. The count of14 alone does not quantify how much total benchmark error each limitation causes.

Inventory: `output/dual_observation_next68/source_admission.json`. It was generated from source detections and fit histories. Benchmark masks did not decide which queries receive the intervention.

## 3. Admit corroborated positives and initialize newly trainable queries

For each source view, backproject eroded detector foreground and known detector foreground/background evidence through the new geometry. Weight by detector confidence. To qualify a candidate foreground pixel in view v, subtract v's own contribution, project the other-view estimate into v, and require both majority projected coverage and positive fraction above.5. Preserve the existing qualified positives; add only detector-positive pixels meeting this other-view criterion. Existing negatives remain except newly admitted positives are removed from them. Missing detections do not contribute a false all-background observation because their detector confidence is zero.

This is geometric corroboration, not independently proven entity/whole-part identity. Repeated correlated detector errors can pass. It also cannot invent a positive for the11queries with no detector candidates.

Two controls:

1. **Admission:** keep the original new-field membership initialization.
2. **Admission + seed:** for queries with no previously eligible view that now gain one, initialize membership with the maximum of the original prior and the confidence-weighted source foreground backprojection divided by evidence mass. Other queries keep their original initialization. Then run the same240step subset decoder.

Only corn became newly eligible in these four scenes. This was the outcome of the uniform source rule, not a hardcoded query name or a GT-based switch. Source corroboration added94corn foreground pixels summed across source views. Pikachu and hand did not obtain qualifying evidence.

| Recipe | Four-scene selected IoU (%) | Ramen selected IoU (%) | Corn selected IoU (%) |
|---|---:|---:|---:|
| Next67 | 51.414 | 41.386 | 0.000 |
| Admission only | 51.476 | 41.645 | 0.000 |
| Admission + source seed | **51.835** | **43.079** | **20.362** |

The matched pair shows that admission alone was insufficient for this cold-start query and source-based initialization recovered some extent. It does not prove a particular rasterizer gradient/culling mechanism caused the failure; no such gradient attribution was measured.20.362 remains low and is not a complete corn reconstruction.

| Scene | Next67 | Admission + seed |
|---|---:|---:|
| figurines | 56.185 | 56.263 |
| ramen | 41.386 | 43.079 |
| teatime | 55.283 | 55.227 |
| waldo_kitchen | 52.800 | 52.770 |

The extra positives have mixed effects on previously trained queries. Do not claim all scenes improved, silently select recipes by query, or count unchanged semantic metrics as a new semantic gain. Keep this as an optional source-admission candidate; no global default was changed.

## Research implication

The last two iterations now distinguish an actual update-domain error (next67, substantial gain), an unsupported additional objective (next68, rejected), missing source supervision, and initialization for newly supported queries. These are not interchangeable explanations for low IoU.

The larger unresolved opportunity is **new, correctly grounded source evidence**. The geometry now uses122–293legal RGB source frames per scene, while the semantic/range teachers still use20–25original frames. The next evidence expansion should use a query-independent source-view selection rule and a fixed recipe for every query, report its extra input/compute budget, and keep target views excluded. More frames are only a hypothesis: named entities and part references may still fail the text detector, so native teacher recognition over query-independent region proposals also needs comparison. These experiments have not been performed here.

For a paper, the desirable contribution is an evidence-grounded native-semantic field with explicitly reconstructed query membership, including how new source evidence can override an incomplete native prior without erasing its open-vocabulary capability. This round alone does not establish that contribution or CVPR/SOTA readiness. Four-query transfer, other datasets, cost comparisons, and independent held-out evaluation remain necessary. UQIS remains outside the current accessible experiments.

## Artifacts and reproduction

- `scripts/research/run_dual_observation_next68.py`: visible/joint controls.
- `scripts/research/run_admission_next68.py --gpus 2,3,4,5`: admission/seed controls.
- `scripts/research/summarize_dual_observation_next68.py`: complete four-scene aggregates.
- `output/dual_observation_next68/summary.json`: per-scene and macro metrics.
- `output/dual_observation_next68/seeded/lerf_<scene>/typed_readout.pt`: candidate, explicitly bound to the same next66 carrier for semantic and selected renders.
- Per-scene `fit.json`: admitted pixel counts, newly eligible queries, source optimization results.

The selected metric is a2D annotation comparison after selecting3D Gaussians and rendering them alone. It is not independently measured3D-GT IoU. The broader next66 input-budget and native-feature/query-time-supervision qualifications continue to apply.
