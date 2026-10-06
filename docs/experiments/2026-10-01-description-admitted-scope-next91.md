# Query-blind source descriptions for secondary identity (next91)

## Mechanism and exploration boundary

Use next89's six secondary hypotheses and the same two original source views per
scope. Ask the frozen Qwen2.5-VL-3B to describe each marked object WITHOUT the
query. Then supply only that description and the verbatim query to the same
model, asking A identity matches / B different or associated / C unknown.
Generated material/attribute statements are uncertain model outputs, not labels.
The task explicitly separates identity from exact mask-boundary certification.
All prompts, descriptions and class logits are saved.

The initial rule required all four primary/secondary descriptions to match and
admitted nothing. After inspecting SOURCE judgments, a second rule admits an
additional scope when its OWN two descriptions match, retaining the parent
without requiring new certification. This is a sequential development experiment,
not preregistered or independent held-out validation. No target RGB/annotations
or per-query manual exception is used by either rule. The parent is unchanged
when expansion is unsupported; this does not certify parent correctness.

Only red spatula passes the second rule. Other secondaries are described as
bottles, funnels, ambiguous food or unrelated items. Some captions are visibly
wrong (e.g. ladle called a key); the method is not a general correctness verifier.
The strict all-four rule remains an all-fallback control, with no redundant fit.

## Completed result

Run the unchanged scope solver on admitted merged evidence; otherwise retain the
parent exactly. All four scenes were evaluated.

| Scene | next88 | Description-admitted expansion |
|---|---:|---:|
| figurines | 67.225 | 67.225 |
| ramen | 49.552 | 49.552 |
| teatime | 62.034 | 62.034 |
| waldo_kitchen | 58.093 | 58.890 |
| Macro | 59.226 | 59.425 |

Selected Gaussian 2D silhouette projection IoU improves59.226→59.425.
Spatula11.17→28.70 accounts for the gain; other queries are retained. Unlike
unconditional next89 expansion, this rule avoids its wrong-cup/bottle/cookware
regressions. This is useful but narrow development evidence, not SOTA or proof
of general identity calibration. Semantic66.930/AP84.146remain unchanged.

## Further source-side bottleneck

At the consumed96x128scope raster, the red-spatula secondary track is nonempty
in11nonseed source views. Fixed3x3positive erosion leaves only6views and8.8%of
its positive pixels. The primary wooden-spatula track retains31.1%; ketchup32.9%,
ladle18.0%. Missing tracker views and removed positive pixels are different losses.
These counts precede geometric-validity and other decoder masks; they are not
final retained training-pixel counts or GT quality measurements.

Next92 tests eroded interior plus distance-transform local maxima (medial positive
evidence), with source identities, negative/unknown masks and teacher probabilities
unchanged. This does not recover lost tracker visibility or subpixel detail.

Artifacts: output/next91/summary.json, lerf_*/joint_identity.json,
source_raster_diagnostic.json; output/description_scope_next91/lerf_*/admission.json.
No NVOS/ScanNet/UQIS, compact-student/four-query or independently annotated3D gain.
