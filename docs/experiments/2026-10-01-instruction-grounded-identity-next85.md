# Instruction-grounded source identity and scope (next85)

## Method change

The BLIP likelihood and erasure controls did not supply reliable identity. This
iteration uses a different task for an optional visual-language provider: judge
whether a marked candidate actually matches the requested referent and scope.

Provider: `Qwen/Qwen2.5-VL-3B-Instruct`, downloaded from the official repository,
revision recorded in each scene's contract. The [official model card](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
reports visual text understanding and localization capabilities. Those statements
motivate this experiment; they do not establish RADIO-GS recall performance.
The existing transformers4.49environment supports the model. Eager attention is
used with the existing PyTorch version; reconstruction dependencies are unchanged.

Freeze next83's16candidate pairs/query, generated from **all** authorized source
views and native geometric pairs. Show each candidate in a1.5xsource RGB crop,
with a red mask contour and unchanged surrounding context. Ask, using the verbatim
runtime query, for one of three judgments:

- A: exact referent with appropriate whole-object/part extent;
- B: wrong or merely associated referent, or wrong whole/part extent;
- C: insufficient visibility/resolution/knowledge.

Use the A-minus-max(B,C)next-token logit margin. Candidate confidence is the weaker
of its two source-view margins; choose the strongest positive pair. If none is
positive, retain the original native seed. This is a fixed categorical rule,
not a calibrated probability of truth or an evaluation-tuned threshold. The
model's unconstrained highest-scoring token is saved as a format diagnostic.
The full prompt, label IDs, checkpoint revision and pixel budget are persisted.

No target RGB, masks, manual per-query choice, query aliases, generated captions,
or source-trained benchmark vocabulary is introduced. The provider is optional;
native text/image feature coordinates and the base map remain unchanged.

## Source-stage observations, before recall evaluation

Source grid: `output/next85/source_choices.jpg`. The chosen candidates now include
the red ketchup packaging, cup, glass coffee vessel, DALL-E label, Waldo figure,
spatula and fish cake. The onion-segments candidate still outlines meat, not onion.
The sake-cup proposal also contains small disconnected paper fragments, illustrating
that accepting the referent does not guarantee a pure element membership target.
These observations did not edit predictions or introduce per-query overrides.

Across416candidate-view inputs, highest-token answers are A300, B18, C98; all are
valid response letters. All13queries have at least one admitted pair, so native
fallback is available but never triggered this round. A false acceptance is visible in the onion example, but the answer counts alone
cannot distinguish model permissiveness from a positive-enriched native shortlist;
they are **not** an accuracy estimate. Reviewing13selected crops
does not establish correctness of all416candidate judgments.

## Reconstruction comparison

After the source review, run the same full-source SAM3 tracking and frozen scope
solver for13queries. Exact same-frame/same-mask cached tracks may be reused, with
the new source metadata attached; remaining predictions are generated afresh.
The54other-query selections retain next80 values. Compare semantic initialization
(matched next80) and source initialization (matched next81regional control).
Both use the same source budget, geometry, native map, objective, step budget and
frozen evaluator. This adds inference compute and a pretrained provider; it is not
an equal-compute field-only comparison or a compact multi-teacher student result.

## Reproduction

- `verify_region_identity_next85.py --scene SCENE`: fixed-candidate source decisions.
- `visualize_grounded_seeds_next85.py`: source-only qualitative review.
- `run_grounded_scope_next85.py --scene SCENE --gpu GPU`: both reconstruction arms.
- `summarize_grounded_scope_next85.py`: all scene and changed-query outcomes.

Selected-Gaussian projection IoU is a2Dprojection metric, not independent3DGT.
2Dsemantic response scores are frozen. NVOS/ScanNet/UQIS and same-map four-query
completion are unchanged. The one-seed hypothesis does not solve text-instance
union or independently prove whole/part identity.

## Completed results and decision

All eight scene evaluations completed. Values below are percentages of selected
Gaussian silhouette projection IoU, averaged across the four scenes. They are
not independently annotated 3D IoU.

| Scene | Retained next80 | Matched source-init next81 | next85 source-init |
|---|---:|---:|---:|
| figurines | 62.879 | 67.162 | 67.099 |
| ramen | 44.931 | 48.324 | 49.594 |
| teatime | 62.145 | 62.049 | 62.033 |
| waldo_kitchen | 57.736 | 55.141 | 57.586 |
| Macro | 56.923 | 58.169 | **59.078** |

The combined identity-provider/source-initialization candidate improves by
2.155 percentage points over the retained recipe. The matched provider change
at fixed source initialization contributes 0.909 points. Do not attribute the
entire gain to the new provider. With semantic initialization, the macro is
56.911 versus 56.923: changing identity evidence alone does not overcome the
old initialization in this solver.

The strongest matched repairs are sake cup 20.33→38.82 and pour-over vessel
3.70→58.82. Against next80, waldo 30.53→82.23 and pikachu 77.51→93.37 also benefit
from source initialization. This does not solve all identity/scope failures:
onion segments and plastic ladle remain zero; ottolenghi falls 83.69→67.68
(already 67.80 in the matched source-init control). Teatime and kitchen scene
means remain slightly below next80. There is no per-query winner assembly.

Preserve next85/source as a higher-mean **LERF research candidate**, while keeping
next80 as the stable reference/default. Do not promote this result to a global
four-query or SOTA claim. This uses a new optional inference provider and extra
compute, not a newly distilled compact student. Semantic response IoU/AP remain
66.930/84.146. No new NVOS, ScanNet or UQIS results were obtained.

## Why the spatula score did not change

Spatula has one scored view (frame 00066). The old and new selected sets contain
17,538 and 18,903 Gaussians; their intersection contains 13,361. Independently
rendering the intersection reproduces all 20,676 positive pixels. Rendering the
5,542 added or 4,177 removed Gaussians alone produces zero alpha, and none of
their centres projects inside this camera image. Fresh candidate rendering is
byte-identical to the saved evaluation mask. Thus this repeated score is not
stale evaluation or an ignored selection update: the changed elements contribute
nothing to the scored view.

This establishes a coverage limitation of this intervention, not proof that
all errors come from multiple instances. The next substantive experiment should
validate whether spatially distinct, identity-supported source hypotheses are
missing from text-union retrieval, and whether their evidence reaches the target
views. Keep reference-image/prompt instance selection separate from text union;
do not merge every positive VLM judgment or treat off-scope uncertainty as a
negative. Also inspect extent purity on accepted candidates (cup fragments and
book extent): identity acceptance alone is insufficient.

## Artifacts

- `output/next85/summary.json`: both complete reconstruction arms.
- `output/next85/source_choices.jpg`: source-only choices, including failures.
- `output/next85/projection_diagnostic/report.json`: independent set rendering.
- `output/next85/retained_outputs.json`: exact retained-output comparison.
- `configs/recipes/next85_research.json`: experiment definition and decision.
