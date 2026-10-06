# Content-conditioned candidate identity (next83)

## Question and change

Next81 showed that better initialization can reconstruct the wrong candidate
more faithfully. Next82 showed that mask/geometric agreement does not supply
missing semantic identity. This experiment changes **which source candidate is
selected**, instead of adding another geometric or confidence weighting rule.

All authorized source-region pairs compete under the existing native SigLIP
query cosine and frozen mutual geometric correspondence. For each of the13
previously seedless queries, retain up to16 candidates, suppressing near-duplicate
Gaussian supports with Jaccard>0.8. The entire source collection is searched;
16 is an explicit bound on verifier inference, not a source-frame/map coverage
limit. Whole/part candidates are not categorically merged by containment.

The local `Salesforce/blip-image-captioning-base` model reads tight unmasked RGB
crops from both views of each pair. Score the mean autoregressive log likelihood
of the **verbatim runtime query**, including its end token, conditioned on each
crop. Use the weaker of the pair's two scores for reranking, then seed tracking
from the higher-likelihood view of the winning pair. The initial decoder token
is BLIP's configured BOS; logits predict subsequent query tokens. Query lengths
are constant within each candidate comparison. These scores are not calibrated
identity probabilities, and caption-model language biases remain possible.

Keep native-pair cosine as the existing seed-confidence definition. Generated
captions are saved only after selection for diagnosis and do not rank candidates.
No query aliases, evaluation vocabulary expansion, manual source labels, target
RGB or target annotation enters construction.

Run the same full-source SAM3 tracker and frozen geometry/semantic field. Refit
only the same13queries, retaining all54other selections. Compare both semantic
initialization (matched next80) and source initialization (matched next81 regional
control), using the same optimization budget and frozen evaluator. This adds a
local inference teacher and changes candidate identity/view; it is not an
identical-compute comparison, a new compact field, or a unified four-query result.

## Source observations before evaluation

`output/next83/source_choices.jpg` contains source-only crops. Sake cup changes
from a bottle to a cup; pour-over vessel changes from cookware to a glass coffee
vessel. Kamaboko retains a fish-cake region in a different view. This establishes
that content-conditioned reranking can change specific incorrect source choices;
it does not yet establish final recall or correctness of every selected region.

The model's generated descriptions sometimes remain coarse or inaccurate (for
example Pikachu is described as a decorated cup), illustrating why generated
captions are not accepted as ground-truth identities. No visual-review edits are
applied to selected seeds.

## Reproduction and boundaries

- `rerank_content_seeds_next83.py --scene SCENE`: full-bank search and BLIP rerank.
- `run_content_scope_next83.py --scene SCENE --gpu GPU`: source tracking and both
  matched scope reconstructions/evaluations.
- `visualize_content_seeds_next83.py`: source-only before/after crops.
- `summarize_content_scope_next83.py`: four-scene, matched-parent and query results.

The geometric bank and correspondence are unchanged. Native semantic scores and
other-query selections stay fixed. A single chosen scope is still not exhaustive
text-instance union. Selected projection IoU is not independent3Dground truth;
NVOS/ScanNet/UQIS and compact multi-teacher distillation are unchanged.

## Complete outcomes and decision

All8scene evaluations complete. Selected-set projection IoU(%):

| Scene | next80 | BLIP / semantic init | next81 regional source init | BLIP / source init |
|---|---:|---:|---:|---:|
| figurines | 62.8787 | 62.9682 | 67.1621 | 67.0127 |
| ramen | 44.9314 | 45.0693 | 48.3236 | 49.7354 |
| teatime | 62.1447 | 62.1302 | 62.0486 | 60.7009 |
| waldo_kitchen | 57.7357 | 55.7045 | 55.1407 | 55.5461 |
| Macro | 56.9226 | 56.4680 | 58.1688 | 58.2488 |

**No promotion. Retain next80.** Semantic-initialized reranking regresses0.4546pp;
source-initialized reranking adds only0.0800pp versus its matched next81 parent,
with substantial query regressions. Its58.2488%macro is not a new accepted
mainline score or a uniformly better method. No per-query/scene winner assembly.
2D semantic IoU66.9301%/AP84.1459% remain frozen.

With the same source initialization, corrected candidate identity raises
pour-over vessel3.70→60.66% and sake cup20.33→38.15%. These gains establish a
concrete recoverable failure in original candidate identity selection. Changing
the same candidates with semantic initialization produces only48.26/21.62%,
showing that reconstruction's starting state still limits use of new evidence.
Neither phenomenon alone explains the whole system.

The new provider also introduces severe errors: ketchup48.04→0% under source
initialization (40.93→0% under semantic initialization), and paper napkin59.51→
43.03%. BLIP chooses a tiny green-labeled bottle for ketchup; its own diagnostic
caption calls the crop a "green and brown truck driving down a road". This is a
provider inconsistency, not evidence that the crop is a truck. The candidate's
high relative query likelihood does not establish reliable visual grounding.
A forced winner among16candidates has no representation for provider uncertainty.

## What this establishes for the next method

- Content-level candidate discrimination can fix errors that geometric/mask
  consistency could not: bottle versus cup, cookware versus coffee vessel.
- A generic small caption model's query likelihood is not a trustworthy universal
  verifier, especially for tiny crops, unusual names and fine object parts.
- Source-initialized membership can exploit a corrected identity, but also
  amplifies an incorrect one. The two stages need a shared explicit uncertainty
  policy rather than unconditional substitution of whichever teacher scores
  highest.
- A stronger source-grounding provider, or source-validated abstention against
  native evidence, should be assessed *before* another full scene rebuild. Do not
  infer an abstention threshold from these evaluation masks or rescue particular
  queries by hand. Preserve all-source native capability and exact runtime-query
  semantics; neither BLIP nor any particular verifier becomes mandatory.

Caption semantic consistency is only a diagnostic here; captions were generated
after selection and were not used to construct predictions. No claimed repair
of these newly introduced verifier failures has been implemented this round.
Native map/query coordinates, geometry and the original retained recipe stay
unchanged. Research contributions and SOTA readiness remain unestablished.
