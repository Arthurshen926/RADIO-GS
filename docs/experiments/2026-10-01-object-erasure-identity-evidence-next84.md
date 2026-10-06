# Object-erasure identity evidence and matched context control (next84)

## Hypothesis and controlled change

Next83 corrected two wrong candidate identities but introduced other failures.
This iteration tests whether candidate-conditioned evidence, rather than raw
query likelihood, can admit the verifier reliably.

Freeze all16candidate pairs/query from next83, all source frames, the local BLIP
model, verbatim runtime queries, Gaussian geometry, native fields and evaluator.
Create1.5xbounding-box crops, preserving the same window for two observations:
full RGB, and the same RGB with only candidate-mask pixels replaced by127gray.
Measure the change in mean autoregressive query log likelihood caused by removing
the object. The weaker of the two source-view gains ranks a candidate. Require
positive gain in both views; if none qualify, retain the original native seed.
This zero-gain rule is fixed from its evidence interpretation, not evaluation GT.

Run a matched raw-likelihood control on the same1.5xwindows, so context expansion
is not attributed to erasure. Both arms use the same source-informed membership
initialization as next81/83. The other54queries retain next80 output exactly.
Only exact same-frame/same-mask tracking predictions are reused; seed evidence
is updated. Otherwise the full authorized source sequence is tracked anew.

Gray erasure is an artificial input intervention, not proof of causal semantic
identity. It can show that an object affects a model's score without proving that
the object matches the requested referent. No inpainting model, benchmark aliases,
manual query override, target image or target annotation is introduced.

## Source observations before evaluation

The source illustration is `output/next84/source_erasure.jpg`. Despite positive
erasure gain in both source views, ketchup selects a wrong green-labeled bottle;
sake cup selects a clear drinking glass. The pour-over choice remains a glass
coffee vessel. Kamaboko is the only query with no positive-gain candidate and
therefore retains its native seed. The fallback rule is real but is not assumed
sufficiently selective: the score can still respond to associated appearances.

This directly distinguishes **visual score dependence** from **correct referent
identity**. A candidate-specific contrast removes some context/prior effects,
but does not automatically turn a caption likelihood into grounded verification.

## Reproduction and boundaries

`counterfactual_identity_next84.py --scene SCENE` computes both source decisions.
`run_counterfactual_scope_next84.py --scene SCENE --gpu GPU` runs tracking,
source-initialized reconstruction and the frozen evaluator for both arms.
`summarize_counterfactual_scope_next84.py` reports all four scenes and all13
changed queries. Exact tracking reuse is documented alongside each run.

The13-query candidate budget is inherited from missing-seed diagnosis, not a
new scene/map image cap. Each query's16pair verifier budget remains explicit.
Semantic score tensors, geometry and other query paths remain unchanged. Selected
projection is not independent3DGT; no NVOS/ScanNet/UQIS/compact-student gain is
implied by these text-query experiments.

## Complete results and decision

All8scene evaluations finished. Selected-set projection IoU(%):

| Scene | Retained next80 | Native/source-init next81 | Tight BLIP next83 |1.5x raw likelihood | Object-erasure gain |
|---|---:|---:|---:|---:|---:|
| figurines | 62.8787 | 67.1621 | 67.0127 | 66.8435 | 67.1348 |
| ramen | 44.9314 | 48.3236 | 49.7354 | 48.0878 | 47.8157 |
| teatime | 62.1447 | 62.0486 | 60.7009 | 62.4406 | 61.9829 |
| waldo_kitchen | 57.7357 | 55.1407 | 55.5461 | 56.4745 | 56.4837 |
| Macro | 56.9226 | 58.1688 | 58.2488 | 58.4616 | 58.3543 |

**No promotion; retain next80.** Erasure loses0.1073pp against the matched-window
raw-likelihood control. Neither resolves the dangerous candidate substitutions.
These58.xvalues are source-initialized experimental branches, not a new accepted
mainline score. Raw context expansion slightly improves the tight-crop experiment,
but does not establish a generally reliable identity verifier.

The erasure rule admits12/13queries; kamaboko alone falls back to its native seed.
Ketchup's wrong bottle passes the both-view rule and still yields0IoU. Sake cup
falls from next83's38.15 to15.85% after switching to a drinking glass. Pour-over
vessel remains59.57%, compared to59.12%for matched raw crops and60.66%for tight
BLIP. The rule therefore does not preserve all the demonstrated content gains.
No threshold adjustment or per-query override was made after seeing these scores.

The conditional score clearly depends on candidate pixels in many cases, yet
that dependence can be driven by generic container appearance instead of the
requested object. Gray erasure also preserves the mask silhouette and produces
an artificial image. It cannot be treated as a calibrated semantic correctness
test. Near-zero positive gains, such as the admitted onion candidate, provide no
established reliability margin; we do not tune one using benchmark outcomes.

## Research decision

Stop prioritizing further BLIP likelihood, crop-context or erasure-score variants.
The experiments establish a missing verifier capability, not evidence that one
more confidence transformation will fix identity. Next compare a genuinely more
capable fine-grained/grounding provider on the *same saved candidate set*, using
source-only content/whole-part decisions, before spending on more reconstructions.
The comparison must permit uncertainty, retain verbatim query meaning and preserve
native text/image spaces. A stronger provider is a hypothesis to test, not an
assumed fix or a mandatory architectural dependency.

Mainline artifacts and default behavior remain unchanged. No per-query winner
assembly, new unit-test framework, threshold search, source-frame cap, target-GT
supervision or other-benchmark claims. Exact retained semantic/other-query outputs
are recorded in `output/next84/retained_outputs.json`. Detailed results are in
`output/next84/summary.json`.
