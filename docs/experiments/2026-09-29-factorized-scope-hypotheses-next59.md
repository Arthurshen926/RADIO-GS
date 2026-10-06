# Factorized source scope hypotheses — next59

## Result and scope

Three full-four-scene fixed-recipe comparisons complete. Broader per-instance
containment candidates give selected-projection IoU48.354 versus next56 47.863
(+0.491pp). Three scenes improve and figurines regresses. This is not the requested
large aggregate breakthrough. Semantic2D IoU58.087/AP74.965 remains unchanged.

The mechanism repairs specific major failures: coffee46.77→56.79, refrigerator
65.15→87.00, hooves25.48→47.17, but causes regressions elsewhere. Preserve the new
hierarchy artifact as a development candidate and next56 as the established
comparison; do not assemble per-query winners or promote a global mainline default.

| Method | figurines | ramen | teatime | waldo_kitchen | Macro |
|---|---:|---:|---:|---:|---:|
| next56 decision |47.633|35.063|58.820|49.935|47.863|
| Joint view scope choice |47.748|35.215|59.656|50.517|48.284|
| Per-instance choice, same compatible candidates |47.748|35.219|59.575|50.469|48.252|
| Per-instance small enclosing candidates |47.475|35.092|59.762|51.087|48.354|

All values are selected Gaussian subsets projected against2D annotation masks,
not independent physical3D GT. These are reused development scenes and queries;
no final-test or new ScanNet/NVOS/UQIS result is established.

## Mechanism

All arms share next52 frozen geometry, next54 retained membership initialization,
next56 annealed/deployment-aligned decoder,120updates/query, source frames and
identity caches. Native semantic scores and hard evaluation are unchanged. No
new teacher, target-image access or query-specific alias is used.

At each update, the current source subset rendering chooses a scope hypothesis.
For a candidate family, define a local comparison window as a3x3 dilation of
its union. Choose the candidate with highest soft IoU with the current rendering
within that window. Stop gradients through this discrete choice. Merge the
chosen instance scopes under the existing text-union task, then form eroded
positive/dilated-complement negative masks. The existing native-field agreement
still qualifies the actual positive and negative write. Optimize the same
subset coverage BCE/Dice objective as next56.

This is a hard alternating geometric heuristic, not probabilistic identity
tracking, a converged EM algorithm, or a learned language-granularity model.
Current-render agreement is self-referential and may reinforce a wrong extent.
Distinct detections are not automatically verified physical instances.

* `joint_view`: compare the union of original detections with the union of
  source-only native-compatible enclosing proposals (next49 cached relations).
  All instances change scope together. This is a stronger-current-decoder control
  for earlier whole-view hypotheses, not a newly invented mechanism.
* `per_instance`: make the same literal/enclosure choice independently for each
  detection, then union chosen scopes. Containment requires90% seed coverage.
* `hierarchy`: keep each literal detection plus at most three smallest-area source
  SAM regions larger than it and covering at least90% of it. Each alternate is
  unioned with the seed to avoid discarding its positive support. When none exist,
  keep the prior compatible candidates. This is a bounded local containment family,
  not a complete learned object/part hierarchy. No native-score preselection of
  those three regions occurs, but native qualification still constrains writing.

The source region masks, source images, candidate relations and all GT files stay
unchanged. All choices are made before target evaluation using source predictions.

## What the comparisons support

Per-instance factorization alone is NOT better than joint view choice here:
48.252 versus48.284. Do not claim that factorization by itself resolves entity
mixing or is a proven contribution. A more flexible candidate family raises the
macro only0.102pp over the same-compatible-candidate instance arm.

However, the refrigerator gain specifically appears after allowing additional
containing candidates:65.15% in the joint arm versus87.00% in the broader arm.
Coffee improves already with joint hypotheses to56.79%; it should not be credited
to the new candidate expansion. These observations distinguish candidate support
from factorization instead of attributing every gain to the proposed design.

Query-averaged tradeoffs, next56→hierarchy:

| Query | Before | After |
|---|---:|---:|
| refrigerator |65.15|87.00|
| hooves |25.48|47.17|
| coffee |46.77|56.79|
| sink |64.66|71.83|
| old camera |42.74|47.76|
| bag (figurines) |47.85|40.14|
| bear nose |15.82|10.19|
| three cookies |78.89|73.09|
| egg |49.32|47.40|

Thus more scope capacity can restore missing coverage and also choose the wrong
requested granularity. It does not solve teacher entity mistakes or absent
concept evidence. The query-time native qualifier remains imperfect. These
numbers are diagnostic, not a basis for per-query deployment exceptions.

## Research consequence

The next missing variable is requested granularity/relationship, not merely a
larger geometric support or independent per-instance switches. The existing
selector sees rendering agreement; it does not explicitly determine whether a
query requests an entity, a part, contents, container or collection. A useful
intervention must distinguish these without benchmark-specific label aliases,
and must retain whole/part and set examples in independent evaluation. Geometry
agreement cannot substitute for that semantic evidence.

Current online source-detector and prior-field dependencies remain, and2D retrieval
has not improved this round. No coherent offline four-query model, full teacher
retraining, independent generalization or CVPR-level contribution is claimed.

Artifacts:
* `scripts/research/decode_factorized_scope_next59.py`
* `scripts/research/run_factorized_scope_next59.py`
* `scripts/research/summarize_factorized_scope_next59.py`
* `output/factorized_scope_next59/summary.json`
* `output/factorized_scope_next59/hierarchy/lerf_<scene>/typed_readout.pt`

All started jobs finished. No tests or audit framework were added.
