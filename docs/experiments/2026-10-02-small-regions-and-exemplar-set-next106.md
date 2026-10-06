# Small-region eligibility and visual-exemplar set discovery (next106)

## Main finding

The failure precedes field reconstruction in two distinct ways. A relative area
floor removes legitimate fine-grained source evidence, and a single-instance
tracker is then treated as if it represented a complete text-query set. Relaxing
eligibility recovers a precise but incomplete onion fragment. Using that fragment
as an automatically retrieved visual example enables additional same-category
source discovery, improving selected-set projection without changing geometry or
native semantic scores. This is a targeted development experiment, not a general
benchmark promotion or a claim of independent 3D segmentation accuracy.

## Source evidence and proposal funnel

The native encoder in `encode_source_regions_next72.py` admits masks only when
0.002 <= mask area / image area <= 0.6. On all authorized source images:

| Scene | Source views | Raw masks | Old encoded masks | Omitted small masks >=64 pixels |
|---|---:|---:|---:|---:|
| ramen |122|9344|8719|625|
| waldo_kitchen |180|14537|13009|1495|

There are no raw masks below64pixels in either cohort. Kitchen also has33masks
above the old upper-area bound. The bounded complement encodes only previously
omitted small masks with the **same** native SigLIP model, gray-masked tight crop,
normalization and original-resolution support. It does not regenerate proposals,
change the field, use target images, or supply query aliases. The64pixel floor is
an explicit experimental eligibility rule, not a validated universal threshold.

Adding these descriptors contributes16/6new regions to the top64 for onion
segments/plastic ladle. We inspect the union of old/new top64:80+70single-region
checks using the unchanged exact-query Qwen3B A/B/C prompt. Positive margins occur
for13/35regions, including1/4small regions. These are model decisions, not verified
labels. Visual source review still finds egg accepted as onion, and diverse spoon
appearances accepted as plastic ladle. The expanded native top1 onion region is
classified C with margin−1.125, despite subsequent useful geometric recall.

Replay reproduces both original next88 pair lists exactly. Of128anchor decisions,
ramen has22selected,103budget-limited,2deduplicated,1withoutpositiveoverlap;
kitchen21/95/3/9. These are anchor decisions; multiple anchors can share a pair.
A green food region (node8032, frame122) has native rank28, but its maximum-Jaccard
partner in the128pool has rank126 (overlap0.04645). Their minimum-native pair score
ranks92, outside the16pairbudget. This is a concrete ranking loss; weak overlap is
not proof of a correct correspondence, and this region is not manually injected
as a prediction. A kitchen spoon-shaped region at rank28 similarly loses at pair
rank28. Its exact material/target identity remains unestablished.

## Controlled intervention and method change

First compare old-bank native top1 and expanded-bank native top1 for the already
identified zero query `onion segments`. Both bypass pair ranking and the unreliable
VLM, use the same single-mask SAM3 tracker over all122authorized source views, and
fit the same frozen next104 geometry/decoder. Native confidence, selected source
frame and mask necessarily change with the retrieved region. Neither prediction
uses evaluation labels. The old seed is frame42/node2512; the new seed is
frame115/node9204. This tests the area intervention **within a native-top1 path**,
not a claim that area filtering alone explains all differences from next104.

After this control revealed precise but incomplete recall, an exploratory second
stage treats the tracked core as a **visual example**, not a complete answer:

1. In each of114source views with a present nonempty core, compute its normalized
   bounding box. Other views retain the original tracking record.
2. SAM3 sees the full authorized source RGB, the exact query and this positive
   geometric example, with the existing0.5confidence threshold.
3. Preserve the core and union discovered instance probabilities in that view.
   Keep individual instance probabilities in the transient cache as well.
4. Rebuild source supervision and the actual selected Gaussian subset with the
   same balanced loss, footprint prior0.1,192x256raster and corroborated negatives.

A matched **text-only** control uses the same114views and core union, but omits the
visual box. It detects zero additional instances. Visual-example discovery expands
104views; sampled core/union pixel counts sum to3763/12067. This adds authorized
source-image inference and is not an equal-cost field-only comparison. It remains
an automatic text-query pipeline: the example is retrieved from sources, not a
user prompt or an annotated evaluation crop.

The discovery experiment was designed after the first control's development
result; it is not independent-test validation. Text union is appropriate to this
query's existing set contract. It must not be generalized to reference-instance
queries, or used to merge whole/part identities indiscriminately.

## Results

Same frozen evaluator and selected-set silhouette output:

| Control | Onion IoU | Mean pixel precision | Mean pixel recall | Ramen selected mIoU |
|---|---:|---:|---:|---:|
| Old-bank native top1 |0.000%|0.000%|0.000%|56.9085%|
| Small-region complement, native top1 |7.8963%|100.000%|7.8963%|57.6871%|
| Same core + text-only discovery |7.8963%|100.0000%|7.8963%|57.6871%|
| Same core + text/visual-example discovery |13.0706%|98.9437%|13.0895%|58.1972%|

All7onion target views have positive IoU after the small-region intervention.
Metrics are means over the query's evaluated views. Precision/recall uses the
same byte10mask threshold and is computed only after predictions are frozen.
Selected-set projection is a2Dsilhouette measure, not independent3Dgroundtruth.
The pure native arms preserve native semantic tensors and every other query's
selection column exactly; the same checks are included for discovery outputs.

## Decision and next work

Do not promote this one-query experiment as a new full-benchmark default. Keep
next104's62.598590%four-scene selected macro as the retained candidate and next80
as stable. No new LERF semantic-IoU, NVOS, ScanNet or four-query/UQIS gain is claimed.

The gain resolves a real part of the zero-query failure but is far from sufficient:
roughly87%of annotated onion pixels still go unrecalled. The next method priority
is **multiple semantic examples and typed set discovery**, with source-independent
checks for compatible identity/extent. Preserve single-view hypotheses until
correspondence is resolved; do not use minimum two-view semantic score as the only
way to expose a candidate. Do not turn VLM A/C labels into truth or simply union
all native candidates. Whole objects, parts and reference instances need separate
scope rules while sharing the frozen map and native semantic representation.

Reproduction scripts are under `scripts/research/*next106.py`. Source galleries,
all150checks, exact pair-funnel replay and metrics are in `output/next106/`.
Raw small-region descriptors are in `output/small_regions_next106/`; source tracks
and the four control outputs remain separate for review. No unit tests, model
training vocabulary, new external teacher download or benchmark label-based
prediction selection were added.
