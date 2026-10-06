# Where the restored small-region response still fails

This diagnostic uses the frozen restoration pilot artifacts only. No new model,
mask, query, fit, per-region gate or benchmark evaluation is introduced. The
restoration decision remains rejected because both source pilots did not pass.

For the same22 supported figurines heldout small masks, direct64 projection
retains substantial native teacher discrimination, while fusion and geometric
readback largely lose it. Thus the large remaining gap is after direct feature
compression. That stage includes cross-view correspondence and geometric
readback as well as feature averaging; this is not proof that geometry is exact.

| Source response stage | Text MSE | Image MSE | Text Top1 | Image Top1 |
|---|---:|---:|---:|---:|
| Direct small native teacher→same64basis |0.0000960|0.0013532|59.1%|68.2%|
| Original fusion→heldout actual readback |0.0005788|0.0078789|0%|0%|
| Restored inventory fusion→same readback |0.0005548|0.0080949|0%|0%|

All errors and Top1 agreements use the same raw1536D teacher responses as their
reference. Direct projection follows the deployed element normalization and
full-native query norm contract. The rows are controlled diagnostic stages,
not additive causal losses. Native teacher recognition itself is not ground
truth; we cannot declare that its small crop names the real object correctly.

The original map's image MSE is5.82times the direct projection error. More
strikingly, centered image response error rises from0.0000383 to0.0034376,
about89.7times. The map loses the pattern of which original source image queries
match the small region, beyond the direct projection's response offset.

## What the source overlaps show

Fourteen of22 supported small masks overlap any retained source mask. Eight
have no parent mask overlap in that heldout source view; their missing context
is unknown, not a zero-valued negative teacher.

Eighteen masks have an exactly positive image-MSE increment after restoration,
including some negligible numerical changes. Among these18,13 overlap retained
masks and8 have a retained region covering at least90% of the small mask while
having at least twice its area. The latter cases account for62.2% of the total
**positive error increment**; all parent-overlap cases account for78.4%.
This is an overlap-associated share, not a causal attribution percentage.

For the13 overlapping regression cases, the tightest maximum-overlap retained
region is34.4times larger on average. Its renderer footprint supports94.2% of
the small region's normalized renderer mass, while its native teacher feature
has mean cosine0.710 to the small crop's teacher. The original overlapping
teacher mixture has cosine0.718. Different granularities therefore share much
of the carrier while requesting noticeably different semantic directions.

The remaining21.6% of positive increment occurs outside every retained mask in
the heldout source view. Whole/part collision is supported by the data but does
not explain all failures. Those uncovered masks can still read features written
from other training views. New-small-crop recognition error, correspondence
error and unseen-part behavior cannot be uniquely separated without independent
object truth.

Geometry-supported pixels are available for nearly all of the22 evaluated
masks:21 have full raster support and one has14/18 pixels. Two additional
heldout masks lack the required observed support and were excluded identically
by the registered paired evaluation. High support establishes a carrier, not
correct cross-view identity or precise geometry.

## Why this particular restoration worsens total image MSE

The restored map improves centered image error by6.52%, but its mean response
is still too high relative to the native teacher. Mean bias grows
0.05752→0.06083; mean squared per-region bias grows0.0044413→0.0048812
(+9.91%). That increase outweighs the centered-error improvement and produces
the+2.74% total image-MSE regression.

This distinguishes the **new regression** from the **large existing deficit**.
A response offset explains this experiment's failed image-MSE gate, but it
cannot explain away the fused field's0% teacher Top1 or its much larger centered
error. Recalibration alone is not established as a solution, and no calibrator
was fitted in this diagnostic.

The next method change should preserve compatible small/whole scope evidence
through fusion and validate its cross-view identity. Merely appending small
regions to the same averaged field supplies more evidence but does not ensure
that the evidence remains distinguishable when retrieved.

## Correctness and preserved attempts

The first diagnostic attempted to summarize an absent parent mixture with a
zero vector. That would incorrectly turn missing source context into teacher
evidence. Its files remain under
`output/small_region_restoration_diagnostic_next32/`; use the corrected
`output/small_region_restoration_diagnostic_next32_v2/figurines.json` instead.
The correction marks absent context as unknown/null and limits parent-mixture
metrics to the14 masks that have context. It does not change any map, paired
prediction, source gate or22-mask projection/readback result.

The corrected CPU readback reproduces both pilot image/text MSEs within2e-7.
All source files, teacher tensors and maps are SHA-bound. No additional VFM,
GPU inference or benchmark data is used.

Seal: `paper/artifacts/small_region_restoration_diagnostic_next32.json`.
