# Fine-pixel source ownership: fixed granularity conflict control

The registered rule improves small-region text response beyond plain inventory
restoration, but both source pilots fail their full preservation gates. It is
rejected without a benchmark run or head refit. Choosing finer evidence alone
does not establish compatibility with whole-object and image-query responses.

## One rule, fixed before results

Use exactly the original retained inventory plus the cached restored small SAM
inventory. For a retained mask to lose a source-writing vote, its area must be
strictly greater than twice the restored-small mask's area and it must cover at
least90% of that small mask. Suppress its vote only on their intersection. Keep
all small masks, then apply the existing equal partition among remaining masks
at each pixel. Geometry,64basis, observed support, descriptors, query bank and
actual readback remain unchanged. No new VFM call or model fit is made.

Containment is only a source-writing relation. It is not a same-instance label,
whole/part identity annotation or a new negative supervision label.

Pixel coverage is exactly conserved. Normalized writing weights outside
affected pixels are bitwise unchanged. A retained mask without a qualifying
child keeps its Boolean support, but its normalized share can increase inside
an affected pixel when another mask loses a vote. A view with no qualifying
parent/child relation is an exact no-op relative to plain restoration.

These are source-writing invariants. An element can cover several pixels, so
changing its aggregated feature can change rendered responses outside the
affected source pixels. No final-render outside-invariance claim is made.

Registration: `output/fine_pixel_ownership_next32/registration.json`.
Independent review completed before GPU launch; its receipt is bound in the
final seal. Six CPU tests check containment orientation, strict area ratio,
overlap-only changes, conserved pixel/lift mass, equal fine-mask partition,
no-parent/no-small no-ops and malformed inputs.

The gate retains every previous requirement: both modalities' small-region MSE
must improve at least5% relative to the original map; retained-region and
boundary MSE can worsen at most1%; small/retained Top1 drops at most1point;
support/basis must be identical. Additionally, against plain restoration,
centered MSE and teacher Top1 must not worsen for small, retained and boundary
populations, with1e-6 numerical tolerance. No new query-pair labels are invented.

## Complete fixed-source results

Changes below are relative to the original map, not to plain restoration.

| Source pilot / method | Small text MSE | Small image MSE | Retained image MSE | Boundary image MSE |
|---|---:|---:|---:|---:|
| figurines plain restoration |−4.15%|+2.74%|+0.86%|+0.58%|
| figurines fine ownership |−7.98%|+4.71%|+1.52%|+1.29%|
| ScanNet0000 plain restoration |−15.72%|−7.71%|−0.45%|+0.87%|
| ScanNet0000 fine ownership |−16.54%|−8.21%|−0.40%|+1.203%|

Figurines now clears the small-text improvement gate but fails image response,
retained image and boundary protection. It also loses some teacher Top1 versus
plain restoration. Its small-region text/image Top1 remains0%/0%.

ScanNet0000 improves small text and image errors; small image Top1 rises from
4.76% to9.52% (one additional agreement among21 regions). It passes the extra
comparison against plain restoration. However, its boundary image MSE increases
1.203%, exceeding the fixed1% gate, so it too is rejected. That threshold is
not relaxed after the result. These are teacher agreement metrics, not object
ground truth or benchmark accuracy.

## Actual scope and correctness

Figurines changes1873 source pixels across20 training views under351 containment
relations, removing4118 coarse region/pixel votes. ScanNet0000 changes1489 source
pixels across32 views under320 relations, removing3842 votes. Counts across
views refer to source raster positions, not unique3D points or objects.

Coverage and outside-affected writing weights are exact for every view. Maximum
per-element source-mass difference versus plain restoration is3.81e-6 and
7.63e-6, respectively, within the preregistered floating-point reduction bound.
Original and plain-restoration map replays agree within2.98e-7. The source
observed domain, basis and unobserved coordinates remain unchanged.

The original retained masks still define the old heldout supervision targets;
ownership-adjusted masks are used only for training-view source writing. Thus
the rule cannot pass by rewriting the evaluation masks to match itself.

The result is consistent with a tradeoff: reducing coarse/fine averaging helps
small detail but can displace useful context or whole-range responses. It does
not verify which source mask has the correct physical object identity. The
large fusion/readback deficit identified in the preceding diagnosis remains;
this fixed size/containment rule is insufficient to resolve it reliably.

Reports: `output/fine_pixel_ownership_next32/{lerf_figurines,scannet_scene0000_00}/report.json`.
Seal: `paper/artifacts/fine_pixel_ownership_next32.json`.
No prior sealed experiment, default method, benchmark result or calibration is
modified.
