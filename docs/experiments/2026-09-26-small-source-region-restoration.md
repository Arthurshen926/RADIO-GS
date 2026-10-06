# Small source-region restoration: two fixed source pilots

The area filter is a real source-capability bottleneck, but removing it alone
does not pass the registered joint gate. ScanNet0000 benefits; figurines fails
the small-region image response gate. Neither map is deployed and no dependent
head refit or benchmark run is authorized by this result.

`prepare_four_teacher_retrieval.py` rejects masks below0.002 of the96x128 raster
before native crop encoding. Thus1–24-pixel proposals never receive their own
SigLIP descriptor in that inventory. Native SAM already filtered its proposals
by its own quality criteria. This is a method/data-selection limitation, not a
claim that every excluded proposal is a correctly identified small object.

The intervention restores only nonempty query-free native SAM proposals below
that old cutoff. It retains the old large-region exclusion and does not alter
existing masks/descriptors. Native-mask bounding boxes, gray127 background,
three-pixel padding,384x384 BICUBIC resize, SigLIP bundle and half descriptor
storage exactly follow the original recipe. Native indices and old retained
mask rasters replay against the raw source manifests.

All new and old descriptors use the same64-coordinate basis and the current
normalized pixel-partition fusion. Only training views write the field. Geometry
and observed support stay fixed; unobserved rows retain their original values.
No opacity weighting, calibration, threshold, old query head or scope head is
changed or reused. New masks can change the overlapping-pixel partition, as
required by the unchanged fusion formula applied to the augmented inventory.

Registration: `output/small_region_restoration_next32/registration.json`.
The text query bank is the original79 WORDS prompts. The image query bank is512
evenly spaced **original training** region descriptors, excluding all restored
and heldout descriptors. Heldout small masks are identical for the two maps.
Original retained-region and boundary targets remain identical too. The native
query is normalized before projection; projected queries are not renormalized.

The predeclared gate requires both pilots and both modalities to improve
heldout small-region MSE by at least5%, while original retained-region and
boundary MSE worsen by at most1%, small/retained Top1 drops by at most1point,
and observed support/basis remain identical. Metrics are teacher response
agreement, not semantic ground truth or actual benchmark IoU.

| Source pilot | Restored train / heldout | Supported heldout | Small text MSE change | Small image MSE change | Gate |
|---|---:|---:|---:|---:|---|
| figurines |228 /24|22 /24|−4.15%|+2.74%|Fail|
| ScanNet0000 |283 /21|21 /21|−15.72%|−7.71%|Pass|

The two unsupported figurines masks are excluded identically in both arms;
their absence is reported, not hidden by changing the evaluation population.
For figurines, small-region teacher Top1 remains0% for both text and image.
For ScanNet0000, small text Top1 rises9.52→28.57%; image Top1 remains4.76%.
The populations are small and their targets are teacher-generated.

Retained-region MSE changes for text/image are+0.10/+0.86% in figurines and
−2.19/−0.45% in ScanNet0000. Retained boundary MSE changes are+0.19/+0.58% and
−0.38/+0.87%, respectively. Thus old-capability protection passes, but new
small-region image preservation is not jointly established.

Both baseline fusion replays agree within2.98e-7. The candidate never changes
the64 basis, geometry binding, observed bitmap or unobserved coordinate rows.
No benchmark queries/labels or target RGB are read. Three focused CPU tests
cover the area boundary, empty-mask rejection, native crop detail/background
and input immutability. Independent read-only review found no blocking issue.

A post-gate source diagnostic also finds6 restored masks in each scene whose
coarse raster equals an existing retained raster, plus3/4 duplicate additions
within the restored inventories. Their native masks/crops can still differ.
These collisions were not used to alter the registered rule or gate; they
illustrate that different native proposals can share the same coarse carrier.

The result supports preserving small evidence as a design requirement. It does
not support unconditional restoration into the existing averaged field. The
next diagnosis must separate loss in the frozen64 projection from collisions
with broader regions during fusion/readback. It must not treat a newly encoded
small crop as a verified semantic annotation.

Pilot reports: `output/small_region_restoration_next32/{lerf_figurines,scannet_scene0000_00}/report.json`.
Seal: `paper/artifacts/small_region_restoration_next32.json`.
