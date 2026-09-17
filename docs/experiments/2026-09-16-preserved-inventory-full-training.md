# Full reconstruction after source-inventory repair

Both scenes completed rebuilt supervision, source-native initialization, full
96D field training, cold-load checks, and real annotated development evaluation
for text, reference image and registered signed prompt queries. **Do not promote
the resulting model as a joint improvement.** Text ranking improves, especially
ramen, but object membership and prompt transfer remain inadequate.

## What was actually rebuilt

The repaired roots under
`/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916/{scene}`
retain query-free regions and append grounded regions. The geometry remains
frozen. Previously compiled region-indexed supervision was not copied forward.

`rebind_source_grounding_inventory.py` checks identical source RGB hashes,
ordered training frames, vocabulary/grounding phrases, native encoder identity,
projection authority, geometry binding and raster dimensions before reusing the
original RGB/text SAM3 predictions. It records the original prediction path and
hash. Only the enclosing inventory binding changes. Changed RGB, encoder or
vocabulary fail closed in tests. The native teacher predictions themselves are
exactly unchanged; this is reuse of identical inference inputs, not relabelling
old region-indexed supervision to pass an identity check.

Grounded text supervision is recompiled against the new episodes and source
geometry. Source-native fusion and the SVD64 basis are fitted afresh to the
preserved training inventory. Structural coordinates and prompt adapters train
from initialization; old trained coordinates are not combined with a newly
rotated semantic basis.

The architecture is the existing protected 64 semantic + 32 structural field.
No unrestricted structural membership head from the rejected experiment is
used. The fixed native decoder reads semantic coordinates for text/image;
RADIO/DINO reconstruction and SAM contour supervision act on structural
coordinates. SigLIP supplies the frozen semantic anchor; its reconstruction
loss is a monitor, not a trainable semantic capability in this protected run.

| Scene | Text tasks / directly supervised concepts | Image / prompt tasks each | Steps | Valid field elements |
|---|---:|---:|---:|---:|
| figurines | 63 / 14 | 461 | 2,766 | 79,500 |
| ramen | 99 / 14 | 852 | 5,112 | 226,194 |

The budget is two complete passes over each image/prompt task, with text cycled
in the same three-modality schedule. This is the previous epoch policy, not the
same absolute step count: previous runs had 2,106 / 3,240 steps and smaller
source inventories. Expanded field support and changed correspondence tasks
also mean this final-model comparison is not a one-variable causal ablation.
The previous exact-common-support fusion experiment remains the evidence for
source-level improvement.

## Verified integrity

- Every valid semantic prefix exactly equals its new source initialization.
- Native decoder exactly equals the initialization basis transpose; bias is zero.
- All initialization-supported elements remain valid; none are excluded as
  untrained at export. Geometric visibility remains larger than semantic support.
- All three modalities have full task coverage and cold-load response error 0.
- Active structural teachers and prompt adapter have nonzero gradients.
- Teacher prediction probabilities are unchanged; new episode/teacher/query/
  geometry bindings match the recompiled supervision.
- Training source files remain hash-identical throughout each run and were
  checked again by the result sealer.

Four new rebinding behavior tests plus ten relevant existing tests pass. Three
existing frozen calibration tests also pass: **17 tests total this iteration**.
These checks establish implementation consistency, not correct object extent.

## Full development evaluation

Values are percentages, IoU / AP. Text uses all 56 / 71 annotated observations.
Image and prompt use 35 / 57 cross-view reference tasks, so they are different
cohorts. Reference masks and two signed points are authorized query inputs;
target RGB is not encoded. These are category-union development targets, not
verified instance tracks, 3D ground truth, or independent final test results.

| Scene / modality | Previous protected field | Rebuilt field |
|---|---:|---:|
| figurines text | 34.63 / 57.64 | 33.14 / 58.30 |
| ramen text | 2.14 / 22.72 | 4.07 / 31.02 |
| figurines reference image | 33.42 / 54.82 | 33.12 / 54.52 |
| ramen reference image | 13.13 / 34.93 | 13.81 / 34.21 |
| figurines registered prompt | 24.36 / 38.84 | 21.60 / 40.28 |
| ramen registered prompt | 22.87 / 38.64 | 21.88 / 32.92 |

Image comparison uses the existing source-only balanced scalar calibration for
both old and new fields; this policy was selected before new benchmark results.
The new joint-training image readouts are degenerate before that stage:

- figurines scale 7.2895, bias -7.8782;
- ramen scale 7.1143, bias -7.8880.

Since cosine cannot exceed 1, `scale*cosine+bias` is always negative, so the
absolute 0.5 readout cannot select any element. Source training image IoU is also
zero. Nonzero calibration gradients and a completed optimizer are insufficient
acceptance criteria. The raw checkpoints are diagnostic intermediates, not
usable uncalibrated image retrieval outputs. The source-calibrated exports
change only the image scale and bias; field features, native decoder and prompt
parameters remain bitwise unchanged.

The sealer identifies `preserved96_protected_imagecal_balanced/field.pt` as the
complete evaluated image-readout candidate, while still marking joint promotion
false. This does not make the remaining image association problem solved.

## Bounded source-only text calibration control

After observing strong ranking but low membership IoU, one additional control
fits balanced positive/negative source-known 3D evidence with the same frozen
monotone calibration routine. It does not use evaluation labels, optimize a
benchmark threshold, or change semantic/structural parameters. Both scenes use
the same rule; no per-scene winner is silently chosen.

| Scene | Original rebuilt text IoU / AP | Source-balanced text IoU / AP |
|---|---:|---:|
| figurines | 33.14 / 58.30 | 25.08 / 56.58 |
| ramen | 4.07 / 31.02 | 11.51 / 30.88 |

This control is rejected as a universal replacement. A monotone element-level
transform can still alter pixel AP because sigmoid and compositing do not
commute. Exact native feature preservation does not guarantee unchanged rendered
ranking or correct absolute membership.

## Conclusions and remaining core problems

The source-evidence repair survives full reconstruction as a text-ranking gain;
ramen AP rises by 8.30 points. It is a necessary correctness repair, not a complete
retrieval solution. Direct text supervision still covers only 14 concepts.

The protected text/image path still predicts membership from native semantic
cosine and a scalar calibration. It does not use the structural suffix to infer
object extent. The rejected unrestricted MLP and this calibration control show
why merely granting a new learned readout access to structure, or adjusting a
single global threshold, is not enough.

Prompt regression must be investigated separately. Source pseudo-IoU in ramen
falls from 83.06% to 79.65%, and real prompt AP falls from 38.64% to 32.92%.
The inventory changes source correspondences, hierarchy mixture, training
support and budget simultaneously; this iteration cannot attribute the decline
to a particular one. Conflict exclusions alone are not evidence of correct
cross-view object identity. The next focused investigation is scale-consistent
object/part membership and query-to-structure transfer, keeping the repaired
source evidence and protected native semantic channel.

No NVOS/ScanNet-OVS promotion or new SAM-style 2D prompt claim follows from this
LERF development evaluation. A new no-RADIO run was not performed this iteration;
previous architectural no-RADIO controls remain separate evidence.

## Reproduction and artifacts

- Training commands and budgets:
  `paper/artifacts/preserved_inventory_training_protocol_20260916.json`.
- Integrity, full metrics and field hashes:
  `paper/artifacts/preserved_inventory_full_training_20260916.json`.
- Full text/reference results:
  `paper/artifacts/{scene}_preserved96_{text,reference}_20260916.json`.
- Rejected text calibration control:
  `paper/artifacts/{scene}_preserved96_calibrated_{text,reference}_20260916.json`.
- Each root contains `grounded_text_supervision.pt`,
  `source_fusion_initialization.pt`, `preserved96_protected/report.json`, raw and
  source-calibrated model artifacts with individual receipts.
- Sealer: `scripts/audits/seal_preserved_inventory_training.py`.
