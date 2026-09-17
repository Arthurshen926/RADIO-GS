# Native teacher retrieval implementation and geometry detail diagnosis

## Completed implementation

1. The actual canonical rendering finetuner now loads sealed frozen geometry
   directly. New candidates verify full Gaussian state, geometry report identity,
   calibrated camera matrices and metric clipping planes. Source frames are
   restricted to the geometry mapping pool; feature validation views must remain
   outside teacher training views. Geometry-heldout frames are not mislabeled
   as benchmark frames. Historical Method-v1 stage/lineage checks are retained
   for historical fields. The new canonical-v1 candidate path is explicit.

   A real 64-dimensional RADIO initialization and four rendering updates ran on
   figurines (train 3/6/8/11; feature validation 16/18), with complete geometry
   binding unchanged. Validation cosine .5355405 -> .5355554 is a smoke result,
   not a meaningful quality improvement or accepted compact model. Initial
   PCA used 4096 rows and one training epoch; this is not the full D512 run.
   Artifacts: frozen_radio_teacher_pilot_20260915/figurines/field64*.pth.

2. build_frozen_geometry_authority.py constructs fresh exact contributions from
   geometry and cameras without feature models, feature manifests or RADIO
   validity gates. It retains original geometric weights; each consumer owns
   its semantic validity. Eight source mapping frames were built on the
   274805-row figurines geometry. The finalized v2 pilot authority hash is
   0e7035c7025ee8af8d9aab52a3884d09d4bda1671b5692b30672666a65b01109.
   The initial pilot directory lacked required builder lineage fields and was
   rejected by the existing loader; it is not a valid downstream artifact.

3. Reused the independent native DINOv2 producer against the new authority.
   It extracted source RGB directly, with an audit hook rejecting RADIO weights
   and canonical RADIO feature-cache reads. No RADIO teacher/model was loaded.
   Six train frames (6/8/11/16/21/23), two feature-validation frames (3/18).
   Native dimension 768; train valid rows 69100; validation valid rows 52090;
   overlap 49688. This proves the new producer branch is RADIO-independent,
   not that every historical CLI has been converted.

4. ElementRetrievalField stores one learned per-element latent and configured
   modality-specific linear query adapters, with explicit observed/unknown
   state. It has no required VFM, feature decoder or geometry optimizer.
   The cold loader verifies file hash, full geometry binding and query encoder
   identity; dimension equality alone is rejected. Text/image/prompt adapter
   mechanics pass tests, but only the image path has real training evidence.

5. A real native-DINO image-query pilot trained 64-dimensional scene latents and
   a 768->64 query adapter jointly for 48 steps. Six source views supervise
   16 deterministic image-patch queries selected from one training view.
   Primary loss: BCE against native teacher query-response probabilities after
   rendering element scores with the exact frozen contribution matrix.
   An initialization drift penalty of .1 is auxiliary. Geometry is unavailable
   to the optimizer. Unknown element evidence is normalized after visibility,
   without changing alpha or allowing hidden surfaces to become visible.

## Retrieval evidence and limitations

Use native_image_retrieval_pilot_support_v2_20260915.json. The initial pilot
without post-visibility unknown normalization is retained as a diagnostic only.

| Feature validation frame | Fixed evaluated coverage | Initial response MSE | Trained response MSE |
|---|---:|---:|---:|
| 3 | .95652 | .0454097 | .0296181 |
| 18 | .94074 | .0332421 | .0172119 |

Both field and query adapter receive gradients. Cold reconstruction of saved
scene state reproduces scores bit-exactly. These numbers measure native
teacher response preservation, not object IoU or ground-truth recall. The
validation queries were already used during training: this tests new views,
not new queries. There is no text/prompt semantic supervision, instance mask
supervision, null calibration experiment, full-scene experiment or final
benchmark promotion yet.

The native branch assets are in:
 /mnt/pool/sqy/results/RADIO-GS/output/frozen_native_dino_pilot_v2_20260915/figurines
The deployment candidate is image_retrieval64_support_v2.pth; teacher caches
and supervision matrices are training artifacts, not required scene state.
The query encoder is native dinov2_vitb14, checkpoint hash
0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73.
The independent cold-native audit blocks both RADIO and training-feature
cache access and encodes a query patch afresh from source RGB.

## Geometry detail: a concrete failure isolated

On mapping development frames 3/6/8, source-depth points were rendered back
into their own view, first with the fixed 7.5 mm tangent / 1 mm normal sigmas,
then with tangent sigma .35*z/sqrt(fx*fy) and normal sigma one tenth of that.

| Frame | Fixed-sigma roundtrip AbsRel | Pixel-footprint roundtrip AbsRel |
|---|---:|---:|
| 3 | .07815 | .00293 |
| 6 | .03935 | .00226 |
| 8 | .04055 | .00198 |

Coverage is 100% in both roundtrip arms. Median horizontal depth-edge magnitude
retention improves from about .007–.010 to .703–.726. This isolates a genuine
representation/raster footprint contribution: even unfused source points lose
detail when rendered with the current fixed physical footprint.

These reference-dependent same-view controls are NOT improved multiview
reconstruction results. On already fused geometry, halving tangent sigma
sharpens edges but loses coverage. Therefore shrinking all existing Gaussians
is not an accepted repair. Next geometry work must coordinate source pixel
footprints with element density and depth discontinuities, then validate
cross-view completeness and sparse references before replacing the frozen
development carrier. No geometry bundle was changed in this iteration.

The authoritative detail report is
geometry_detail_stages_pixel_footprint_20260915.json. The earlier report without
world-normal conversion is superseded and must not supply roundtrip claims.

## Next bounded work

- Preserve MoGe detail through boundary-aware footprint/density construction;
  regenerate correspondence caches only for geometry that passes source gates.
- Build object-level source episodes for text, image and signed prompts,
  preserving unknowns and cross-view identity; add unseen-query validation.
- Train the same latent/matching interface on actual membership and mask
  retrieval, adding optional native teacher relation/boundary losses.
- Compare no-RADIO, RADIO-only and mixed teachers at equal splits/capacity;
  separate gains from teacher choice, geometry and retrieval loss.

Verification: 458 tests passed, four existing dependency/sparse-tensor warnings;
actual geometry-stage diagnostics, native extraction, query optimization and
canonical finetuner runs completed separately. No new benchmark is promoted.
