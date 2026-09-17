# Matched geometry visual review

Generated four scenes, each with the first two declared geometry-heldout
frames, through scripts/audits/visualize_frozen_geometry.py.
Output: paper/artifacts/geometry_visuals_corrected_20260915.

Each 2D row uses the same camera, metric clipping planes, and shared reference
depth scale for historical and rebuilt geometry. Columns: MoGe reference,
historical depth/alpha/relative error, rebuilt depth/alpha/relative error.
Depth/error holes are white. Alpha white means opaque; black means missing.
Error is clipped for display at 0.5, not for quantitative evaluation.
No RGB textures, semantic predictions or generated imagery are used.

The 3D PNGs show at most 30,000 deterministically sampled Gaussian centres per
model, colored by world Z. They are not a mesh or Gaussian footprint display.
Both arms use the same robust box derived from the rebuilt sample; excluded
sample counts are shown. Interactive HTML retains every sampled centre,
including points outside that static box, and embeds its JavaScript offline.
Point density is not directly comparable because original element counts differ.

## Construction being inspected

Source MoGe point-map Z and normals -> COLMAP scale and accepted per-view
affine depth calibration -> calibrated camera rays and inverse-transpose normal
alignment -> trusted-neighbor scale recovery for rejected mapping views ->
source-only depth-supported voxel fusion -> surface-oriented thin Gaussians.
Heldout source frames do not enter scale recovery or fusion.
Current resolution: width 480; voxel size 1 cm; Gaussian tangent sigma 7.5 mm,
normal sigma 1 mm; opacity 0.8. These are geometric carrier design choices,
not learned confidence. The final residual joint-scale fit was not adopted.

## Observations

Waldo frames 1 and 9: historical depth is near zero across most of the view,
despite opaque alpha. Rebuilt depth recovers scene structure and removes the
dominant near-camera occlusion failure. There are still holes and boundary
errors.

Figurines frames 1 and 13: the rebuilt geometry has missing background/edge
regions. Some small-object depth relief is weaker than both the MoGe reference
and the historical render, particularly frame 13. This is evidence against
declaring full detail preservation from mean depth improvement. The
visualization does not isolate whether loss occurs during scale calibration,
view filtering, voxel fusion or Gaussian compositing.

The sampled rebuilt 3D centres appear more concentrated on coherent surfaces
than historical centres, but this is not a quantitative completeness or
surface-accuracy result. It cannot override the 2D detail failures.

## Next geometry check

On fixed source development views, compare calibrated per-view depth,
pre-voxel points, fused points and final raster depth. Stratify errors by depth
edges, thin/small structures and smooth surfaces. Record missing support and
edge displacement, not only global AbsRel. Diagnose the stage losing detail
before changing voxel width or covariance. Keep this local geometry work
separate from semantic gradients and do not tune on final heldout views.
