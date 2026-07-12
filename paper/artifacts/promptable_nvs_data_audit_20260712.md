# NVOS / SPIn-NeRF data asset audit (2026-07-12)

This report records the files that are physically present under
`/mnt/pool/sqy/3d_understanding/segmentation_benchmarks`. Archive sizes and
SHA-256 values were recomputed from the files on disk; extracted-file and
frame counts were recomputed with `find`, excluding AppleDouble metadata under
`__MACOSX` where stated. Existing download manifests were not treated as proof
of file integrity.

## Outcome

- **NVOS: complete.** All eight official tasks have a target mask/RGB copy, a
  distinct reference RGB, and positive, negative, and visualization scribble
  assets. Every reference and target basename maps to the required NeX LLFF
  undistorted source images.
- **SPIn-NeRF segmentation annotations: complete.** The canonical ten scenes
  contain 461 GT masks, 461 pseudo visualizations, and 461 cutouts.
- **SPIn-NeRF upstream RGB cohort: incomplete (9/10 scenes).** All 423
  annotated frames outside Fork map to source RGB. Fork has 38 annotations but
  zero source RGB frames because its upstream archive is unavailable.
- **Formal SPIn-NeRF 10-scene evaluation is fail-closed.** No 10-scene SPIn
  manifest or main result may be emitted until the original Fork RGB/camera
  asset is supplied and verified. A nine-scene diagnostic, if run, must be
  labelled as such and is not comparable to a published ten-scene macro.

## NVOS

### Archive and extracted payload

| Asset | Bytes | Recomputed SHA-256 |
|---|---:|---|
| `NVOS/nvos-data.zip` | 114,548,360 | `3e09d7caf5ca49672f80afe419225c9ef13c31803228ac201fd002dc7cf197fa` |

The extracted official annotation tree contains 49 files: one README; eight
target masks; eight target-view RGB copies; eight reference-view RGB images;
and eight positive, eight negative, and eight visualization scribbles. The
NeX `llff_undistorted` payload contains 740 upstream files totalling
3,523,116,397 bytes, excluding the locally generated 157,591-byte
`download_manifest.json`.

| NeX base scene | Undistorted RGB | Distorted RGB | NVOS use |
|---|---:|---:|---|
| fern | 20 | 20 | `fern` |
| flower | 34 | 34 | `flower` |
| fortress | 42 | 42 | `fortress` |
| horns | 62 | 62 | `horns_center`, `horns_left` |
| leaves | 26 | 26 | `leaves` |
| orchids | 25 | 25 | `orchids` |
| room | 41 | 41 | downloaded upstream; not an NVOS task |
| trex | 55 | 55 | `trex` |

The strict eight-task manifest is
`manifests/nvos_strict_unseen_v1.json`. Its file SHA-256 is
`bafc48ce30a0a637f5ea4d81a196ea240f80c153c41a3e257b6a2fd45fa3f2ea`
and its embedded protocol hash is
`d21b52d3c2155f82d63c18a6c9c4a56eb25c42f09c10306aa924135317cadaf9`.
The manifest records each official target frame in
`excluded_training_frame_ids`; target RGB and target masks therefore remain
ineligible for photometric/feature-field training or query-time selection.
The geometry queue additionally removes every sparse point whose COLMAP track
contains the target image (11.3%--30.8% of source points depending on the
scene). It retains upstream camera poses/intrinsics, which may have been jointly
estimated from the full capture and are explicitly recorded as a shared
calibration exception.

## SPIn-NeRF

### Official archives

| Asset | Bytes | Recomputed SHA-256 | Role |
|---|---:|---|---|
| `official/SPIn-NeRF/Multiview-Segmentation-Data.zip` | 88,854,757 | `da177c7b58a867eba2df8e293e98f769cdc4205a991a8b342f366d92dfc3a536` | multiview segmentation annotations used here |
| `official/SPIn-NeRF/spinnerf-dataset.zip` | 6,015,320,134 | `93ea6d36bc517f0f27da8f8ddd874575b6b9decf6d1fa8895e02b96e8652f189` | separate object-removal/perceptual-inpainting dataset |
| `official/SPIn-NeRF/statue.zip` | 34,872,589 | `397b7a2e9d952418dc73e1f18ba341f6dc030bf26884dca4eda2dc7f0c80be5f` | separate supplementary scene asset |

`spinnerf-dataset.zip` must not be substituted for the multiview segmentation
benchmark. The official README describes it as ten removal/inpainting scenes,
each with clean ground-truth captures, object-present training views, and
removal masks; the segmentation table instead uses the scenes and annotations
in `Multiview-Segmentation-Data.zip`.

### Annotation inventory

| Scene | GT masks | Pseudo | Cutout |
|---|---:|---:|---:|
| orchids | 24 | 24 | 24 |
| leaves | 26 | 26 | 26 |
| fern | 20 | 20 | 20 |
| room | 41 | 41 | 41 |
| horns | 62 | 62 | 62 |
| fortress | 42 | 42 | 42 |
| fork | 38 | 38 | 38 |
| pinecone | 99 | 99 | 99 |
| truck | 65 | 65 | 65 |
| lego | 44 | 44 | 44 |
| **Total** | **461** | **461** | **461** |

The canonical extracted annotation payload contains 1,394 files and
94,247,754 bytes: 1,383 PNGs, ten `annotations.json` files, and one README.
The archive also expands 1,394 AppleDouble metadata files (649,519 bytes)
under `__MACOSX`; these are not benchmark frames.

### Upstream RGB provenance and coverage

| Source archive | Bytes | Recomputed SHA-256 | Covered scenes |
|---|---:|---|---|
| `source_images/llff_google_drive/nerf_llff_data.zip` | 1,780,545,599 | `b8be42c77ce345e647812cb69d1f92d2a85159f2464847e99458e53d13cb1d96` | orchids, leaves, fern, room, horns, fortress |
| `source_images/nerf_real_360/nerf_real_360.zip` | 1,653,956,363 | `e5996aa08cf9a22c28adc21d9321ca302bd737ad71d9959dcdab825d6981b0ad` | pinecone |
| `source_images/tandt/tandt_db.zip` | 682,628,995 | `816e62f22a161abbfe841d2a6b10cdf036e297c9fa289b3bfeee9c6ec526d7e1` | truck |
| `source_images/lego_real_night_radial/lego_real_night_radial.tar.gz` | 586,506,544 | `fe97e2698d88525f9937f37bbd05ad01c277f602ed932e2ccee778ff4519e06b` | lego |

The four present source archives total 4,703,637,501 bytes. Coverage was
checked by canonical annotation basename/index mapping, including the official
Truck split-prefix removal and the missing `orchids/image014` annotation.

| Scene | Source RGB frames present | Annotated frames | Mapped | Status |
|---|---:|---:|---:|---|
| orchids | 25 | 24 | 24 | complete; annotation intentionally skips `image014` |
| leaves | 26 | 26 | 26 | complete |
| fern | 20 | 20 | 20 | complete |
| room | 41 | 41 | 41 | complete |
| horns | 62 | 62 | 62 | complete |
| fortress | 42 | 42 | 42 | complete |
| fork | 0 | 38 | 0 | **blocked: upstream source unavailable** |
| pinecone | 99 | 99 | 99 | complete |
| truck | 251 | 65 | 65 | complete |
| lego | 102 | 44 | 44 | complete |
| **Total** | **668** | **461** | **423** | **9/10 scenes** |

A second preflight bound those RGB files to the upstream COLMAP cameras. The
four indexed LLFF exports (orchids, leaves, fern, fortress) use strict
`imageNNN` canonical indices into lexicographically ordered COLMAP cameras;
room, horns, pinecone, and truck use exact basename stems; Lego removes exactly
one official `0_`/`1_` split prefix before an exact match. All nine available
scenes passed one-to-one mapping. Fuzzy/nearest-name matching is forbidden, and
the generated per-scene map records both names, the rule, RGB resolution, and
intrinsics scaling. Truck's source RGB is 979×546 while its mask is 980×546;
the evaluator therefore applies the frozen nearest-neighbor prediction-to-GT
resize rather than silently cropping either asset.

The expected Fork archive is `source_images/fork/fork.zip` (recorded upstream
size 481,028,506 bytes). It is absent, `source_images/fork/` is empty, and the
download audit records the official upstream as unavailable with no verified
mirror. The 38 Fork `_cutout.png` files are 504×378 RGBA annotation-derived
cutouts, while the base `.png` files are grayscale masks. They are neither
full source RGB nor camera/calibration data. Using a cutout as RGB would both
remove the scene background and leak the target segmentation, so it is
forbidden.

## Fail-closed eligibility rule

At audit time `manifests/` contains only `nvos_strict_unseen_v1.json`; there is
no SPIn-NeRF 10-scene manifest. The required release gate is:

1. obtain and checksum the original Fork RGB/camera bundle;
2. map all 38 Fork annotations to unique source cameras;
3. build and validate a manifest containing exactly all ten scenes and all 461
   masks; and
4. only then permit a formal ten-scene macro result.

Until all four conditions pass, any SPIn run is diagnostic-only and must not
occupy a ten-scene main-table row.
