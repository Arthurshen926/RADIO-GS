# ScanNet-PFPR-Small v2

ScanNet-PFPR-Small v2 evaluates a pose-free RGB patch as a 3-D location query
inside a known ScanNet scene:

```text
held-out depth-aligned RGB patch + canonical field -> ranked public 3-D points
```

It retains v1's label-free point-retrieval target, public 5 cm candidate
domain, fixed 10 cm NMS, and evaluator-private 3-D anchor.  It changes only a
previously inconsistent query-image raster, so it is a new immutable release
rather than a relabelled v1 score.

## Corrected query/field observation contract

The full ScanNet field is built from the registered RGB-D raster: RGB is
bilinearly resized to the depth camera resolution before 3DGS, C-RADIO feature
extraction, MPR, and canonical DINO reconstruction.  V1 instead cropped the
raw color image.  For a typical ScanNet stream this compared a `128 x 128`
patch from `1296 x 968` RGB against field descriptors reconstructed from
`640 x 480` observations, creating an approximately two-fold local field-of-
view mismatch.

V2 fixes this before any method inference:

1. select a valid depth pixel in a held-out RGB-D frame;
2. back-project it with evaluator-private depth/pose to form the private 3-D
   anchor;
3. resize only the source RGB image to the corresponding depth-aligned raster,
   exactly as full-`.sens` field materialization does;
4. crop a fixed `128 x 128` RGB patch centered at the original depth pixel;
5. reveal only `scene_id` and that RGB patch to the method.

The query still exposes no pose, depth, source-frame ID, anchor, mask,
instance, or class.  Crop construction reads no annotations.  The manifest
records `query_raster_contract: depth_aligned_rgb_v2`; v1 manifests retain
`native_color_rgb_v1` and are never merged with v2.

## Evaluation

The method produces one score per public 5 cm annotation-mesh point.  The
evaluator applies deterministic 10 cm Euclidean NMS, then reports:

- R@1/R@5/R@10 at 10 cm;
- R@1 at 5/10/20 cm;
- top-1 mean and median Euclidean 3-D error;
- first-correct MRR at 5/10/20 cm;
- query-micro and scene-macro aggregates.

No instance ID, mask IoU, connected component, or target label enters the
primary PFPR result.

## Field source and fidelity variants

The corresponding canonical field excludes the source frame of every v2
query, then selects a query-free full-`.sens` depth-voxel-coverage prefix.
The frozen first source has 240 frames and therefore uses
`canonical-full-observation-mpr-v1`. The next source-fidelity promotion
materializes a distinct 480-frame prefix and uses the separately declared
`canonical-full-observation-mpr-v2`; it preserves the same exclusion digest,
registration, Gaussian readout, public candidates, and metrics. This avoids
the invalid configuration where a 480-view geometry source is quietly reduced
to a 240-view semantic MPR cache. Source manifests store the full sensor
digest and only a hash of the withheld frame set. The public manifest carries
the same one-way per-scene commitment, and the scorer rejects a field unless
its render contract proves an exact commitment match. The scorer requires
meaningful continuous Gaussian support (`>= 0.01`) for public candidates
before evaluator anchors are opened. Results from the two MPR contracts are
separate rows; they are not averaged or cherry-picked by anchor/rank metrics.

The official-native C-RADIO DINO/SAM teacher variant remains a field-side
option: native official adaptor maps are extracted before registration and
their provenance may be required at scoring time.  It does not alter a query
patch, candidate point, NMS radius, or evaluator target.

For the shared official-capability field route, checkpoint selection uses the
same label-free retention rule as the 3-D point-prompt interface: it maximizes
held-out official DINO/SAM fidelity while requiring each capability cosine to
stay within `0.002` of its initialization and the raw MPR probe to stay within
`0.020`.  These fixed field-side limits are not selected from PFPR anchors,
distances, ranks, or scores, and apply identically to every PFPR scene.

The formal field queue uses that native-official route at full registered
ScanNet resolution and performs a fail-closed source audit before any GPU
stage: each scene's field_source_contract.json must contain exactly the same
one-way held-out-frame digest as manifest.public.json.  Thus a field cannot
silently be built from query source frames even if a later scorer would
otherwise reject it.

## Construction

```bash
bash radio_gs/scripts/run_repo_python.sh \
  -m radio_gs.benchmarks.scannet_pfpr.build_benchmark \
  --frames-root /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/frames_dense20 \
  --annotations-root /mnt/pool/sqy/3d_understanding/segmentation_benchmarks/ScanNet-PFIR-Small/annotations \
  --source-pfir-public-manifest output/scannet_pfir_small_v1/test_v1_final/manifest.public.json \
  --output-dir output/scannet_pfpr_small_v2/test_v2_r1 \
  --benchmark-version scannet-pfpr-small-v2 \
  --query-raster-contract depth_aligned_rgb_v2
```

The v2 full-observation field source must be materialized from the v2 method
and evaluator manifests, so its withheld-frame digest is specific to this
release.  Results from v1 and v2 are separate tables; neither is tuned using
the private anchors.

For an existing materialized v2 source,
run_pfpr_v2_full_sens_field_queue.sh builds a disjoint field shard and
run_pfpr_v2_full_sens_query_queue.sh scores completed scene shards against
the immutable test_v2_r1 release.

For the 480-view MPR promotion, use
`run_pfpr_v2_mpr480_field_queue.sh` with one scene and a new output root. It
waits for the separately materialized 480-view source, verifies the public
held-out-frame digest and source-size requirement, then invokes the same
field pipeline with `canonical-full-observation-mpr-v2`. It rejects an old
240-view raw MPR cache on resume. The ordinary query scorer and frozen point
retrieval metrics remain unchanged.

When a v2 field fails the fixed public-candidate support gate, the next
source-fidelity rung is a separately materialized 960-view prefix and
`run_pfpr_v2_mpr960_field_queue.sh`.  It requires the same held-out-frame
digest, rejects non-v3 raw-MPR caches, and fixes the geometry bootstrap at 240
coverage-ranked RGB-D frames / 300k Gaussians before lifting
`canonical-full-observation-mpr-v3`.  This is a field-side construction
budget, not an anchor-, rank-, or score-conditioned adjustment; v2 and v3
remain distinct report rows.

Before the v3 queue creates any raw/DINO/SAM MPR cache, it audits all frozen
Gaussians against the release's **public geometry-only** 5 cm candidate domain.
The audit opens neither query crops nor evaluator-private anchors and requires
continuous support of at least `0.95` at the same 5 cm cell / `0.01` support
definition used by the scorer.  A failure rejects the construction rather than
lowering the gate or emitting a partial retrieval score.
