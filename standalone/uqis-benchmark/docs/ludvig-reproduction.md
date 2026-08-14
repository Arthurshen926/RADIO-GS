# LUDVIG reproduction and UQIS adapter

## Claim boundary

The implementation is a benchmark-local LUDVIG comparator. It reuses frozen
upstream feature extraction/uplift behavior where applicable, but it is not an
official paper reproduction and its mesh AP/IoU cannot be compared directly to
the historical LERF-OVS localization/segmentation percentages.

## Environment used

- two RTX 3090 24GB GPUs; UQIS query runs used physical GPU1;
- PyTorch 2.4 / CUDA 11.8 environment with the patched Gaussian rasterizer;
- vendored DINOv2 ViT-g/14 register checkpoint;
- OpenCLIP ViT-B/16 LAION checkpoint;
- bound upstream LUDVIG source hashes and driver shim.

Checkpoints and upstream datasets are not included in the standalone package.

## Per-scene fields

LUDVIG does not expose one universal multimodal field, so UQIS honestly builds
two persistent fields for every scene:

1. **OpenCLIP 512-D field** for text: render legal mapping RGB views, extract
   tiled OpenCLIP features, uplift with the Gaussian inverse-render weights,
   normalize, and retain the full Gaussian carrier.
2. **DINOv2/PCA40 field** for image and point prompts: extract DINO sliding-window
   tokens, apply frozen PCA40, inverse-render uplift and upstream integer
   pruning to 600,000 Gaussians.

Across nine scenes the controlled v0.1 run charged 18 persistent fields and
37,209,952,854 bytes (34.66 GiB). Text diffusion depends on both fields.

## Query adapters

- **Image:** encode the normalized 224×224 crop with DINO, apply PCA40, compare
  against the DINO field, then continuous K64 Gaussian-to-mesh readout.
- **Point-2D:** render only method field state into the authorized camera,
  raster-adjoint the click into field relevance, and read out the mesh. No
  captured query RGB or SAM is allowed.
- **Point-3D:** compile the world point directly against the DINO Gaussian field
  and use the same mesh readout.
- **Text:** compute LERF hardest-negative OpenCLIP relevance against `object`,
  `things`, `stuff`, and `texture`; optionally align it through frozen source
  indices to the DINO carrier and run graph diffusion.

## Text graph diffusion

The frozen candidate configuration uses a spatial 64-NN graph, DINO feature
edge weights, 20 iterations, feature bandwidth 0.5, regularizer bandwidth 2.0
and seed quantile 0.999. It consumes neither evaluator labels nor RGB/SAM at
query time. The current implementation rebuilds topology per isolated query;
the next release should persist one content-bound topology per scene and charge
its bytes.

## Reproduction sequence

The exact RADIO-GS scripts use this order:

1. stage mapping observations after query-union frame exclusion;
2. train/validate one legal 3DGS geometry per scene;
3. build DINO Phase-B features and PCA, then uplift/prune the DINO field;
4. build the independent OpenCLIP field;
5. create the method field inventory and storage receipt;
6. stage one opaque workspace per query;
7. run every query in a fresh process, preferably grouped by public scene ID;
8. copy and seal all mesh arrays before evaluator-private data opens;
9. run the evaluator once and consume the ledger.

Representative command names in the parent repository are
`stage_uqis_mapping_observations.py`, `run_uqis_geometry_queue.py`,
`run_uqis_dino_field_queue.py`, `run_uqis_clip_field.py`,
`run_uqis_query_queue.py`, and `seal_uqis_method_execution.py`.

## Important differences from LERF-OVS

LERF localization accepts a peak inside any target box and its labeled frames
overlap the scene imagery. UQIS excludes query frames from mapping and evaluates
all official mesh vertices. LUDVIG's stronger historical segmentation path also
uses DINO diffusion plus SAM; strict UQIS prompt/text execution forbids that
query-time RGB/SAM route. The resulting UQIS numbers are expected to be lower
and answer a different question.

## Candidate outcome

On the 31-target Core cohort, the benchmark-local LUDVIG system reached text
AP 0.23304 and four-modality UQ-Rank 0.40004. Direct CLIP on the same
expressions reached text AP 0.16365, so graph diffusion added 0.06939 AP. On
the separate 36-query Relational Text Challenge, direct CLIP reached AP
0.09705 and diffusion reached 0.16750 (+0.07045, 72.6% relative).

These are non-formal candidate results. They validate the complete query,
field, mesh-readout, sealing and evaluator path, but they do not become an
official LUDVIG reproduction or a formal UQIS leaderboard row.
