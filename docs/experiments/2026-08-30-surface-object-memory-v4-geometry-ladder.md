# Surface-Aligned Object-Centric Memory v4: first geometry ladder

## Decision boundary

This round changes only carrier geometry and registration.  It does not train
features, object tokens, query adapters, or posterior calibration.  Historical
SUGM-v3 method code is not imported.  The retained benchmark rows are sealed in
`paper/artifacts/v4_retained_baselines_20260830.json` and are comparators, not
inputs to the geometry experiment.

Milestone 1 is now complete on the preregistered three-scene development
cohort.  ScanNet, LERF Figurines, and LERF Ramen all pass with one shared
projection configuration (`maximum_splat_radius=1`, surface band 1.5 voxels,
at most eight contributors).  Only the oracle object-codebook stage is opened;
learned codebook, query encoder, and compression remain separately gated.

An earlier availability check was wrong: it relied on stale README and open
issue text.  The default Microsoft repository branch contains the MoGe-3 model
implementation, and `Ruicheng/moge-3-vitl` and `Ruicheng/moge-3-vitg` publish
pretrained MoGe-3 checkpoints.  The availability receipt has been corrected;
there is no longer an external release blocker.

## Implemented contract

- backend-independent `SurfaceCarrier` projection, lift, render, and adjacency;
- Gaussian exact-renderer transport baseline;
- triangle-mesh raycast oracle;
- sparse surface voxel/surfel z-buffer;
- positive/negative/unknown evidence and multi-view sufficient statistics;
- constrained affine depth calibration and confidence-weighted sparse fusion;
- hash-bound geometry and method receipts;
- static rejection of v3 imports and concrete scene identities in v4 source.

The sparse carrier retains only contributors in a front-surface band of 1.5
voxels and caps each pixel at eight deterministic contributors.  The initial
one-element z-buffer was rejected because it improved same-view metrics but
regressed tracked cross-view transfer.

## Coordinate-convention bug found and repaired

The prepared ScanNet `transforms.json` stores NeRF/OpenGL camera axes.  The
first mesh loader incorrectly treated those matrices as OpenCV poses.  A visual
mesh-RGB audit exposed the error: first-frame raycast coverage was about 31%.
Applying the same OpenGL-to-OpenCV conversion used by the training dataset,
`diag(1,-1,-1,1)`, raised coverage to 98.57% and aligned the rendered mesh RGB
with the source image (support-region PSNR 17.96 dB).  All pre-fix ladder
numbers were overwritten and are not valid evidence.  A regression test now
seals the conversion.

## ScanNet mesh-label oracle result

The diagnostic explicitly opens mesh vertex labels and records that access in
its receipt.  Labels are used only to provide shared cross-view correspondence;
they are not persisted as method state.

| Carrier | same-view soft mIoU | tracked cross-view soft mIoU | boundary leakage | element purity | oracle-surface coverage | effective contributors |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian exact renderer | 0.62363 | 0.43729 | 0.35075 | 0.87864 | 1.00000 | 24.2208 |
| Sparse surface, 4 cm | 0.78225 | 0.45461 | 0.22323 | 0.94768 | 0.94828 | 2.2420 |
| Mesh oracle | 0.94332 | 0.60167 | 0.04155 | 0.98414 | 1.00000 | 2.3434 |

Relative to Gaussian transport, the mesh oracle improves all four primary
metrics.  The sparse surface improves same-view round-trip by 0.15863, tracked
cross-view transfer by 0.01733, purity by 0.06904, and reduces leakage by
0.12752.  Coverage decreases by 0.05172 and is reported rather than used to
mask another regression.

Depth is unavailable in the Gaussian transport shards.  Against mesh raycast
depth, the sparse carrier has mean/median absolute residual 0.07136/0.02129 m.
Its mean unsigned normal cosine is 0.89854 with median angular error 5.12
degrees; the mesh self-oracle is exactly 1.0/0 degrees.

## Source-only SAM result

This second ladder uses 82 sealed query-independent SAM proposals from 16
source views.  It opens no benchmark image, mask, label, query text, or target
RGB.  Because proposals are not tracked, cross-view best-proposal overlap is a
non-gating diagnostic.  Purity is evaluated only over 153 exactly pixel-disjoint
proposal pairs; parent/part overlaps are excluded by construction.

| Carrier | same-view soft IoU | same-view leakage | mutually-exclusive purity | mask coverage | effective contributors |
|---|---:|---:|---:|---:|---:|
| Gaussian exact renderer | 0.57361 | 0.29523 | 0.99523 | 1.00000 | 24.2208 |
| Sparse surface, 4 cm | 0.71220 | 0.15232 | 0.99895 | 0.92933 | 2.2420 |
| Mesh surface | 0.84544 | 0.08887 | 0.99952 | 0.99819 | 2.3434 |

The source-only direction agrees with the oracle on round-trip, leakage,
mutually-exclusive purity, and registration ambiguity.  The untracked
best-target-proposal overlap is slightly lower and remains explicitly
non-gating because it does not establish object identity across views.

## LERF MoGe-3 inference and fusion

The revision-pinned official MoGe-3 ViT-L checkpoint was run on all 120 sealed
source frames in each LERF scene.  Both scenes have 100% valid point-map
coverage.  COLMAP calibration uses only sparse points whose complete tracks
remain inside the source split.  Rejected local affine fits fall back to the
single robust global scale rather than dropping the view.

| Scene | strict source-only COLMAP points | locally calibrated / global fallback views | fused 4 cm elements | median dispersion |
|---|---:|---:|---:|---:|
| Figurines | 1,104 | 76 / 44 | 84,357 | 1.78 cm |
| Ramen | 16,320 | 49 / 71 | 44,995 | 1.82 cm |

The model checkpoint SHA-256 is
`9b41b7b9f65ad80aab7ad686f5e9cc0d1fd33f1964022618dfbcd52fc1fb7925`.

## Shared projection-footprint correction

The first LERF run inherited a maximum three-pixel surfel splat from the
ScanNet prototype.  It passed Figurines but failed all three primary directions
on Ramen.  A bounded shared ablation changed only this footprint and used the
same value for all scenes:

| max radius | Figurines round-trip delta | Ramen round-trip delta | Figurines gate | Ramen gate |
|---:|---:|---:|:---:|:---:|
| 3 | +0.09076 | -0.10959 | pass | fail |
| 2 | +0.13083 | -0.03700 | pass | fail |
| 1 | +0.24930 | +0.09507 | pass | pass |

The monotone Ramen degradation and simultaneous Figurines improvement identify
over-wide splatting on the coarse feature raster as the error.  Radius one is
not a relaxed gate: coverage remains non-compensatory and purity must still not
regress.  The same radius-one setting was rerun on ScanNet.  Its label oracle
still improves round-trip by 0.15019, tracked cross-view transfer by 0.00403,
leakage by 0.12422, and purity by 0.06923.  The source-only ScanNet arm improves
round-trip, leakage, and purity; its untracked best-proposal transfer remains a
non-gating negative diagnostic (-0.00569).

Final LERF source-only values are:

| Scene / carrier | round-trip IoU | leakage | exclusive purity | coverage | effective contributors |
|---|---:|---:|---:|---:|---:|
| Figurines Gaussian | 0.36052 | 0.51150 | 0.99005 | 0.99655 | 26.8425 |
| Figurines sparse surface | 0.60983 | 0.25011 | 0.99952 | 0.97488 | 2.1112 |
| Ramen Gaussian | 0.61592 | 0.25203 | 0.99932 | 0.99794 | 66.1851 |
| Ramen sparse surface | 0.71099 | 0.18324 | 0.99937 | 0.99560 | 1.4020 |

## Completed geometry gate

The aggregate report evaluates exactly three expected scenes.  Both LERF arms,
the ScanNet sparse arm, and the ScanNet mesh-oracle stop rule pass.  Milestone 1
is complete.  This authorizes only an oracle object codebook; it does not
authorize learned masks, text/image query, or compression.

## First oracle object-codebook result

The v4 codebook stores two token IDs and weights per surface element plus
explicit unknown mass.  An opt-in ScanNet diagnostic used direct 3-D instance
IDs only as oracle association keys.  With the exact oracle memberships, the
same element posterior obtains 0.67296 held-out 2-D soft mIoU and 1.0 3-D soft
mIoU/purity, so the carrier and sparse top-2 representation have sufficient
capacity.

A stricter source-lifted arm remains below gate: using 12 mapping and four
held-out views obtains 0.46701 2-D and 0.46016 3-D soft mIoU, despite 0.94878
purity and 0.86855 top-1 accuracy.  Only 41.03% of elements receive known
membership.  This localizes the next failure to object write coverage/completion,
not token capacity or token merging.  Learned soft-codebook training therefore
remains blocked until source-mask association can cover unseen object extent
without erasing unknown exterior.

## Gate result and next action

Next actions are restricted to improving source-to-token write coverage and
testing the oracle image/prompt selection path.  Text querying remains closed.
Generic graph propagation, connected components, target RGB, and historical
v3 instance modules remain forbidden.

## Bound artifacts

- `paper/artifacts/v4_scannet_geometry_ladder_oracle_a_20260830.json`
  (`98680aad862bceaef30cfbfd4a7de99b339405cf2d2a84d7766b7c8c6cdd674b`)
- `paper/artifacts/v4_scannet_source_mask_geometry_ladder_a_20260830.json`
  (`111179443c46912461ebeb736be2ea9608640f36a8c71937d4cdde2db0545353`)
- `paper/artifacts/v4_geometry_gate_partial_20260830.json`
  (`5547fe78decb98ab9bd0775f5986779cb42108b0c2cb91cb61b25bbe1b800025`)
- `paper/artifacts/v4_moge3_availability_20260830.json`
  (`4a8af5662f55516f4427cc31ec623e33c76f53960a2bee4311f4d50843f88dd9`)
- `paper/artifacts/v4_scannet_geometry_ladder_oracle_r1_20260830.json`
  (`a7e80df177769d8c64374a170dcdd1b1cbe1c61e6d83599f0d272a6245e8cc44`)
- `paper/artifacts/v4_scannet_source_mask_geometry_ladder_r1_20260830.json`
  (`85a2e64d791af6bb101902c88dfa6c0405bd9ca0281972891daeda3d3c1f2a0a`)
- `paper/artifacts/v4_lerf_figurines_source_mask_geometry_gate_r1_20260830.json`
  (`b85d414ea54dd49c378b0f4359ae8fec994f878662abc96aba006ca00e9fa4d1`)
- `paper/artifacts/v4_lerf_ramen_source_mask_geometry_gate_r1_20260830.json`
  (`86974e34d52121e7e6686361e8ca1583b4a9378579ba2e16c9810d11b3e1c6af`)
- `paper/artifacts/v4_geometry_gate_complete_r1_20260830.json`
  (`a5b3f66a664dddda26896eaf9320d1307a019d082585d245bf4652809dbf2aec`)
- `paper/artifacts/v4_scannet_object_codebook_oracle_gate_a_20260830.json`
  (`0fd6a9eae6b73fc14387321e973571e3468688df3db03ad20d1a8f2d7f2ee42f`)
