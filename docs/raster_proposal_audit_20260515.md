# Raster/Proposal Direct-3D Audit

Date: 2026-05-15.

Protocol note: this is a historical negative-ablation audit. The later
paper-facing direct-3D row supersedes the `mean+2.5std` selector with the fixed
global `thr0p25` selector plus RGB snap; these raster/proposal rows remain
diagnostics and are not the promoted protocol.

This audit implements the remaining expert-requested direct-3D branches and
tests them without using LERF masks for scoring or calibration.

## Implemented Branches

- `--registration_assignment_mode raster_contrib`: uses gsplat rasterizer
  Gaussian-pixel intersections and accumulates rendered SigLIP2 features over
  Gaussian footprints.
- `--registration_assignment_mode raster_dominant`: keeps the strongest
  rasterized Gaussian hit per rendered pixel.
- `--registration_assignment_mode raster_gaussian_top1`: keeps the strongest
  rasterized pixel hit per Gaussian footprint.
- `--selection_refinement proposal_components`: builds a GT-free 3D support
  pool from primitive scores, groups it into connected voxel proposals, and
  selects the top proposal components.

## Figurines Probe Results

All rows use the same mean+2.5std selector with a 0.5% floor, 1.8% cap,
voxel-max score context, SigLIP2 text head, and no GT mask tuning.

| Branch | Views | Weight | Proposal | Registered frac. | mIoU | Acc@0.25 |
|---|---:|---|---|---:|---:|---:|
| Historical center VPR cache | 128 all poses | uniform | no | 0.252 | 0.4829 | 0.8214 |
| Raster all footprint | 4 official | uniform | no | 0.332 | 0.0002 | 0.0000 |
| Raster per-pixel dominant | 128 all poses | alpha-depth | no | 0.084 | 0.0178 | 0.0536 |
| Raster per-Gaussian top1 | 4 official | alpha | no | 0.332 | 0.0004 | 0.0000 |
| Proposal components on promoted VPR scores | cached 128 | uniform | yes | 0.252 | 0.0430 | 0.0357 |

## Conclusion

The true rasterizer-hit registration paths are implemented and auditable, but
they are not competitive with center-based uniform VPR in the current low
feature-resolution readout. The likely failure is feature assignment mismatch:
all-footprint accumulation pollutes primitive features across foreground and
background, while per-pixel dominant assignment is too sparse for LERF object
selection. The proposal branch is also negative because simple connected voxel
support collapses many small object parts into the wrong component. These should
remain negative ablations/future work rather than replacing the paper-facing
VPR row.
