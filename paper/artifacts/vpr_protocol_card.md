# VPR Protocol Card

This card documents the paper-facing LERF-OVS direct 3D object-selection
protocol.

| Item | Setting |
|---|---|
| Query target | 3D Gaussian primitives; rendering is only used after selection for mask evaluation |
| Registration feature | Rendered RADIO-compatible features after VFA, projected by frozen SigLIP2 summary head |
| Registration views | Posed RGB/render views selected from all COLMAP poses; main row uses 128 evenly spaced views |
| Leakage audit | Train-view-only 24-view row is reported separately and remains above raw center readout |
| Visibility hygiene | Depth and alpha checks; no LERF masks or text labels are used during registration |
| Text scoring | Frozen SigLIP2 text/query embeddings with scene-wise softmax scoring |
| Selection | Fixed global softmax-score threshold `thr0p25` with 0.5% floor and 1.8% cap for the paper-facing row; mean+std and fixed-ratio sweeps are diagnostic only |
| Context aggregation | GT-free voxel-max score aggregation, resolution 80, blend 0.50 |
| Persistent storage | No extra trained state; optional registered primitive/voxel caches are reported separately |
| Metrics | OpenGaussian-style mIoU and Acc@0.25 after rendering selected primitives |

Conclusion: VPR is a protocol-aligned primitive-level readout from rendered
features, not a GT-mask registration procedure and not a separate teacher rerun.
