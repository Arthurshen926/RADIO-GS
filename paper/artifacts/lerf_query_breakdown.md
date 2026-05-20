# LERF Query Breakdown

- selection: `thr0p25`
- queries: 208
- object-weighted mIoU: 0.5274
- object-weighted Acc@0.25: 0.7260
- scene-mean mIoU: 0.4801
- scene-mean Acc@0.25: 0.6760
- caveat: Object-weighted metrics are diagnostic query-level aggregates. Scene-mean metrics match the paper-facing four-scene aggregate. Label groups are deterministic keyword diagnostics for appendix analysis; only footprint bins are purely geometric.

## Scene Breakdown

| Group | Count | mIoU | Acc@0.25 | Acc@0.5 | Mean GT px | Mean selected Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| figurines | 56 | 0.5309 | 0.7857 | 0.6250 | 9930.1607 | 1283.3929 |
| ramen | 71 | 0.5805 | 0.7465 | 0.6901 | 21558.7183 | 4943.5634 |
| teatime | 59 | 0.5662 | 0.7627 | 0.6780 | 24271.5593 | 6581.8475 |
| waldo_kitchen | 22 | 0.2429 | 0.4091 | 0.1818 | 68027.3182 | 7967.2273 |

## Footprint Bins

| Group | Count | mIoU | Acc@0.25 | Acc@0.5 | Mean GT px | Mean selected Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| tiny | 85 | 0.5232 | 0.7059 | 0.6118 | 3840.9059 | 3778.4000 |
| small | 88 | 0.6017 | 0.8523 | 0.7500 | 15289.9205 | 4945.4773 |
| medium | 30 | 0.3836 | 0.5333 | 0.3333 | 65932.1667 | 6058.3000 |
| large | 5 | 0.1538 | 0.0000 | 0.0000 | 273085.2000 | 9671.2000 |

## Label Groups

| Group | Count | mIoU | Acc@0.25 | Acc@0.5 | Mean GT px | Mean selected Gaussians |
|---|---:|---:|---:|---:|---:|---:|
| container_or_part | 54 | 0.4503 | 0.6111 | 0.5000 | 42409.0185 | 5902.9074 |
| multi_instance_likely | 26 | 0.7120 | 1.0000 | 0.9231 | 8772.0385 | 6086.6538 |
| other | 100 | 0.5195 | 0.7200 | 0.5800 | 18442.4600 | 3355.1000 |
| reflective_or_transparent | 12 | 0.4799 | 0.6667 | 0.5833 | 92756.1667 | 9368.8333 |
| texture_like | 36 | 0.6198 | 0.8333 | 0.7778 | 17370.7778 | 6557.9722 |
