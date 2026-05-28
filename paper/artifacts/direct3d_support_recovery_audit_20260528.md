# Direct3D Support Recovery Audit

- Baseline: `output/radio_gs/lerf_direct3d_deployed_opacity_gate_masks_20260528`
- Promoted variant: `component_guard065`
- Tag: `thr0p35`

## Macro Metrics

| Variant | mIoU | Acc@0.25 | Boundary-F | Trimap IoU |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.4836 | 0.6426 | 0.5983 | 0.3103 |
| none | 0.4430 | 0.6616 | 0.5880 | 0.2821 |
| rgb_grabcut | 0.4675 | 0.6765 | 0.5869 | 0.3299 |
| component_guard065 | 0.4806 | 0.6688 | 0.5997 | 0.3242 |
| component_guard055 | 0.4803 | 0.6496 | 0.5976 | 0.3160 |

## Scene Metrics

### baseline

| Scene | mIoU | Acc@0.25 | Boundary-F |
| --- | ---: | ---: | ---: |
| figurines | 0.5147 | 0.6607 | 0.6475 |
| ramen | 0.5726 | 0.7887 | 0.7258 |
| teatime | 0.5482 | 0.7119 | 0.6910 |
| waldo_kitchen | 0.2988 | 0.4091 | 0.3288 |

### none

| Scene | mIoU | Acc@0.25 | Boundary-F |
| --- | ---: | ---: | ---: |
| figurines | 0.4202 | 0.6607 | 0.5953 |
| ramen | 0.5698 | 0.8310 | 0.7514 |
| teatime | 0.5151 | 0.7458 | 0.6896 |
| waldo_kitchen | 0.2669 | 0.4091 | 0.3156 |

### rgb_grabcut

| Scene | mIoU | Acc@0.25 | Boundary-F |
| --- | ---: | ---: | ---: |
| figurines | 0.4695 | 0.6607 | 0.6077 |
| ramen | 0.5801 | 0.8451 | 0.7545 |
| teatime | 0.5391 | 0.7458 | 0.6853 |
| waldo_kitchen | 0.2811 | 0.4545 | 0.3002 |

### component_guard065

| Scene | mIoU | Acc@0.25 | Boundary-F |
| --- | ---: | ---: | ---: |
| figurines | 0.4871 | 0.6607 | 0.6273 |
| ramen | 0.5802 | 0.8310 | 0.7471 |
| teatime | 0.5529 | 0.7288 | 0.7051 |
| waldo_kitchen | 0.3021 | 0.4545 | 0.3195 |

### component_guard055

| Scene | mIoU | Acc@0.25 | Boundary-F |
| --- | ---: | ---: | ---: |
| figurines | 0.5024 | 0.6607 | 0.6442 |
| ramen | 0.5744 | 0.8169 | 0.7344 |
| teatime | 0.5480 | 0.7119 | 0.6910 |
| waldo_kitchen | 0.2964 | 0.4091 | 0.3209 |

## Acc@0.25 Crossings

| Variant | Scene | Frame | Query | Before | After | Delta |
| --- | --- | --- | --- | ---: | ---: | ---: |
| none | ramen | frame_00024 | `wavy noodles` | 0.0000 | 0.4490 | 0.4490 |
| none | ramen | frame_00060 | `wavy noodles` | 0.0000 | 0.4267 | 0.4267 |
| none | ramen | frame_00065 | `wavy noodles` | 0.0000 | 0.3645 | 0.3645 |
| none | waldo_kitchen | frame_00140 | `spoon` | 0.0683 | 0.3580 | 0.2896 |
| none | teatime | frame_00043 | `stuffed bear` | 0.1792 | 0.2560 | 0.0768 |
| none | teatime | frame_00140 | `stuffed bear` | 0.2331 | 0.2512 | 0.0181 |
| none | waldo_kitchen | frame_00089 | `ketchup` | 0.3896 | 0.2324 | -0.1572 |
| rgb_grabcut | ramen | frame_00024 | `wavy noodles` | 0.0000 | 0.4306 | 0.4306 |
| rgb_grabcut | ramen | frame_00060 | `wavy noodles` | 0.0000 | 0.4035 | 0.4035 |
| rgb_grabcut | ramen | frame_00065 | `wavy noodles` | 0.0000 | 0.3475 | 0.3475 |
| rgb_grabcut | waldo_kitchen | frame_00140 | `spoon` | 0.0683 | 0.3519 | 0.2836 |
| rgb_grabcut | teatime | frame_00043 | `stuffed bear` | 0.1792 | 0.2910 | 0.1118 |
| rgb_grabcut | ramen | frame_00081 | `napkin` | 0.2295 | 0.2707 | 0.0412 |
| rgb_grabcut | teatime | frame_00140 | `stuffed bear` | 0.2331 | 0.2649 | 0.0317 |
| component_guard065 | ramen | frame_00024 | `wavy noodles` | 0.0000 | 0.4306 | 0.4306 |
| component_guard065 | ramen | frame_00060 | `wavy noodles` | 0.0000 | 0.4035 | 0.4035 |
| component_guard065 | ramen | frame_00065 | `wavy noodles` | 0.0000 | 0.3475 | 0.3475 |
| component_guard065 | waldo_kitchen | frame_00140 | `spoon` | 0.0683 | 0.3519 | 0.2836 |
| component_guard065 | teatime | frame_00043 | `stuffed bear` | 0.1792 | 0.2910 | 0.1118 |
| component_guard055 | ramen | frame_00024 | `wavy noodles` | 0.0000 | 0.4306 | 0.4306 |
| component_guard055 | ramen | frame_00060 | `wavy noodles` | 0.0000 | 0.4035 | 0.4035 |

Conclusion: `rgb_grabcut_component_guard` is a GT-free support-preserving cleanup. It targets cases where the projected support is multi-component and avoids forcing a largest-component decision unless one component dominates the refined mask.
