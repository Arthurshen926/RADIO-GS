# Source coverage and cross-carrier evidence, next73

This iteration keeps the frozen Gaussian map / native semantic and structural
memory / typed query / transient element-set architecture. It does not retrain
the compact student, change benchmark thresholds, or use target RGB to predict.
All numbers below are development results; there is no new SOTA claim.

## LERF: a complete four-scene source-coverage result

The next72 64-source semantic reconstruction is now complete on all four scenes.
Geometry, old extent evidence, scoring and confidence transport remain fixed.
The fresh old-cohort control uses regenerated teachers, so both it and the
retained mainline are reported.

| Four-scene macro | 2D IoU | AP | Selected-subset projection IoU |
|---|---:|---:|---:|
| Retained next68 | 64.882 | 81.021 | 51.835 |
| Regenerated old teacher cohort | 64.966 | 81.353 | 51.871 |
| 64 legal source teachers | **66.098** | **82.578** | **53.279** |

All four scenes improve all three metrics over the regenerated old-cohort
control. The gain over the retained mainline is +1.215 / +1.557 / +1.445 pp.
This is evidence for missing teacher coverage, not evidence for a novel loss.
The 64 cohort is an experimental control, not a newly imposed production cap.
All-source counts remain 293/122/169/180. Their feature reconstruction is ongoing;
ramen all122 already gives61.659 IoU/77.485 AP/43.722 selected, showing saturation.
Native feature dimension is1536; the semantic fitting raster remains96x128.

| Scene, 64 teachers | 2D IoU | AP | Selected projection |
|---|---:|---:|---:|
| figurines | 67.403 | 84.880 | 56.947 |
| ramen | 61.876 | 77.598 | 43.682 |
| teatime | 73.017 | 89.655 | 57.917 |
| waldo_kitchen | 62.095 | 78.178 | 54.571 |

The selected-subset metric is a 2D projection of selected Gaussians, not an
independent 3D-ground-truth benchmark. This result does not certify four-query
performance or replace the outstanding independent LERF3D evaluation.
Expanded range evidence is not bundled with this gain: its two-scene control
was mixed (next72). No scene-wise winner stitching is used.

## NVOS: transfer the evidence, not old Gaussian indices

`fit_nvos_evidence_next73.py` projects the retained pre-video native/structural
query result through the old map into legal source cameras. It reconstructs
those source probabilities through the new carrier's forward operator. Old
unobserved pixels are unknown; all new Gaussian support is recomputed. The
same video observations and original output operator are used on both carriers.
This is an explicit query-prior transfer experiment, not a claim that the new
carrier already contains a rebuilt native multi-teacher feature map.

Three bounded alternatives isolate where prior errors enter:

1. `joint`: video data terms plus source-rendered prior data terms at weight0.1.
2. `completion`: reconstruct prior first, initialize video optimization from it.
   This can retain erroneous prior values in the inverse problem's null space;
   it is NOT a hard guarantee of only modifying unsupported elements.
3. `unsupported`: optimize video from the same neutral initialization as the
   no-prior control, then use the separately reconstructed prior only where
   video evidence mass is zero. This prevents changes to video-supported
   element values, but does not certify that added elements belong to the query.

The reusable `solve_rendered_evidence` core is in
`radio_gs/querying/typed_evidence.py`. It fits bounded membership with full
visible forward operators and their exact transposes, using a Huber residual.
It accepts evidence from arbitrary typed providers; it cannot create instance
identity from consistency alone. No new tests/audit framework was added.

Initial evidence already rejects unconditional prior transfer: on new leaves,
video-only91.805 drops to66.503 with joint prior. Even restricting completion to
video-unsupported elements gives78.147: absence of video evidence does not
license a semantic prior to assert membership. Orchids joint improves65.911 to
68.516, but this isolated improvement cannot justify a global replacement.
The complete old/new, eight-query controls are recorded in
`output/next73/summary.json`; final aggregates are appended below after completion.

A second full-eight old/new geometry control consumes the already available
all-source video tracks. It completes the two-scene next72 pilot, without
inventing a new tracker or repeating the old-geometry result as a new method.
Outputs: `output/nvos_transfer_full_video_next72`.

## Consequences for method and paper

- Keep source coverage expansion as a justified input correction. It is not the
  central paper contribution and does not by itself reach mainstream LERF scores.
- A better geometry does not make a transferred semantic guess true. The missing
  ingredient is reliable query-instance support for completion, not a stronger
  generic smoothing term or more weight on the historical prior.
- Do not promote these NVOS prior variants unless the full cohort improves.
  Source masks, object-presence logits and old semantics remain fallible evidence.
- The high-capacity reference still depends on historical text-confidence
  transport. Arbitrary new-query calibration and compact multimodal student
  transfer remain unfinished; the new generic solver alone does not solve them.
- ScanNet has no newly trained result in this iteration. Public10 still needs
  native semantic reconstruction for0200/0645 and a matched full point protocol.
  UQIS remains paused as requested.


## Completed eight-query NVOS results

| Evidence rule | Old geometry IoU / AP | New geometry IoU / AP |
|---|---:|---:|
| Same limited-source video, no prior | 79.562 / 94.704 | 83.244 / 96.993 |
| All-source video, no prior | 80.110 / 94.835 | 83.395 / 97.107 |
| Joint source-rendered prior | 81.825 / 95.108 | 81.629 / 94.125 |
| Prior initialization | 79.764 / 94.858 | 81.858 / 96.625 |
| Unsupported-only prior completion | 79.742 / 94.855 | 81.976 / 96.883 |

Every row has eight queries in each geometry arm. All prior-transfer variants
fail replacement. All-source video on new geometry reaches83.395IoU/97.107AP,
versus the retained complete route83.484IoU/96.057AP: improved ranking, no IoU
win. Do not combine the best per-scene rows. The all-source old/new difference
is+3.285pp, while all-source versus limited-source video on the same new map
is only+0.151pp. Geometry is effective; more of the same tracker evidence
has largely saturated.

All293/122/169/180 original-resolution source SAM caches are now complete.
Native full-source feature encoding/fusion for the remaining LERF scenes is
still running under the next72 worker queues. No claim is made about their
unavailable final scores.
