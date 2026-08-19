# LERF identity/extent dual-posterior closure

The source-only, query-free instance hierarchy now closes the main LERF
failure identified in the two-round analysis.  The method keeps two typed
outputs from the same persistent Universal Field v1:

- an identity map from field SigLIP2 relevancy for localization;
- an object-extent map from official SAM3 multiscale source proposals,
  official SigLIP2 masked-crop identity, cross-view association and exact-MPR
  lifting for segmentation.

No second persistent semantic field is introduced.  Proposal construction uses
32 legal source views per scene and opens neither evaluation RGB nor evaluation
masks.  Query text is opened only by the readout.

## Full-four result

| Benchmark | Baseline mIoU | Candidate mIoU | Delta | Stability |
|---|---:|---:|---:|---|
| LERF2D | 0.31417 | 0.39584 | +0.08167 | positive on 4/4 scenes |
| LERF3D | 0.33450 | 0.39684 | +0.06233 | positive on 4/4 scenes |

LERF2D full4 localization accuracy is exactly preserved at `0.87981`.  LERF3D
Acc@0.25 improves from `0.57692` to `0.61538`, and Acc@0.50 improves from
`0.27404` to `0.40865`.  Before the query-independent peak-component guard,
the new 3D posterior already reaches `0.37572`; the instance posterior itself
therefore accounts for +4.12 mIoU points and the component guard contributes
the remaining +2.11 points.

The first evaluation exposed a typed-output implementation bug: the extent map
was upsampled to image resolution, while the independent identity map was not,
so the unchanged `30x30` localization smoother ran at the wrong scale.  The
corrected V9 path renders and resizes both outputs independently.  An
extent-disabled Figurines equivalence check restores LocAcc exactly and agrees
with primitive mIoU to less than `1e-5`.  V6--V8 are diagnostic-only.

This is stable full-cohort development evidence and is suitable for promotion
into the method definition.  It is not a prospectively blind result and does
not alone justify an SOTA claim.  The machine-readable authority is
`paper/artifacts/lerf_identity_extent_dual_posterior_full4_result_20260817.json`.
