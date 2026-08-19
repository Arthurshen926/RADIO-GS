# Typed instance-readout promotion and five-benchmark continuation

## Promoted development method

The persistent representation remains Universal Field v1.  The promoted
readout family separates query identity from object membership instead of
averaging primitive and region text scores.

| Contract | Frozen baseline | Promoted candidate | Gate |
|---|---:|---:|---|
| LERF2D full4 mIoU | 0.31417 | 0.39584 | 4/4 scenes positive; LocAcc unchanged |
| LERF3D full4 mIoU | 0.33450 | 0.39684 | 4/4 scenes positive |
| ScanNet paper8 19-class mIoU | 0.33535 | 0.33810 | positive mIoU and mAcc |
| ScanNet paper8 15-class mIoU | 0.33370 | 0.33617 | positive mIoU and mAcc |
| ScanNet paper8 10-class mIoU | 0.42213 | 0.42548 | positive mIoU and mAcc |
| NVOS full8 macro IoU | 0.52687 field-only | 0.81776 RGB-assisted | sealed prediction before GT |
| SPIn Available9 | no new full9 result | running | requires 9/9 Method-v1 gates |

LERF uses the source-view official-SAM/SigLIP identity--extent posterior.
ScanNet keeps categorical identity in the field posterior and applies only a
low-margin bounded SAM-affinity residual.  NVOS uses independent box and point
SAM hypotheses with post-decoder consensus.

## Rejected candidates

- Direct official-SAM categorical propagation failed independent ScanNet
  confirmation and is not part of the method.
- Increasing LERF proposal recall by lowering the absolute SigLIP descriptor
  floor from 0.55 to 0.50 and expanding the per-view candidate set from 3 to
  5 failed the first-scene stop rule: Figurines LERF3D mIoU fell from
  `0.51650` to `0.4978`.  The remaining error is therefore not addressed by
  admitting more weak regions; identity-calibrated association remains the
  required gate.
- Replacing the absolute descriptor floor with a query-listwise margin raised
  accepted Figurines queries from 4 to 8, but reduced LERF3D mIoU further to
  `0.4657`.  Cross-view overlap plus a field-peak anchor is therefore not
  sufficient to identify weak-text instances; this diagnostic is rejected
  before full4 expansion.
- Joint box-plus-signed-point prompting reduced NVOS macro IoU from `0.81776`
  to `0.68259`; it is retained only as a negative control.
- Historical SPIn carriers cannot substitute for a new D512/L512 Method-v1
  field result.

## SPIn materialization correction

On the current host, every GPU has about 17 GiB of unrelated resident load.
The remaining fields therefore use exact low-peak execution: FP16 L512 storage,
FP32 decode and Adam moments, full-grid xFormers attention, exact token-wise MLP
chunking, column-staged raster-adjoint replay, and temporary CPU residency of
the L512 table while the frozen capability graph is differentiated.  These
changes preserve the method objective and alter only execution residency.

Plain forward chunking was insufficient under the shared-GPU load because the
outer activation checkpoint rebuilt and retained every frozen-MLP chunk during
backward.  The accepted execution path now applies the exact frozen-head VJP:
it saves the MLP input, recomputes one token block at a time in backward, and
immediately accumulates only the input gradient.  Official weights, full-grid
attention, outputs, and first-order gradients are unchanged.  A direct
standard-versus-bounded comparison passed with maximum input-gradient
difference `1.19e-7` (float rounding scale).

The frozen full9 runner verifies all nine fields and signed margins before
opening RGB, seals all target predictions before opening evaluation masks, and
then scores the fixed Available-Nine cohort.  Until that receipt exists, SPIn
is reported as incomplete.

At the 2026-08-19 22:56 CST handoff, gates exist for `horns`, `leaves`,
`lego`, and `orchids` (4/9).  `fern`, `fortress`, `pinecone`, `room`, and
`truck` each have a live assigned-GPU runner plus an idempotent watchdog that
retries only after the scene lock is released.  A separate frozen readout
waiter starts automatically at 9/9.  No new Method-v1 Available9 score is
reported before that receipt; the historical-carrier rows remain controls.
