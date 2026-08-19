# NVOS exact-W two-round SAM3 sentinel (2026-08-17)

## Decision

`nvos-method-v1-two-round-exact-logodds-sam3-v1` is **not promoted**.  The
frozen fern sentinel regressed from the sealed one-shot foreground IoU
`0.8304733` to `0.8171996`.  The pre-registered stop rule was applied, so a
two-round full8 score does not exist and must not be inferred from this run.

The machine-readable result is
`paper/artifacts/nvos_method_v1_two_round_exact_consensus_sam3_fern_sentinel_result_20260817.json`.

## Fixed pipeline

The candidate used the already sealed Method-v1 signed field, signed-evidence
selected official SAM3 box output, and ten-trial point-SAM output.  No target
mask or metric entered prediction.

1. Render the registered target camera's exact accepted-hit 3DGS compositor
   at `756x1008`.
2. Lift field, box, and point probabilities independently with
   `W.T @ p / (W.T @ 1)`.
3. Fuse the three primitive observations by coordinate-wise median of clipped
   log odds (`epsilon=0.05`).
4. Rerender with `W @ u / (W @ 1)`.
5. Use the rerender as the official SAM3 mask-logit input, together with the
   original frozen signed points and a fixed padding-16 box; average ten
   single-candidate outputs and threshold at zero.

The official checkpoint's prompt encoder requires a `288x288` mask input for
the `1008` image resolution.  The generic SAM predictor docstring says
`256x256`, but the first no-GT sentinel interface check showed that this
checkpoint's image embedding is `72x72`, requiring the authoritative
`sam_prompt_encoder.mask_input_size=(288,288)`.  The runner now validates this
fail-closed.

## Exact transport audit

The fern target compositor completed without OOM alongside the existing GPU
workload:

- Gaussian rows: `760,715`
- accepted front-to-back hits: `69,732,384`
- visible Gaussian rows: `623,025`
- supported target pixels: `100%`
- rerender probability range/mean: `[0.04999995, 0.94994986] / 0.24157892`

Thus the result is not an API fallback or incomplete-visibility failure.  It
uses the requested exact target-view `W.T` and the same exact `W` for replay.

## Paired fern result

| Stage | Foreground IoU | Pixel accuracy | Delta from previous |
|---|---:|---:|---:|
| Sealed one-shot box+point parent | 0.8304733 | 0.9456364 | — |
| Exact `W.T/W` rerender at 0.5 | 0.8232267 | 0.9445676 | -0.0072466 |
| Round2 official SAM3 aggregate | 0.8171996 | 0.9409577 | -0.0060271 |
| Total two-round delta | — | — | -0.0132737 |

The loss decomposition closes exactly within stored float precision:

```text
-0.0072465501 + -0.0060271268 = -0.0132736769
```

At ground-truth resolution, round2 changed `298,434` pixels relative to the
one-shot mask.  Of those changes, `121,362` corrected one-shot errors and
`177,072` introduced errors, a net harmful balance of `55,710` pixels.  The
round2 output kept almost the same foreground area as the parent (`0.2984287`
versus `0.2984628` at SAM resolution), so the degradation is primarily a
boundary/location shift rather than gross foreground-size collapse.

## Ten-trial stability

The ten round2 trial IoUs lie in `[0.8152728, 0.8182231]`, with mean
`0.8164858`, population standard deviation `0.0011927`, and range `0.0029504`.
Official SAM qualities lie in `[0.9296875, 0.94140625]`.  Every trial is below
the sealed one-shot parent despite low variance, so this is a systematic
transport/self-conditioning failure rather than unlucky signed-point sampling.

The exact rerender already loses `0.00725` IoU: same-view adjoint averaging
mixes foreground/background evidence on partially overlapping Gaussian
footprints.  The mask self-prompt then makes the transported support a strong
dense condition and does not recover the original one-shot boundary.  Its
second `0.00603` loss shows that another SAM pass is not an automatic extent
repair when its dense prompt is already biased.

## Scope and interpretation

The existing full8 figures remain:

- signed-field selected box SAM3: `0.8176168` macro foreground IoU;
- one-shot box+point supermajority: `0.8177621`.

They are context only.  There is no paired two-round full8 figure because the
fern sentinel failed and `promotion=false`.  The implementation is retained as
a reproducible negative ablation and as an executable exact-W/mask-prompt
reference.  No frozen parameter was changed after the fern metric was opened,
and no other full8 target mask was opened for this candidate.

