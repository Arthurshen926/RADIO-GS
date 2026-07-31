# PFPR LUDVIG-style DINO uplift sanity

Status: diagnostic only; not an official LUDVIG reproduction and not a
paper-metric comparison.

## Scope and result

- benchmark: corrected `scannet-pfpr-small-v2`
- coverage: 6/20 scenes, 60/200 queries
- continuous-support gate: disabled in this historical rapid-iteration cache
- top-1 median error: 0.690 m
- R@1/R@5/R@10 at 10 cm: 16.67% / 35.00% / 41.67%
- saved query-micro and scene-macro metrics: exactly reproduced by a fresh
  evaluator pass over the immutable score vectors

The scorer sees only `scene_id` and a depth-aligned 128 x 128 RGB crop. It
does not see query pose, depth, mask, class, instance, source frame, or the
private 3-D anchor. The evaluator alone opens the anchor to calculate ranked
Euclidean errors.

## Why the name is “LUDVIG-style”

Official LUDVIG commit
`4461fc515439bb498a75d71738a1e73cf7a452ed` does not implement PFPR
patch-to-point retrieval. This cache was produced by RADIO-GS's direct-DINO
scorer, not an official LUDVIG wrapper. The five material differences are:

1. PFPR retrieval is not an official LUDVIG task or metric.
2. C-RADIO DINOv3-7B replaces official DINOv2 ViT-g with registers.
3. A canonical MPR field replaces official multi-view inverse-rendering
   uplift into Gaussians.
4. No official LUDVIG kNN graph diffusion or scribble regularizer is used.
5. The query is one held-out RGB crop rather than scene-view feature maps.

The historical prediction report did not record its generator commit.
Therefore the audit binds the cache itself and records the current scorer
only as audit-time context; it does not retroactively claim implementation
identity.

## Immutable provenance

- score-vector set SHA256:
  `0e58c1914b3054bf7ac58da2aca7a1287574bd430310b924e379b0f5b2f38503`
- prediction-report SHA256:
  `20413b383b17cabb2a31248e31c21298ac8849625eb014990b2da07c3f76032d`
- source saved-result SHA256:
  `398c925c77c413f11fab05b94554b612ecdb953dcd1f5c506621daaac44f4d11`
- scorer SHA256 at audit:
  `e5081e267bc9d3a6e16b12b71559d20124be444b522c7ad49e9cfb8ccf7cc42c`
- repository commit at audit:
  `1367eb7186213b9290e7d737551fea5fe629f052`
- recomputed query-micro identical: yes
- recomputed scene-macro identical: yes

The full local audit remains outside the small paper snapshot at
`output/protocol_audit_20260731/pfpr_ludvig_style_v2_partial6/audit.json`.
The 75 KB per-query evaluation is intentionally not duplicated here.
