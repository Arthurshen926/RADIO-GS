# PFPR LUDVIG protocol audit

Status: exact released LUDVIG DINO/PCA/uplift plus a custom one-scene PFPR
adapter is complete; the older six-scene C-RADIO/DINOv3 sanity is retained.
Neither is a LUDVIG paper-metric comparison because LUDVIG has no PFPR task.

## Exact-LUDVIG adapter closeout

- scene/query coverage: `scene0050_02`, 10 public crops;
- field coverage: 120 query-held-out views, 300,000 shared Gaussians;
- public domain: 31,143 candidates, all with strictly positive readout support;
- exact component: released vendored ViT-g/14 behavior, scene PCA40, sliding
  reconstruction, and `GaussianModel.apply_weights` inverse-render uplift;
- custom component: center-3x3 query pooling, primitive cosine, and continuous
  opacity-weighted Gaussian/5 cm cell readout;
- Top-1 mean/median: 1.9873/1.9546 m;
- R@1/R@5/R@10 at 10 cm: 0/0/0;
- R@1/R@5/R@10 at 20 cm: 0/10/10%;
- interpretation: normal negative sanity for an untuned learning-free adapter,
  not evidence of an evaluation-protocol mismatch.

The official reg4 checkpoint has SHA256
`746ecb8c6301c645c5c855be91687d274587d6e48fdaec4a729753160b34a283`,
but the released LUDVIG vendored model has no register-token support. The
primary exact run explicitly permits and discards only `register_tokens`,
matching the released `strict=False` behavior without hiding other mismatches.
Phase A/B/C/D/E manifest SHA256 values are
`de3f0281ae863d9f640eaabce5c805f35a9fca03221ceee0a67756fc9023e22a`,
`1d546a335e2f3ec807c69b23f06a7876d1a325d53e16a94b0d64f8b7556d147b`,
`dcf8d864da50aa455f805d94bce707daf4cfc4ac1f602ea03141b38a57eb13fb`,
`e7f615ed0013cc32b858c92df364c36585dcfa1a10758202bcb361035c42b6ee`,
and `d85c59aebb21cb6c6b6c1251c737f82c88c36eea58e4c8175167a71797c32901`.

Only Phase E opens evaluator-private anchors. Phase B/C/D reached 58/52/49 C
on physical GPU0 with zero pause events, so the executed 78/81/70 C thermal
policy with one stable cool sample did not limit this reproduction.

## Historical C-RADIO/DINOv3 scope and result

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
