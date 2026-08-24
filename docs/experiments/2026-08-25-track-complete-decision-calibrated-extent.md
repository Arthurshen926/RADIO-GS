# Track-complete, decision-calibrated extent closure

## Question

The previous source gate trained and evaluated target-view-visible proposal
fragments.  It did not establish that a decoder recovered a complete physical
object track over the source-observed Gaussian domain.  It also replayed raw
cosine identity outside the extent authority while calibrating the posterior
with a different threshold.

## Method changes

1. Aggregate all confirmed same-object proposal memberships into a sparse
   track authority using noisy-OR positive evidence. Missing membership remains
   unknown; only explicit different-instance evidence is negative.
2. Split train/dev/audit by physical `episode_object_id`, not target view.
3. Evaluate over every source-observed Gaussian, in anchor-preserving chunks.
4. Normalize extent conditioning against the anchor score gauge.
5. Convert identity cosine to a decision logit before adding the authority-
   gated residual. The strict candidate fixes primitive threshold at `0` and
   posterior threshold at `0.5`, so authority-off rows replay the same decision.

The frozen D512/L512 Universal Field, source RGB contract, and episode physical
authority are unchanged.

## Decisive results

The uncalibrated 10-step smoke gate failed: dev macro delta `-0.04641`, audit
macro delta `-0.07330`, and outside-authority foreground probability `0.65462`.
The primitive and posterior thresholds were in incompatible gauges.

Decision calibration immediately reversed the audit delta to `+0.05422` after
only 10 steps.  The strict fixed-threshold run at identity center `0.6805`
passed both track-disjoint dev and full source-domain audit:

| split | Ramen delta | Teatime delta | Waldo delta | macro delta |
|---|---:|---:|---:|---:|
| dev | non-negative | non-negative | non-negative | `+0.00466` |
| audit | `+0.02773` | `+0.02088` | `+0.00786` | `+0.01882` |

Checkpoint:
`/mnt/pool/sqy/results/RADIO-GS/output/optimization_20260825/lerf_track_complete_extent_fixed_decision_v1/center0p6805_s20260826.pt`
(`e2fcbae54d2e4526b4238fbe70fa1025f2d306f87bc1461b861579042f2c8837`).

The nearby center `0.6686` failed dev (`-0.00353` macro), so the decision zero
point is a real identifiable variable rather than harmless score scaling.

## Text diagnosis and rejected candidates

With the passed image extent frozen in the same run, old object-bound attribute
text still failed:

| diagnostic | macro delta |
|---|---:|
| text identity + oracle image anchors | `-0.02173` |
| text identity + text-local anchors | `-0.03837` |

Removing the row identity map made image completion fail (`-0.03081` audit),
showing that it contains necessary boundary evidence. Mixing the weak text
authority into extent training also failed: weights `0.25/0.5` produced image
audit deltas `-0.04900/-0.10857`.

Track-consistent phrase selection was implemented and materialized, but manual
audit rejected its semantic authority: examples include `glass parking meter`
and `thick binder` for Waldo tracks, and the held-out Waldo track had zero audit
eligible episodes. Cross-view consistency can stabilize an incorrect phrase;
it cannot replace a genuine region caption.

## Conclusion

The physical completion mechanism is now supported under a substantially
stricter, leak-free source contract. The old full4 failure is explained in part
by decision-gauge mismatch. However, LERF text deployment is not promoted:
oracle-anchor text failure proves that a genuine object-bound language identity
authority (or a separately calibrated text identity map) is still required.
Running benchmark full4 before that source gate passes would be metric search,
not method validation.
