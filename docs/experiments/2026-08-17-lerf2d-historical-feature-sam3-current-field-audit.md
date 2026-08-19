# Historical feature-only SAM3 readout on current LERF2D field

The historical `0.588944` LERF2D row cannot be restored as a current
Universal Field v1 result.  A full four-scene transfer already exists and is
numerically positive, but it is a non-promotable, transductive development
diagnostic.

## What the historical number contains

The historical boundary head improved its own peak-component masks from
`0.570723` to `0.588944` scene-macro mIoU (`+0.018222`).  Most of the apparent
gap to the current method is therefore not a SAM-boundary gain.  It comes from
the older rendered semantic field and its post-hoc readout: scene-specific
softmax temperatures (`50/40/25/25`), threshold sweeps, and acceptance guards
were selected on the same four evaluated scenes.  The repository's July 11
protocol audit already classifies this row as a legacy post-hoc diagnostic.

## Existing current-field full4 transfer

The current transfer uses the fixed primitive relevancy readout, a peak
component, the old feature-only head, and no target RGB.  The rendered 1280-D
inputs decode from the Method-v1 source fields whose Universal Field v1
migration reports verify bitwise unchanged RADIO decode.

| Scene | Peak component | Old SAM head | Delta | Accepted |
|---|---:|---:|---:|---:|
| Figurines | 0.404034 | 0.410184 | +0.006150 | 30/56 |
| Ramen | 0.263189 | 0.281068 | +0.017880 | 16/71 |
| Teatime | 0.406878 | 0.420405 | +0.013527 | 21/59 |
| Waldo Kitchen | 0.337751 | 0.339336 | +0.001586 | 1/22 |
| Scene macro | 0.352963 | 0.362748 | +0.009786 | 68/208 |
| Sample micro | 0.349753 | 0.361517 | +0.011764 | 68/208 |

This is a real, all-scene-positive boundary correction.  It is not the missing
object-extent solution: only 68 of 208 masks are changed, accepted masks are
constrained to stay close in area and spatial support to the coarse mask, and
the head cannot recover an object that the primitive identity heatmap never
covered.

## Contract audit

The training and target frame sets do not overlap, and evaluation does not
open target RGB.  The old probability/logit bug is also irrelevant here: these
heads binarized the stored `[0,1]` SAM probabilities directly at `0.5`.

The blockers are elsewhere:

- Each source-view SAM cache was prompted with the exact benchmark category
  list obtained from the evaluation label directory.
- Four different scene-specific heads were trained.  They encode the
  benchmark-query pseudo masks and are required in addition to Universal Field
  v1 at cold start.
- The heads were trained on historical scene-specific HCD feature domains, not
  on the current Universal Field distribution.
- Historical gates and temperatures were selected retrospectively on the
  official evaluation cohort.
- Old checkpoints and SAM caches are not fully content-bound to every source
  image, cache shard, and rendered training feature.

Consequently the branch is useful as evidence that boundary polishing has
about one mIoU point of headroom on the current coarse masks, but it is not a
valid unified open-query method candidate and cannot support a `0.588944`
current-field claim.

The contract-correct replacement is query-independent source-view automatic
SAM hierarchy construction, exact-MPR positive/negative/unknown membership,
cross-view object association, and proposal-level text identity at query time.
That targets object membership rather than only polishing an already-selected
coarse boundary.

Machine-readable audit:
`paper/artifacts/lerf2d_historical_feature_sam3_current_field_contract_audit_20260817.json`.
