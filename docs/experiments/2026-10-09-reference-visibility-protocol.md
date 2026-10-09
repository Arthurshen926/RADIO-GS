# Match visible reference evidence without removing occluders

Fixed full eight requests, same seven fields/120 source views, same retained v10
positive-SAM current-query masks (inherited RGB/SAM generation cost, not rerun).
No target GT in prediction; same candidate Gaussian subsets Bpositive>0 intersect
member_available, same greedy reference soft-IoU criterion and binary membership.
No new state, training, weight/threshold search or per-case choice.

1. subset_reference: reproduce previous mask_composition, projecting each subset
   by itself for reference comparison, hence removing all other scene geometry.
2. Primary visible_reference: use normalized full-scene Gaussian contributions
   to project each candidate membership into the current reference camera. Divide
   by observed-field mass only where the observed fraction exceeds0.5; unsupported
   rows abstain. Candidate foreground still thresholded at0.5. Full-scene visibility
   remains fixed, so an occluded object's region is not treated as visibly outside
   the supplied reference mask. This is the same observed-support normalization
   convention as the existing retained response, using the contribution operator.

Only reference observation operator changes; scope selection is still greedy,
not globally optimal, and scene visibility can leave hidden extent ambiguous.
Target actual-subset rendering and normalized response scoring remain unchanged.
Preserve reference actual-subset fit as a separate diagnostic; in visible mode it
need not equal the reference projection used for selection.

All16 predictions plus eight exact baseline selection replays before scoring.
Report all cases and both formal metrics/precision/recall/AP. Interpret versus
raw-scribble arms as changed input condition, never raw standard-mode recovery.
This is sequential development after the first mask study, not a blind final test.
No global default promotion or full-five claim follows automatically.
