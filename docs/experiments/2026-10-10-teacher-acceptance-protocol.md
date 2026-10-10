# Source-teacher acceptance insertion: conditional capability diagnostic

Four LERF scenes/68 texts, same exported fields and 72 partial-relation source
views. Extra input is cached RAW v10 query-conditioned source detection masks.
This is explicitly SOURCE-ASSISTED DIAGNOSTIC, not a deployed/source-free result,
not training, not model promotion, and not complete v10 identity admission.
No target labels or target images enter selection, prediction or stopping.
All four scenes/all controls sealed before any scoring.

For each query/view, on the exact existing candidate regions (source_records),
bilinearly resize max detector mask probability to the existing 96x128 raster,
threshold at0.5, then greedily select the union that strictly improves source
teacher IoU, lowest-index tie. This is the previous feasible diagnostic operator,
not a global optimum. Selected coordinates receive acceptance1, others0.5 (zero
positive preference). Empty detection means no selection/unknown, NOT negative.
Binary confidence replaces native confidence as part of this privileged acceptance
intervention; gains cannot be attributed to identity selection alone.

Controls: admission, native_parts, native_protected (exact cached payloads),
teacher_parts, teacher_protected (PRIMARY: same mixture/member update),
teacher_union_parts, teacher_union_protected (SECONDARY composition intervention).
Teacher mixture preserves the current finite P/N update. Teacher union replaces
same-view mixture with max-positive/min-negative over accepted regions, preserving
unknown negatives. This marginal union is NOT exact source-raster union after
lifting. Both pairs prepare P/N once for protection; no threshold/temperature sweep.

Acceptance matrices are query-transient diagnostic artifacts outside field.pt,
never a scene answer bank or standard runtime inputs. Field bytes and geometry
unchanged. Report source detection receipts, source-fit fidelity, all seven arms,
response/AP/actual-set projection and per-scene changes. No cross-metric winner
bundle. If teacher mixture restores scope, prioritize field-owned conditional
acceptance transfer with non-evaluation supervision. If only teacher union does,
acceptance and composition must migrate jointly. If neither does, examine lifting,
unknown prior retention, coverage and member decoding before training a new head.
