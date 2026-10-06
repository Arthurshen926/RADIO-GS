# Cross-view scope evidence and uncertain query connectivity — next22

This round tests two fixed method changes against unchanged maps and scoring contracts. Neither experiment is a new backbone, geometry reconstruction, query encoder, or target-fitted threshold. Results are completed below by the sealed summary; unsuccessful candidates must not replace the previous candidate.


## Completed results

All four LERF scenes / 208 observations and all eight NVOS tasks completed. Five focused tests passed (`tests/test_scope_view_relation.py`, `tests/test_uncertain_scope_bridge.py`). All 1773 frozen source hashes remain unchanged.

| Fixed cohort / arm | IoU % | AP % | Selected-element rendering IoU % |
|---|---:|---:|---:|
| LERF previous local, no graph | 50.3637 | 68.8626 | 34.1887 |
| LERF new local, no graph | 49.4534 | 68.5999 | 33.4460 |
| LERF direct replacement control | 45.5385 | 65.6652 | 31.9752 |
| NVOS reference-core baseline | 81.1608 | 91.5057 | — |
| NVOS single uncertain connector | 81.1392 | 91.4980 | — |

Both candidates are rejected; no performance improvement is claimed. The previous LERF graph-composed candidate (50.4911% IoU, 36.7387% selected) remains untouched. NVOS baseline masks replay exactly; continuous scores differ by at most 2.3842e-7. New connectivity is nearly a no-op or slightly harmful, so this particular short-gap hypothesis does not explain the remaining benchmark gap. It does not rule out every possible connectivity issue.

Machine-readable paired results: `output/scope_connectivity_next22/summary.json`. Receipt: `paper/artifacts/scope_connectivity_next22_complete.json`. These remain development diagnostics under the previously frozen scoring conventions, not an independent final-test result.

## Cross-view evidence fusion

Four LERF scenes, 208 observations per arm, native-grid probability IoU/AP plus the existing selected-element rendering metric. The latter is not volumetric 3D IoU. The comparison to next20 uses the same no-graph setting on both sides; next20's graph-composed result must not be compared directly to this no-graph candidate.

Implementation: `caption_view_consistency_next22.py`, `scope_view_relation.py`. First replay the old all-source renderer-adjoint accumulation. Then subtract each current view's numerator, denominator and valid-view count before projecting the remaining evidence back into that view. Lone-view floating-point remnants cannot certify a correspondence. Classify visible masks as mutual support, current-part, current-whole, weak visibility, disjoint or uncertain. Mutual support contributes foreground and background; containment contributes foreground only; other cases abstain. Keep the old deployed concept route and bounded local correction, with exact baseline fallback outside accepted evidence.

This holds out only the current view's numerical contribution. Caption grouping and the parent field have seen all source views. The last four source frames are in-sample diagnostics, not independent validation. SAM masks are pseudo labels. Same-caption unions are not independently verified object identities.

### Findings

| Scene | Previous local IoU % | New local IoU % | Previous covered element/concept fraction % | New fraction % |
|---|---:|---:|---:|---:|
| figurines | 50.7945 | 50.2831 | 44.6791 | 13.7001 |
| ramen | 35.9660 | 34.9004 | 55.9684 | 28.2623 |
| teatime | 60.9620 | 60.3559 | 45.3041 | 30.4726 |
| waldo_kitchen | 53.7323 | 52.2743 | 44.9764 | 22.0024 |

The local candidate regresses in all four scenes against the previous local correction. Do not cherry-pick scenes or change thresholds from these results. A full graph-composition sweep is not justified by this failed prerequisite.

Two concrete limitations are visible in the implementation and diagnostics:

1. Whole-view categorical rejection removes a great deal of usable evidence. In figurines only 520/2076 view/group observations are mutually supporting; 700 are containment cases and 856 abstain. An ambiguous aggregate cannot reliably assign a single relation to every pixel and every instance in the group. The measured coverage collapse is real; its exact causal contribution to IoU is not separately isolated.
2. Foreground-only accumulation uses `numerator=n, denominator=n`. Every nonzero contribution therefore has posterior one, even if its original renderer-weighted foreground fraction `n/d` was small. Two weak tails can appear as repeated, zero-variance positive evidence. This is a limitation of the experimental evidence representation, not proof that all production fusion paths contain this bug. It is a reason to reject this candidate, not tune its acceptance thresholds.

The original source accumulation replays within 2e-6; frozen baseline probability masks and selected-object results replay exactly. The target result therefore cannot be explained by a changed evaluator or changed geometry in this comparison.

## NVOS connectivity

Eight tasks, the same maps and reference-core completion as next19. Add edges through a single uncertain node with score at least .25 and two candidate neighbors. Forbid uncertain-to-uncertain edges. Connector nodes do not become output candidates merely by connecting components; the original positive-only repair still follows. Baseline and candidate receive the same explicitly prepared query scores, with no global query cache.

This tests a narrow claim: a short uncertain gap is the main cause of lost object components. It does not verify physical object identity across a bridge. Reference RGB/SAM and positive prompts remain allowed; target RGB and negative scribbles are not opened. This is the existing positive-only all-view contract, not a claim of full official signed-prompt equivalence.

All eight baseline scoring masks must match prior results exactly, with continuous optimizer replay tolerance 1e-6. All parameters are fixed before evaluation. Target GT is read only after predictions are saved.

## Next method prerequisite

Do not add another graph or a larger field to compensate for unidentified scope errors. Preserve native semantic routing and the previous accepted experimental baseline. The next evidence representation must separately carry visibility mass, foreground mass, background evidence, view identity, and scope identity. Containment may grant one-sided support, but must not turn small absolute evidence into confidence one. Split physical/scope hypotheses before averaging their masks; uncertain correspondence must not reject an entire otherwise useful view. Validate correspondence on source views excluded from hypothesis construction, then evaluate probability masks and actual selected elements jointly. A targeted ablation should isolate confidence normalization from relation rejection before attributing the full regression to either one.

No new ScanNet OVS result is produced by these controls; UQIS remains paused due to access. No default promotion, independent final-test claim, SOTA parity claim, or four-query completion claim is warranted.
