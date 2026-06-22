# TPAMI Figure Quality Audit

Date: 2026-05-31

This audit records the current main-paper figure choices after visual inspection
of `paper/figures`. The goal is to keep only figures that directly support the
paper's main claims and to move diagnostic or protocol-mismatched assets out of
the main narrative.

## Selection rule

Main-paper qualitative figures must satisfy all four criteria:

1. The protocol matches a main quantitative table.
2. The example is visually legible after reduction to journal two-column width.
3. The baseline failure and GaussFM improvement are both understandable without
   extra explanation.
4. The figure does not depend on a stronger auxiliary readout than the method
   row claimed in the caption.

## Main-paper figures

| Figure | Decision | Rationale | Required caveat |
| --- | --- | --- | --- |
| `paper/figures/radio_gs_framework.pdf` | Keep as Fig. 1 after top-journal redraw | The current version is a vector-first conceptual figure with four columns: offline supervision, stored compact map, global readout heads, and protocol evidence. It keeps the central visual claim on one compact map, uses dashed paths only for training constraints, and avoids long protocol caveats inside the artwork. | Caption must distinguish training-only supervision from inference readouts and state that the main compact readout does not use a VPR cache or official RGB SAM decoder. |
| `paper/figures/lerf_2d3d_ovs_qualitative.png` | Keep as main qualitative | Best match to the recent open-vocabulary 2D/3D LERF-OVS taxonomy. Highest-quality examples are `old camera`, `green apple`, `pumpkin`, `tea in a glass`, `apple`, and `bag of cookies`. The earlier `green toy chair` case was replaced because its compact 3D selection was less clean. | Caption must state that 2D query is rendered-view and 3D query is primitive-level. |
| `paper/figures/scannet_openvocab_3d_query_qualitative.png` | Keep, preferably as a compact main figure | Correct binary query-point visualization for the VALA-aligned-style protocol. The strongest rows are `door` and `picture`, where the local OpenGaussian reproduction is visibly weaker; `cabinet` is acceptable but visually denser. | Use binary query masks in the main paper; reserve full 19-class coloring for appendix. |
| `paper/figures/lerf_direct3d_support_policy_ablation_qualitative.png` | Keep as main ablation or first appendix figure | Strongest ablation visual because it explains the largest method-level gain: support-aware primitive selection recovers small/fragmented objects (`knife`, `spoon`, `wavy noodles`, `plate`). | Caption should say no VPR cache and no official RGB SAM3 decoder are used. |

## Appendix-only or diagnostic figures

| Figure | Decision | Rationale |
| --- | --- | --- |
| `paper/figures/lerf_sam_dino_tasks_qualitative.png` | Appendix diagnostic only | Useful for frozen-head task discussion, but DINO matching is not visually strong enough for the main paper. |
| `paper/figures/lerf_vpr_direct_3d_qualitative.png` | Appendix diagnostic only | Strong boundary examples with official SAM3 box readout, but it can confuse the compact-only direct-3D claim. |
| `paper/figures/lerf_sam3_box_direct_3d_qualitative_pad16.png` | Appendix diagnostic only | Good for upper-bound/assisted-boundary discussion, not for the main compact-field result. |
| `paper/figures/lerf_rendered_grounding_qualitative.png` | Appendix or archive | Older rendered-grounding style and naming; superseded by the newer 2D/3D LERF qualitative figure. |
| `paper/figures/lerf_main_qualitative_comparison.png` | Appendix or presentation | Visually strong four-query comparison, but it only covers direct 3D against Dr. Splat and is less aligned with the current 2D/3D taxonomy figure. |
| `paper/figures/alpha_depth_boundary_cases.png` | Failure/diagnostic appendix | Shows boundary/edge cases rather than positive method evidence. |

## Example ranking

Recommended main-paper ordering:

1. LERF 2D/3D OVS: `old camera`, `green apple`, `pumpkin`, `tea in a glass`, `apple`, `bag of cookies`.
2. Direct3D support ablation: `wavy noodles`, `knife`, `spoon`, `plate`.
3. ScanNet binary query: `door`, `picture`, `cabinet`.

Avoid using the DINO/SAM adaptor figure as the leading qualitative evidence:
it is useful for a frozen-head diagnostic but the DINO matching panels are not
visually decisive enough to support the strongest journal claim.

## Figure-1 redesign notes

The new framework figure replaces the older module-heavy layout with a
paper-facing hierarchy:

1. Offline supervision and frozen heads are on the left.
2. The stored scene representation is the central `Hybrid Gaussian Code Field`.
3. Global decoders and support policy are separated from stored per-scene data.
4. Protocol evidence is on the right with visual icons: rendered-view,
   direct primitive, direct point, and frozen-head probes.
5. Training-only constraints are placed in a bottom band with dashed arrows.

This layout supports the intended main claim: one compact foundation-feature
Gaussian map supports rendered-view query, direct primitive query, and direct
point query, while VPR and frozen heads remain supervision or diagnostic
interfaces rather than extra inference caches.
