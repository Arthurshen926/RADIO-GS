# Prompt metric alignment: fixed cutoffs-only control

The base prompt loss trains the query-adapter route, while deployed prompt retrieval uses scale-gated structural affinity. This is an objective/readout mismatch; it does not by itself prove that fitting affinity will improve complete-object retrieval.

All four scenes used 12 passes, seed42, Adam0.01, source training views and the original two validation views. Only the 32 affinity cutoffs were trained; the parent deployment scale and default-scale buffer were frozen, and the exact same scale was used for every training region. Geometry, the native64+structure32 latent, all other heads, and architecture were exactly preserved. No benchmark labels selected parameters.

| Scene | Source fixed-scale IoU before → after | Development prompt IoU before → after | AP before → after |
|---|---:|---:|---:|
| figurines | 23.45 → 20.62 | 47.38 → 44.96 | 65.20 → 62.50 |
| ramen | 29.09 → 27.27 | 33.93 → 29.72 | 50.19 → 49.29 |
| teatime | 37.41 → 34.50 | 68.83 → 63.07 | 86.93 → 85.57 |
| waldo_kitchen | 57.20 → 51.72 | 21.43 → 20.84 | 53.53 → 50.73 |

Scene-macro prompt IoU: 42.89% → 39.65%; AP: 63.96% → 62.02%.

Text/image observation metrics are exactly unchanged, and cold source-query scores have zero text/image error. Every saved tensor except affinity.cutoffs is byte-value identical to the parent. The inherited trainer history field updated_elements counts sampled rows, not latent updates; structural_updated_rows is zero. Grounded supervision hashes were independently checked against their source receipts.

These are fixed reference-query development diagnostics using category-union masks, not an official four-query benchmark. No scene-wise winner selection and no automatic promotion. This experiment isolates metric fitting, but source pair BCE and known-element masks still do not supervise complete cross-view object extent. Better alignment needs reliable scope-aware evidence, rather than blindly increasing metric training.

Training at mask-derived per-region scales while deploying one fixed scene scale is a separate input mismatch. This second registered control removes it: only the parent deployment scale is used, with no scale selection. All runs use read-only dependency hardlinks and write only new candidate directories.

Receipt: `paper/artifacts/prompt_metric_deployscale_next19.json`. Results: `output/prompt_metric_deployscale_next19/{scene}/summary.json`.

Decision: reject this candidate. Source validation worsened in every scene. Neither aggregate AP nor one scene improvement overrides that failure.

Teatime recovery: adding dependency hardlinks changed inode ctime and caused a concurrent immutable-reader rejection without changing source bytes. Once all links were established, stable reads passed. In-process cold checks exceeded GPU memory with training data resident; failed attempts were retained, and cold validation ran separately with all three original queries per modality, one at a time. Both candidates produced exactly zero text/image score error. Full reference-query evaluation was also run in separate processes.
