# Single-Query Latency Evidence

Lower is better. The current table converts frozen evaluation profiles into single-query latency units. These are conservative profile-derived values, not a clean warm-GPU microbenchmark: the LERF rendered profile includes teacher/evaluator work and visualization I/O, and the direct-3D profile includes query-select-render mask generation.

| Task | Unit | #Queries | Total time | Latency / query | Peak VRAM | Source |
|---|---|---:|---:|---:|---:|---|
| LERF rendered-view OVS | view-query | 356 | 124.770 s | 350.5 ms | 2076 MiB | `output/radio_gs/profiles/freeze_lerf_*_overlay_20260502` |
| LERF direct 3D OVS | object-query | 208 | 492.472 s | 2367.7 ms | - MiB | `output/radio_gs/lerf_direct3d_prompt_ensemble_policy_20260528/figurines/lerf_direct_3d_selection_results.json, output/radio_gs/lerf_direct3d_prompt_ensemble_policy_20260528/ramen/lerf_direct_3d_selection_results.json, output/radio_gs/lerf_direct3d_prompt_ensemble_policy_20260528/teatime/lerf_direct_3d_selection_results.json, output/radio_gs/lerf_direct3d_prompt_ensemble_policy_20260528/waldo_kitchen/lerf_direct_3d_selection_results.json` |
| ScanNet point query | class-query | 389 | 150.903 s | 387.9 ms | 1666 MiB | `output/radio_gs/profiles/freeze_scannet_v67_all_eval_20260502` |

## Notes

- LERF rendered-view OVS: conservative profile; includes teacher branch, all queries, and visualization I/O.
- LERF direct 3D OVS: query-select-render evaluation; includes selected-primitive rendering and mask writing.
- ScanNet point query: legacy 10-scene profile; reports point-query class scoring throughput.
