# Exact historical source archives

These files preserve code bytes referenced by existing result seals. They are
not modified copies of the current implementation and are not a dependency
bundle or a new benchmark run.

| Archive | Independent original authority | SHA-256 |
|---|---|---|
| `evaluate_full_field_rebuild_next19.py` | `paper/artifacts/full_field_rebuild_next19_complete.json` | `68dac7b79f67aae255125fb9bee7d68e7c156d92c574b369e7f9e5e0744b5a7e` |
| `ludvig_run_sam_20260801.py` | `paper/artifacts/promptable_nvs_protocol_registry.yaml` | `1f1bc95d70f22fcfb1cfe2df1bc3f2416ebb667c2faeddb7dc24bcd9bceed68f` |

The LUDVIG file was recovered from git commit
`4a668744707b93aa5394585810677e0a3f3e80ba`, path
`reproductions/ludvig/run_ludvig_sam.py`. Its original relative-path assumptions
require the original repository layout if executed; do not execute this renamed
archive directly as though it were a freshly configured launcher.

The LERF evaluator was recovered by reversing the two recorded September 24
candidate-entrypoint patches. Its resulting bytes exactly match the independent
September 19 seal. The current entrypoint adds candidate field/output naming,
prevents reuse of baseline query caches for alternate fields, and adds candidate
source-validation metadata. Its score/render/selection/metric block is byte
identical to this archive. Recovery/diff evidence is in
`output/publication_readiness_next31/recovered_next19/`.

`verify_current_four_release_next31_v2.py` accepts an archive only when its hash
matches the *original* authority. It retains the current-path mismatch in the
report. Neither existing code nor historical seals are overwritten. The original
seven-task global validator remains separate; missing SPIn/PFPR assets are not
hidden by the active-four check.
