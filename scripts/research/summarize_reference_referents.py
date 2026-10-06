"""Summarize completed reference-identity runs under the retained NVOS contract."""
import json
from pathlib import Path

root = Path('output/reference_referent_verification')
manifest = json.loads((root / 'manifest.json').read_text())
rows = []
for scene in manifest['tracks']:
    paths = {'baseline': Path('output/reference_integration/replay') / scene / 'evaluation'}
    paths.update({arm: root / arm / scene / 'evaluation' for arm in ['veto', 'affirmative']})
    rows.append(dict(scene=scene, arms={
        arm: json.loads((folder / 'evaluation.json').read_text())['metrics']
        for arm, folder in paths.items()}))
result = dict(
    scenes=rows,
    macro={arm: {metric: sum(row['arms'][arm][metric] for row in rows) / len(rows)
                 for metric in ['iou', 'average_precision']}
           for arm in ['baseline', 'veto', 'affirmative']},
    subset_diagnostics={arm: json.loads((root / arm / 'subset_projection_diagnostic.json').read_text())
                        for arm in ['veto', 'affirmative']},
    verification=json.loads((root / 'verification.json').read_text()),
    mismatch_diagnostic=json.loads((root / 'mismatched_judgments/shard_0.json').read_text()),
    signed_prompt_protocol=False,
    source_refit_complete=True,
)
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['macro'], indent=2))
