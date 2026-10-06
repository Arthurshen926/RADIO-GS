"""Plot all four scenes under unchanged output contracts; no query selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.utils.immutable_artifacts import sha256_file

root = Path('output/full_field_rebuild_next19')
source = root / 'paired_comparison.json'
data = json.loads(source.read_text())
assert data['complete'] and len(data['scenes']) == 4
scenes = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
labels = ['Figurines\n16 → 20 views', 'Ramen\n16 → 25 views',
          'Teatime\n16 → 23 views', 'Waldo kitchen\n16 → 22 views']
metrics = [('iou', 'LERF2D IoU'), ('ap', 'LERF2D AP'),
           ('selected_iou', 'Selected-render IoU')]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
x = np.arange(4)
for ax, (metric, title) in zip(axes, metrics):
    before = np.array([data['scenes'][s]['arms']['final']['before'][metric] for s in scenes]) * 100
    after = np.array([data['scenes'][s]['arms']['final']['after'][metric] for s in scenes]) * 100
    ax.bar(x - .18, before, .36, color='#A6ABB4', label='Previous field')
    ax.bar(x + .18, after, .36, color='#2573B8', label='Rebuilt field')
    for i, delta in enumerate(after - before):
        ax.text(i, max(before[i], after[i]) + 2, f'{delta:+.2f}', ha='center', fontsize=9,
                color='#176437' if delta >= 0 else '#AE3030')
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Percent; labels show change in percentage points')
axes[1].legend(loc='upper center', fontsize=9)
fig.suptitle('Complete semantic-field reconstruction: all four scenes, same geometry and scoring', fontsize=13)
fig.text(.5, .025, 'Expanded source and training budgets; development evaluation. Selected-render IoU is not volumetric 3D IoU.',
         ha='center', fontsize=10)
fig.tight_layout(rect=(0, .065, 1, .95))
output = root / 'paired_comparison.png'
assert not output.exists(), output
fig.savefig(output, dpi=160)
(root / 'paired_comparison_plot.json').write_text(json.dumps({
    'source_sha256': sha256_file(source), 'plot_sha256': sha256_file(output),
    'implementation_sha256': sha256_file(__file__), 'all_scenes': scenes,
    'arm': 'final', 'no_per_query_or_scene_selection': True,
}, indent=2) + '\n')
print(output)
