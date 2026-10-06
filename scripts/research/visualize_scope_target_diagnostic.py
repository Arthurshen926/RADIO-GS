"""Post-evaluation target-label visualization; never an inference input."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radio_gs.scripts.eval_prerendered_lerf_features import load_lerf_objects
from radio_gs.scripts.eval_opengaussian_lerf_baseline import SCENE_GT_FRAMES

root = Path('output/containment_scope_control/target_diagnostic')
root.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
records = []
row = 0
for scene, query in [('teatime', 'coffee'), ('figurines', 'rubber duck with buoy')]:
    directory = Path('/mnt/pool/sqy/3d_understanding/lerf_ovs/label') / scene
    frames = load_lerf_objects(directory, frames=SCENE_GT_FRAMES[scene])
    admitted = [(frame, obj) for frame, objs in frames.items() for obj in objs if obj.query == query]
    for frame, obj in [admitted[0], admitted[-1]]:
        rgb = Image.open(directory / (frame + '.jpg')).convert('RGB')
        axes[row, 0].imshow(rgb)
        axes[row, 0].contour(obj.mask, levels=[.5], colors=['lime'], linewidths=1)
        axes[row, 0].set_title(f'{scene}: {query}, {frame}, GT green')
        prediction = Image.open(Path('output/consolidated_response_control') / scene /
                                'fusion/native_protected/evaluation/hierarchical_pipeline' /
                                (frame + '_' + query + '.png'))
        axes[row, 1].imshow(rgb)
        axes[row, 1].contour(np.asarray(prediction) > 10, levels=[.5], colors=['red'], linewidths=1)
        axes[row, 1].set_title('Default actual selected subset, red; diagnostic only')
        for ax in axes[row]:
            ax.axis('off')
        records.append(dict(scene=scene, query=query, frame=frame))
        row += 1
fig.tight_layout()
fig.savefig(root / 'target_scopes.jpg', dpi=110)
(root / 'diagnostic.json').write_text(json.dumps(dict(
    records=records, target_labels_used=True, purpose='post-evaluation failure interpretation only',
    inference_or_training_input=False, predictions_generated_before_diagnostic=True), indent=2) + '\n')
