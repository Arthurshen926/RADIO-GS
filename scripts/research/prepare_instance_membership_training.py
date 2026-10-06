"""Index real instance supervision outside every maintained evaluation room.

This prepares an additional supervised research condition; it neither trains a
model nor changes the current zero-target-label method. Repeated scans of one
room always share a partition. ScanNet whole-instance IDs are not part labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, default=Path('configs/pipeline/system_v8.json'))
    p.add_argument('--data', type=Path, default=Path('/mnt/pool/sqy/3d_understanding/agile3d_data/ScanNet'))
    p.add_argument('--official-train', type=Path, default=Path('/mnt/pool/sqy/3d_understanding/ScanNet-UQIS-10/source/scanrefer-code/data/scannet/meta_data/scannetv2_train.txt'))
    p.add_argument('--output', type=Path, default=Path('output/independent_instance_membership'))
    args = p.parse_args()
    jobs = json.loads(args.recipe.read_text())['jobs']
    excluded = sorted({j['name'].split('_')[0] for j in jobs if j['benchmark'] in ['scannet', 'uqis']})
    scenes = sorted(s for s in args.official_train.read_text().split()
                    if s.split('_')[0] not in excluded and (args.data / 'scans' / f'{s}.ply').is_file())
    rooms = sorted({s.split('_')[0] for s in scenes},
                   key=lambda s: hashlib.sha256(('whole-instance-membership-v1/' + s).encode()).hexdigest())
    validation_rooms = set(rooms[:len(rooms) // 5])
    partitions = {name: [s for s in scenes if (s.split('_')[0] in validation_rooms) == heldout]
                  for name, heldout in [('fit', False), ('mechanism_validation', True)]}
    args.output.mkdir(parents=True, exist_ok=True)
    split = dict(data_root=str(args.data), official_training_list=str(args.official_train),
        benchmark_recipe=str(args.recipe), excluded_room_prefixes=excluded, partitions=partitions,
        policy='all available official-training scans; room-grouped fixed hash 80/20; no benchmark-room rescan',
        supervision='additional 3D whole-instance labels, separate from zero-target-label v8',
        not_provided=['part hierarchy', 'per-instance semantic class mapping',
                      'region teacher maps in these rooms', 'query image crops', 'language referent labels'],
        model_trained=False, current_method_changed=False, final_benchmark_labels_read=False)
    # Fix room assignments before inspecting their labels.
    (args.output / 'split.json').write_text(json.dumps(split, indent=2) + '\n')

    def inventory(scene):
        path = args.data / 'scans' / f'{scene}.ply'
        vertex = PlyData.read(path)['vertex'].data
        labels = np.asarray(vertex['label'], dtype=np.int64)
        instance_ids, counts = np.unique(labels[labels >= 0], return_counts=True)
        return dict(scene=scene, path=str(path), points=len(vertex),
                    labelled_points=int((labels >= 0).sum()), unknown_points=int((labels < 0).sum()),
                    instances=[dict(instance_id=int(i), points=int(n)) for i, n in zip(instance_ids, counts)])

    with ThreadPoolExecutor(4) as pool:
        rows = list(pool.map(inventory, scenes))
    by_scene = {r['scene']: r for r in rows}
    counts = {name: dict(scans=len(names), rooms=len({s.split('_')[0] for s in names}),
        instances=sum(len(by_scene[s]['instances']) for s in names),
        points=sum(by_scene[s]['points'] for s in names),
        unknown_points=sum(by_scene[s]['unknown_points'] for s in names)) for name, names in partitions.items()}
    report = dict(**split, counts=counts, scenes=rows,
        label_contract='same nonnegative instance ID within a scan is positive; different labelled IDs are negative; -1 remains unknown',
        source='existing AGILE3D ScanNet PLY instance IDs; labels are training-only assets, never query-map inputs')
    (args.output / 'inventory.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main()
