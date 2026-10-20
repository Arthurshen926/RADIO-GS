"""Prepare independent complete-menu/union targets without duplicating labels.

This compiles a hashed target manifest, NOT a trained multi-task utility corpus.
The missing Mask-Adapter and literal-category evidence are explicit. Ground-truth
labels are used for training-target coverage only, never for choosing a teacher
response, a retained view split, or a benchmark prediction.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from plyfile import PlyData

from radio_gs.data.scannet_source_region_semantics import load_scannet_raw_to_nyu40, official_vertex_nyu40_labels
from radio_gs.scannet_constants import OPENGAUSSIAN_NYU40_CLASS_SPLITS as SPLITS, NYU40_ID_TO_NAME
from radio_gs.query_field.bounded_experiment import BoundedExperiment, tree_bytes
from scripts.research.replay_tmm_response import STUDY, POLICY, sha
from radio_gs.querying.response_study_adapters import read

SOURCE = Path('output/native_phrase_binding_20261006')
RAW = Path('output/referent_grounding_20261005/raw')
MAPPING = Path('output/independent_source_reliability/scannetv2-labels.combined.tsv')


def view_folds(frames):
    """Three chronological blocks; no annotations or query labels enter splits."""
    frames = np.asarray(frames, dtype=np.int64)
    if len(frames) < 6 or np.any(np.diff(frames) <= 0):
        raise ValueError('ordered unique source frames required')
    blocks = np.array_split(np.arange(len(frames)), 3)
    result = []
    for i, held in enumerate(blocks):
        keep = np.setdiff1d(np.arange(len(frames)), held)
        result.append(dict(fold=i, candidate_view_indices=keep.tolist(), evidence_view_indices=held.tolist(),
                           candidate_frames=frames[keep].tolist(), evidence_frames=frames[held].tolist()))
    return result


def compile_room(scene, split, raw_to_nyu40):
    folder = SOURCE/'maps'/scene; spec = read(folder/'map.json')
    annotation = RAW/scene
    segmentation = annotation/(scene+'_vh_clean_2.0.010000.segs.json')
    aggregation = annotation/(scene+'.aggregation.json')
    mesh = annotation/(scene+'_vh_clean_2.ply')
    xyz = np.load(folder/'mesh_xyz.npy')
    vertices = PlyData.read(mesh)['vertex'].data
    official_xyz = np.stack([vertices[k] for k in ('x', 'y', 'z')], 1)
    if not np.array_equal(xyz, official_xyz):
        raise ValueError('map carrier is not the exact official mesh vertex order')
    labels, audit = official_vertex_nyu40_labels(scene_id=scene, vertex_count=len(xyz),
        segmentation=read(segmentation), aggregation=read(aggregation), raw_to_nyu40=raw_to_nyu40)
    visibility = sparse.load_npz(folder/'visibility.npz').tocsc()
    membership = sparse.load_npz(folder/'membership.npz')
    offsets = np.load(folder/'region_offsets.npy')
    if visibility.shape != (len(xyz), len(spec['source_frames'])) or offsets[-1] != membership.shape[1]:
        raise ValueError('map observation carrier mismatch')
    folds = view_folds(spec['source_frames'])
    targets = []
    for class_id in SPLITS['19']:
        truth = labels == class_id
        local = visibility[truth]
        targets.append(dict(class_id=class_id, text=NYU40_ID_TO_NAME[class_id],
            vertices=int(truth.sum()), visible_vertices=int((local.getnnz(axis=1) > 0).sum()),
            views=int((local.getnnz(axis=0) > 0).sum()),
            target_kind='union of all instances of this category, never one representative instance'))
    for fold in folds:
        seen_candidate = visibility[:, fold['candidate_view_indices']].getnnz(axis=1) > 0
        seen_evidence = visibility[:, fold['evidence_view_indices']].getnnz(axis=1) > 0
        fold['class_coverage'] = {str(c): dict(
            target_vertices=int((labels == c).sum()),
            candidate_visible=int(((labels == c)&seen_candidate).sum()),
            evidence_visible=int(((labels == c)&seen_evidence).sum()),
            covisible=int(((labels == c)&seen_candidate&seen_evidence).sum())) for c in SPLITS['19']}
        fold['category_decision_unit'] = 'one complete 19-column response; all subset metrics share its index'
    paths = [folder/n for n in ('map.json', 'mesh_xyz.npy', 'membership.npz', 'visibility.npz',
                               'region_offsets.npy', 'descriptors.npy', 'transfer.npz')]
    paths += [segmentation, aggregation, mesh, MAPPING]
    return dict(scene=scene, split=split, vertices=len(xyz), source_views=len(spec['source_frames']),
        official_semantic_audit=audit, targets=targets, view_folds=folds,
        input_hashes={str(p): sha(p) for p in paths},
        target_labels_reconstructed_in_memory=True, dense_labels_saved=False,
        complete_menu_columns=SPLITS['19'], absent_classes_retained=True,
        available_native_teacher='original SigLIP regional descriptors',
        required_uncompiled_inputs=['same-region Mask-Adapter descriptors and original category text encoding',
                                    'literal-category SAM source evidence for all 19 classes',
                                    'complete candidates compiled without their held evidence views'],
        inherited_geometry='full official scan mesh; geometry not held out',
        view_split_limit='chronological blocks are disjoint, but boundary poses are not purged; no statistical independence claim',
        utility_training_ready=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='cross_task_targets_v1')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    design = read(SOURCE/'design.json'); splits = {k: design[k] for k in ('fit', 'calibration', 'validation')}
    names = sum(splits.values(), [])
    benchmark_names = {j['name'].split('_')[0] if j['name'].startswith('scene') else j['name']
                       for j in read('configs/pipeline/system_v10.json')['jobs']}
    if (len(set(names)) != len(names) or len({n.split('_')[0] for n in names}) != len(names)
            or {n.split('_')[0] for n in names}&benchmark_names):
        raise ValueError('training/calibration/development rooms overlap each other or benchmark room prefixes')
    sources = [__file__, 'radio_gs/data/scannet_source_region_semantics.py', 'radio_gs/scannet_constants.py']
    manifest = dict(splits=splits, source_hashes={p: sha(p) for p in sources},
        design_sha256=sha(SOURCE/'design.json'), mapping_sha256=sha(MAPPING),
        complete_menu=SPLITS['19'], all_splits=SPLITS,
        purpose='official independent complete-menu and all-instance-union targets; preparation only',
        no_benchmark_labels=True, no_new_feature_training=True, default_changed=False)
    run = BoundedExperiment(STUDY, args.run, POLICY, resume=args.resume); run.preflight(2*2**20)
    if args.resume:
        if read(run.root/'manifest.json') != manifest:
            raise ValueError('target preparation protocol changed')
    else:
        run.json('manifest.json', manifest)
        run.json('source_snapshot.json', {p: dict(text=Path(p).read_text(), sha256=sha(p)) for p in sources})
    raw_to_nyu40 = load_scannet_raw_to_nyu40(MAPPING)
    reports = []
    for split, scenes in splits.items():
        for scene in scenes:
            path = run.root/(scene+'.json')
            if path.exists():
                report = read(path)
                if any(sha(p) != h for p, h in report['input_hashes'].items()):
                    raise ValueError('retained independent inputs changed')
            else:
                report = compile_room(scene, split, raw_to_nyu40); run.json(scene+'.json', report)
            reports.append(report)
            print('targets', split, scene, sum(t['vertices'] > 0 for t in report['targets']), flush=True)
    if (run.root/'summary.json').exists():
        return
    coverage = {split: {str(c): sum(next(t['vertices'] for t in r['targets'] if t['class_id'] == c) > 0
        for r in reports if r['split'] == split) for c in SPLITS['19']} for split in splits}
    summary = dict(complete=True, target_manifests_ready=True, utility_training_ready=False,
        rooms=len(reports), complete_menu_targets=len(reports), cross_view_menu_episodes=3*len(reports),
        binary_union_targets=19*len(reports), room_counts={s: len(v) for s, v in splits.items()},
        class_room_coverage=coverage, report_hashes={r['scene']+'.json': sha(run.root/(r['scene']+'.json')) for r in reports},
        missing_fit_classes=[c for c, n in coverage['fit'].items() if not n],
        missing_calibration_classes=[c for c, n in coverage['calibration'].items() if not n],
        dense_artifact_bytes=0, study_bytes=tree_bytes(STUDY), independent_final_test=False,
        calibration_and_development_previously_opened=True, default_changed=False)
    run.json('summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    main()
