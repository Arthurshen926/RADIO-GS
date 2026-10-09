"""Verify full UQIS legacy evidence can address the frozen Gaussian rows."""
from pathlib import Path

import numpy as np
import torch

from scripts.research.run_query_field_confidence_mixture_functions import ROOT, read, write, receipt, check


def main():
    design = read(ROOT / 'design.json'); legacy = read('configs/pipeline/system_v10.json')
    records = []
    for job in design['fields']:
        if job['family'] != 'uqis': continue
        scene = job['scene']; source = read(Path(job['observations']) / 'observations.json')
        check(source['geometry'])
        geometry = torch.load(source['geometry']['path'], map_location='cpu', weights_only=True)
        xyz = geometry['state_dict']['xyz'].numpy()
        old = next(j for j in legacy['jobs'] if j['benchmark'] == 'uqis' and j['name'] == scene)
        old_path = Path(old['scene']); old_manifest = read(old_path)
        mesh_path = old_path.parent / 'mesh_xyz.npy'; mesh = np.load(mesh_path)
        ids = geometry['source_vertex_ids'].numpy()
        exact = xyz.shape == mesh.shape and np.array_equal(xyz, mesh)
        mapped = bool((ids >= 0).all() and (ids < len(mesh)).all() and np.array_equal(xyz, mesh[ids]))
        records.append(dict(scene=scene, gaussian_rows=len(xyz), v10_rows=len(mesh),
            exact_direct_row_coordinates=exact, exact_source_vertex_mapping=mapped,
            source_vertex_ids_unique=len(np.unique(ids)) == len(ids),
            v10_source_views=old_manifest['source_count'], current_training_views=len(source['splits']['train_frames']),
            current_geometry=source['geometry'], v10_manifest=receipt(old_path), v10_coordinates=receipt(mesh_path)))
    if len(records) != 9: raise ValueError('UQIS carrier cohort incomplete')
    write(ROOT / 'diagnostics/uqis_carrier_correspondence.json', dict(records=records,
        code=receipt(Path(__file__)), exact_direct_scenes=sum(r['exact_direct_row_coordinates'] for r in records),
        exact_mapped_scenes=sum(r['exact_source_vertex_mapping'] for r in records),
        benchmark_labels_accessed=False, model_change=False, source_evidence_transferred=False,
        interpretation='Geometric feasibility only: does not establish teacher/split equivalence or task performance'))
    print([(r['scene'], r['exact_direct_row_coordinates'], r['exact_source_vertex_mapping']) for r in records], flush=True)


if __name__ == '__main__': main()
