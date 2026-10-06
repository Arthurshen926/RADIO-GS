"""Compile query-free third-view scope evidence on an existing map."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from radio_gs.querying.witnessed_observation_scope import compile_witnesses


def compile_map(source, output):
    started = time.monotonic()
    torch.set_num_threads(4)
    spec = json.loads(source.read_text())
    root = source.parent
    membership = sparse.load_npz(root/'membership.npz')
    visibility = sparse.load_npz(root/'visibility.npz')
    frames = np.repeat(np.arange(visibility.shape[1]), np.diff(np.load(root/'region_offsets.npy')))
    graph, report = compile_witnesses(membership, visibility, frames)
    output.mkdir(parents=True, exist_ok=True)
    for name in spec['map_files']:
        os.link(root/name, output/name)
    sparse.save_npz(output/'scope_witnesses.npz', graph)
    report.update(seconds=time.monotonic()-started, source=str(source),
        query_read=False, labels_read=False, source_images_added=0,
        geometry_changed=False, native_features_changed=False)
    spec.update(scope_witnesses='scope_witnesses.npz', scope_witness_build=report,
        map_files=spec['map_files']+['scope_witnesses.npz'])
    spec['map_bytes'] = sum((output/name).stat().st_size for name in spec['map_files'])
    (output/'map.json').write_text(json.dumps(spec, indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    compile_map(args.scene, args.output)


if __name__ == '__main__':
    main()
