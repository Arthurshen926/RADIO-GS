"""Compile all physical scope hypotheses from an existing frozen map."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from radio_gs.querying.observation_hypotheses import compile_hypotheses, compile_tracks


def compile_map(scene, output, tracks=False):
    started=time.monotonic();source=scene.parent;spec=json.loads(scene.read_text())
    output.mkdir(parents=True,exist_ok=True)
    membership=sparse.load_npz(source/'membership.npz')
    visibility=sparse.load_npz(source/'visibility.npz')
    frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(source/'region_offsets.npy')))
    hypotheses=compile_hypotheses(membership,visibility,frames)
    for name in spec['map_files']:os.link(source/name,output/name)
    sparse.save_npz(output/'scope_hypotheses.npz',hypotheses)
    count=np.diff(hypotheses.indptr)
    report=dict(hypotheses=len(count),supported_hypotheses=int((count>0).sum()),
        multiview_hypotheses=int((count>1).sum()),mean_views=float(count.mean()),
        matched_observations=int(hypotheses.nnz),maximum_regions=None,
        query_independent=True,labels_read=False,query_manifest_read=False,
        source_rgb_read=False,new_teacher_calls=0,original_map_arrays_unchanged=True,
        rule='all region hypotheses; one physical correspondence per other view; no transitive semantic merge',
        identity_rule='weighted expected native cosine across physically corresponding observations',
        seconds=time.monotonic()-started)
    spec.update(parent_map=str(scene),scope_hypotheses='scope_hypotheses.npz',
        scope_hypotheses_build=report,map_files=spec['map_files']+['scope_hypotheses.npz'])
    if tracks:
        matrix,track_report=compile_tracks(hypotheses,frames)
        sparse.save_npz(output/'instance_tracks.npz',matrix)
        spec.update(instance_tracks='instance_tracks.npz',instance_tracks_build=track_report)
        spec['map_files'].append('instance_tracks.npz')
        report['instance_tracks']=track_report
        report['seconds']=time.monotonic()-started
    spec['map_bytes']=sum((output/name).stat().st_size for name in spec['map_files'])
    (output/'map.json').write_text(json.dumps(spec,indent=2)+'\n')
    (output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    print(spec['scene_id'],report,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tracks',action='store_true')
    a=p.parse_args();compile_map(a.scene,a.output,a.tracks)


if __name__=='__main__':main()
