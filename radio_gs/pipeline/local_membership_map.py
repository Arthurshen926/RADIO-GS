"""Compile local native membership on mesh and Gaussian observation maps."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse
import torch

from radio_gs.querying.local_observation_membership import local_membership


def write(path, value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def registered(scene_path, output):
    source=scene_path.parent;spec=json.loads(scene_path.read_text())
    transfer=sparse.load_npz(source/'transfer.npz')
    updated,relative,report=local_membership(transfer,np.load(source/'descriptors.npy').astype(np.float32),
        np.load(source/'region_offsets.npy'))
    for mode in ['transfer','scope']:
        folder=output/mode;folder.mkdir(parents=True,exist_ok=True)
        for filename in spec['map_files']:
            if filename!='transfer.npz':os.link(source/filename,folder/filename)
        sparse.save_npz(folder/'transfer.npz',updated)
        current=dict(spec,local_membership=report,parent_map=str(scene_path),
            map_files=list(spec['map_files']))
        if mode=='scope':
            refined=sparse.load_npz(source/'membership.npz').multiply(relative).tocsr()
            sparse.save_npz(folder/'local_membership.npz',refined)
            current['positive_membership']='local_membership.npz'
            current['map_files'].append('local_membership.npz')
        current['map_bytes']=sum((folder/f).stat().st_size for f in current['map_files'])
        write(folder/'map.json',current)
    write(output/'build.json',report)


def categorical(scene_path, output, offsets):
    spec=json.loads(scene_path.read_text());output.mkdir(parents=True,exist_ok=True)
    field=torch.load(spec['native_observation_field'],map_location='cpu',weights_only=False)
    transfer=sparse.csr_matrix((field['weights'].float().numpy(),field['columns'].numpy(),field['crow'].numpy()),shape=field['shape'])
    updated,_,report=local_membership(transfer,field['descriptors'].float().numpy(),offsets)
    field.update(crow=torch.from_numpy(updated.indptr),columns=torch.from_numpy(updated.indices),
        weights=torch.from_numpy(updated.data),region_offsets=torch.from_numpy(offsets),
        source_frames=spec['source_frames'],local_membership=report)
    torch.save(field,output/'observation_field.pt')
    spec['native_observation_field']=str(output/'observation_field.pt')
    spec['local_membership']=report
    if spec.get('local_auxiliary_scene'):
        categorical(Path(spec['local_auxiliary_scene']),output/'auxiliary',offsets)
        spec['local_auxiliary_scene']=str(output/'auxiliary/scene.json')
    write(output/'scene.json',spec);write(output/'build.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--offsets',type=Path,help='Explicit source-view region provenance for older compiled category maps')
    a=p.parse_args()
    torch.set_num_threads(4);spec=json.loads(a.scene.read_text())
    if spec.get('schema')=='registered_native_observation_map_v1':registered(a.scene,a.output)
    else:categorical(a.scene,a.output,np.load(a.offsets))


if __name__=='__main__':main()
