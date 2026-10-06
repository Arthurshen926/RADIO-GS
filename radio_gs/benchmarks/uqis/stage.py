"""Deliver one query and its public mesh domain, without source images or GT."""
import argparse
import json
from pathlib import Path
import shutil

from .construct import write_json


def stage(package: Path, modality: str, query_id: str, output: Path):
    public=package/'method'
    rows=json.loads((public/'queries'/f'{modality}.json').read_text())['queries']
    query=next(q for q in rows if q['query_id']==query_id)
    domains=json.loads((public/'scenes.json').read_text())['scene_domains']
    domain=next(d for d in domains if d['scene_id']==query['scene_id'])
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(public/domain['mesh_xyz_path'],output/'mesh_xyz.npy')
    query=dict(query)
    if modality=='image':
        shutil.copyfile(public/query['crop_rgb_path'],output/'reference.png')
        query['crop_rgb_path']='reference.png'
    write_json(output/'query.json',query)
    write_json(output/'scene.json',{**domain,'mesh_xyz_path':'mesh_xyz.npy'})
    return query


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--modality',choices=['text','image','point_2d','point_3d'],required=True)
    p.add_argument('--query-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(stage(a.package,a.modality,a.query_id,a.output),indent=2))


if __name__=='__main__':
    main()
