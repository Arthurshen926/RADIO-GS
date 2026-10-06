"""Two predetermined independent rooms: real RGB-D vs old point render inputs.

The query-independent map uses the exact deployed observation compiler. These
are training/development rooms, never benchmark rooms. No target is added to
the map. All finite source poses are eligible, with the deployed stride of20.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

import numpy as np

from radio_gs.benchmarks.uqis.sensor import SensorIndex


def main():
    root=Path('output/referent_grounding_20261005/real_observation_control')
    root.mkdir(parents=True,exist_ok=True)
    scenes=['scene0271_00','scene0540_00']
    design=dict(scenes=scenes,selection='first room in pre-existing fit and validation splits',
        arms=['old 12 synthetic point views','12 uniform real observation frames','all stride20 real observation frames'],
        model_fits=0,geometry='same official mesh vertices',encoders='same SigLIP and SAM3',
        role='training observation diagnosis; no benchmark query, label, or model selection',
        source_frame_limit=None,
        comparison_limits='Twelve-view control matches view count, not camera poses. Real appearance and camera distribution both change; full also adds coverage.')
    (root/'design.json').write_text(json.dumps(design,indent=2)+'\n')
    def download(scene):
        folder=root/'raw'/scene;folder.mkdir(parents=True,exist_ok=True)
        file=folder/(scene+'.sens')
        if not file.exists():
            temporary=file.with_suffix('.partial')
            urllib.request.urlretrieve(f'https://kaldir.vc.in.tum.de/scannet/v1/scans/{scene}/{scene}.sens',temporary)
            temporary.rename(file)
        sensor=SensorIndex(file);frames,_=sensor.exclude([])
        print(scene,'sensor ready',len(frames),'finite poses',file.stat().st_size,flush=True)
        return dict(scene_id=scene,sens_path=str(file.resolve()),field_frame_ids=frames.tolist())
    with ThreadPoolExecutor(2) as pool:mapping=list(pool.map(download,scenes))
    for folder in ['mapping','method']:(root/folder).mkdir(exist_ok=True)
    (root/'mapping/scenes.json').write_text(json.dumps(dict(scenes=mapping),indent=2)+'\n')
    domains=[dict(scene_id=s,mesh_xyz_path=str(Path('output/learned_instance_additions/maps',s,'mesh_xyz.npy').resolve())) for s in scenes]
    (root/'method/scenes.json').write_text(json.dumps(dict(scene_domains=domains),indent=2)+'\n')
    config=json.loads(Path('output/uqis_map_only_v1/maps/scene0030_00/map.json').read_text())['configuration']
    config.update(name='independent real observation grounding control',benchmark_package=str(root),
        no_benchmark_scenes=True,maximum_source_frames=None)
    (root/'config.json').write_text(json.dumps(config,indent=2)+'\n')


if __name__=='__main__':main()
