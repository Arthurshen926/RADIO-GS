"""Explicit reference-instance runtime; evaluation is a separate command."""
import argparse,json,os,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--query',required=True);p.add_argument('--output',required=True);p.add_argument('--gpu',default='0');p.add_argument('--tracks',help='Explicit source-evidence replay; omit to rerun tracking');a=p.parse_args()
    scene=json.loads(Path(a.scene).read_text());query=json.loads(Path(a.query).read_text())
    if query['intent']!='reference_instance':raise ValueError('This provider requires an explicit reference-instance anchor')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,PYTHONPATH='.',CUDA_VISIBLE_DEVICES=a.gpu,OMP_NUM_THREADS='4')
    py='/root/miniconda3/envs/cybersim_agent/bin/python';sam='/root/miniconda3/envs/sam3/bin/python';tracks=a.tracks
    commands=[]
    query_path=a.query
    if query.get('modality') in ('image','prompt_3d','prompt_2d') and 'reference_rgb' not in query:
        query_path=str(out/'resolved/query.json')
        commands.append([py,'scripts/research/resolve_map_query.py','--scene',a.scene,'--query',a.query,'--output',str(out/'resolved')])
    if not tracks:
        commands.append([sam,'scripts/research/track_nvos_sources_next43.py','--scene',scene['scene'],'--scene-input',a.scene,'--query',query_path,'--all-legal-sources','--output',str(out/'tracks')]);tracks=str(out/'tracks'/scene['scene']/'tracks.pt')
    commands.append([py,'scripts/research/fit_reference_instance.py','--scene',a.scene,'--query',query_path,'--tracks',tracks,'--output',str(out)])
    commands.append([py,'scripts/research/decode_expanded_extent_next72.py','--inputs',str(out/'subset_inputs.json'),'--key',scene['scene'],'--output-base',str(out/'sets'),'--mode','admission','--scope-initialization','source','--pixel-loss','balanced','--support-prior','footprint','--steps','240','--boundary-radius',str(scene['boundary_radius'])])
    commands.append([py,'scripts/research/compose_reference_readout.py','--response-field',str(out/'field.pt'),'--selection-field',str(out/'sets'/scene['scene']/'typed_readout.pt'),'--output',str(out/'typed_readout.pt')])
    (out/'execution.json').write_text(json.dumps(dict(scene=a.scene,query=a.query,tracks_replayed=bool(a.tracks),reference_seed_mask_reused=bool(query.get("reference_mask")),commands=commands),indent=2)+'\n')
    for command in commands:subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
