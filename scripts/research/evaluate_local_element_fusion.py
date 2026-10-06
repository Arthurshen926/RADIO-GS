"""Fixed local/global and single/dual-native controls on reserved rooms."""
import argparse,hashlib,json,shutil
from pathlib import Path
from evaluate_independent_scope_reliability import evaluate_jobs

ROOT=Path('output/local_element_fusion')
SCENES=['scene0606_01','scene0629_01','scene0653_01']
MODES=['siglip_vote','adapter_vote','average_prior','local_single','local_dual','global_dual']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');args=parser.parse_args()
    if args.development:
        parents={n:Path('output/independent_source_reliability') for n in ['scene0050_02','scene0231_01','scene0412_00']}
        parents.update({n:Path('output/element_retrieval_reliability') for n in ['scene0575_00','scene0591_02','scene0599_02']})
        output=ROOT/'development';output.mkdir(exist_ok=True)
        shutil.copyfile(ROOT/'scannetv2-labels.combined.tsv',output/'scannetv2-labels.combined.tsv')
        split=json.loads((ROOT/'split.json').read_text());split['validation']=list(parents)
        split['validation_policy']='previously opened development rooms; models already frozen; no parameter selection'
        (output/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    else:parents={n:ROOT for n in SCENES};output=ROOT
    models=[ROOT/f'{n}.pt' for n in ['local_single','local_dual','global_dual']]
    (output/'frozen_models.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in models},indent=2)+'\n')
    jobs=[]
    for name,parent in parents.items():
        if args.development:shutil.copyfile(parent/f'{name}.queries.json',output/f'{name}.queries.json')
        views=json.loads((parent/'evidence'/name/'views_0.json').read_text())['views']
        auxiliary=parent/'maps'/name/'adapter/mask_adapter/scene.json'
        for mode in MODES:
            source=auxiliary if mode=='adapter_vote' else parent/'maps'/name/'compiled/scene.json'
            scene=json.loads(source.read_text());scene['native_observation_reliability']='entropy_information'
            scene.pop('observation_weighting',None);scene['source_streaming']=True;scene['source_evidence']=dict(views=views,raster_hw=[96,128])
            if mode=='average_prior':scene['local_element_control']='average_prior'
            if mode in ['local_single','local_dual','global_dual']:
                scene['local_element_checkpoint']=str(ROOT/f'{mode}.pt');scene['map_contract']['supervised_local_readout']=True
            if mode in ['average_prior','local_dual','global_dual']:scene['local_auxiliary_scene']=str(auxiliary)
            dest=output/'configs'/mode/f'{name}.scene.json';dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(scene,indent=2)+'\n')
            jobs.append((mode,name,dest))
    evaluate_jobs(output,list(parents),jobs)
    if args.development:
        path=output/'summary.json';summary=json.loads(path.read_text());summary['new_scene_for_this_mechanism']=False
        summary['previously_opened_development_rooms']=True;path.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
