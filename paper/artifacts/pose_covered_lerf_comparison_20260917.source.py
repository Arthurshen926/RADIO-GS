"""Compare complete recipes under identical evaluation contracts and cohorts."""
import argparse,json,shutil
from pathlib import Path
from statistics import mean
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    base=Path('output');result=dict(schema='pose_covered_lerf_comparison.v1',independent_final_test=False,architecture_only_ablation=False,source_inputs_and_training_budget_expanded=True,scenes={},implementation_sha256=sha256_file(__file__))
    name='joint_supervised_image_residual'
    for scene in ('figurines','ramen'):
        old=base/'lerf_ropefix_multiscale_pipeline_20260917'/scene;new=base/'lerf_pose_covered_pipeline_20260917'/scene
        assert json.loads((new/'execution.json').read_text()).get('finished')
        row=dict(inputs={},metrics={},per_view_text={},field_sha256={},native_decomposition={})
        for task in ('text','reference'):
            paths=[old/'scoped_joint_image_validation'/(task+'.json'),new/(task+'_evaluation.json')]
            xs=[json.loads(p.read_text()) for p in paths]
            protocols=[]
            for root,path,x in zip((old,new),paths,xs):
                row['inputs'][str(path)]=sha256_file(path)
                assert sha256_file(root/'field'/name/'field.pt')==x['runs'][name]['field_sha256']
                protocol=x['protocol'].copy()
                for k in ('fields','field_sha256','semantic_train_frames','source_semantic_train_frames'):protocol.pop(k,None)
                protocols.append(protocol)
                dispatch=json.loads(path.with_suffix('.dispatch.json').read_text())
                assert dispatch['result_sha256']==sha256_file(path)
                assert dispatch['protocol_sha256']==sha256_file(path.with_suffix('.protocol.json'))
            assert protocols[0]==protocols[1],(scene,task,'evaluation protocol differs')
            xs=[x['runs'][name] for x in xs]
            row['field_sha256']={'before':xs[0]['field_sha256'],'after':xs[1]['field_sha256']}
            keys=('frame','category') if task=='text' else ('reference_frame','target_frame','category')
            ids=[{tuple(o[k] for k in keys) for o in x['observations']} for x in xs];assert ids[0]==ids[1]
            if task=='text':
                for key in ('observation_miou','average_precision_macro','positive_support_recall_macro'):
                    row['metrics']['full_text_'+key]={'before':xs[0][key],'after':xs[1][key],'delta':xs[1][key]-xs[0][key]}
                for frame in sorted({o['frame'] for o in xs[0]['observations']}):
                    row['per_view_text'][str(frame)]={label:{k:mean(o[k] for o in x['observations'] if o['frame']==frame) for k in ('iou','average_precision')} for label,x in zip(('before','after'),xs)}
            else:
                assert xs[0]['tasks_per_modality']==xs[1]['tasks_per_modality']
                row['unavailable_prompt_tasks']={'before':xs[0]['unavailable_prompt_tasks'],'after':xs[1]['unavailable_prompt_tasks']}
                for mode in ('text','image','prompt'):
                    for k in ('iou','average_precision'):
                        u,v=[x['macro'][mode][k] for x in xs];row['metrics']['reference_'+mode+'_'+k]={'before':u,'after':v,'delta':v-u}
        for filename in ('native_compression','source_semantic_stages'):
            path=new/'diagnostics'/(filename+'.json');d=json.loads(path.read_text());row['inputs'][str(path)]=sha256_file(path);row['native_decomposition'][filename]=d['macro']
        result['scenes'][scene]=row
    a.output.write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(__file__,a.output.with_suffix('.source.py'));print(json.dumps({s:r['metrics'] for s,r in result['scenes'].items()},indent=2))
if __name__=='__main__':main()
