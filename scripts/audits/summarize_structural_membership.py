"""Read fixed-protocol diagnostics; reject a structural readout that loses ranking."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    artifacts=Path('paper/artifacts')
    root=Path('/mnt/pool/sqy/results/RADIO-GS/output/component_retrieval_pipeline_20260915')
    result={'schema':'structural_membership_acceptance.v1','benchmark':'LERF annotated development; not independent final test','scenes':{}}
    for scene,steps in [('figurines',2106),('ramen',3240)]:
        paths={key:artifacts/f'{scene}_{name}_20260916.json' for key,name in [('text','structural_membership_text'),('reference','structural_membership_reference'),('parent_text','partitioned_system_text'),('parent_reference','partitioned_system_reference')]}
        records={key:json.loads(path.read_text()) for key,path in paths.items()}
        parent=f'partition96_protected_{steps}'
        rows={}
        for run in ('structural_membership_v1','structural_membership_v1_semantic_only'):
            report=json.loads((root/scene/run/'report.json').read_text())
            assert sha256_file(root/scene/run/'field.pt')==report['field_sha256']
            text=records['text']['runs'][run];reference=records['reference']['runs'][run]['macro']
            baseline=records['parent_text']['runs'][parent]
            baseline_reference=records['parent_reference']['runs'][parent]['macro']
            calibrated=records['parent_reference']['runs'][parent+'_imagecal_balanced']['macro']
            gates={'native_state_exact':report['all_parent_state_exactly_preserved'] and report['native_cosine_max_error']==0,
                   'cold_load_exact':report['cold_load_max_error']==0,
                   'text_ap_no_regression':text['average_precision_macro']>=baseline['average_precision_macro'],
                   'text_iou_no_regression':text['observation_miou']>=baseline['observation_miou'],
                   'image_ap_no_regression':reference['image']['average_precision']>=calibrated['image']['average_precision'],
                   'image_iou_no_regression':reference['image']['iou']>=calibrated['image']['iou'],
                   'prompt_unchanged':reference['prompt']==baseline_reference['prompt']}
            rows[run]={'gates':gates,'promote':all(gates.values()),'text':{'iou':text['observation_miou'],'ap':text['average_precision_macro']},'reference':reference}
        result['scenes'][scene]={'artifact_hashes':{str(p):sha256_file(p) for p in paths.values()},'runs':rows}
    result['promote']=all(row['promote'] for scene in result['scenes'].values() for row in scene['runs'].values())
    path=artifacts/'structural_membership_acceptance_20260916.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
