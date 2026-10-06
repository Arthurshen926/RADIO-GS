"""Label-assisted diagnosis of category recognition versus referring identity.

The evaluator's category name is an oracle input, not a legal replacement for
the user's expression. Nothing from this script enters automatic inference.
"""
import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text


@torch.inference_mode()
def main():
    torch.set_num_threads(4);root=Path('output/identity_bottleneck_20261005')
    for condition in ['baseline','unmasked','object_context']:
        complete=json.loads((root/(condition+'_system')/'predictions_complete.json').read_text())
        if complete['jobs']!=9:raise ValueError('Full automatic predictions must precede oracle category diagnosis')
    package=Path('output/scannet_uqis_9_v0_3_1_candidate/evaluator')
    data=json.loads((package/'targets.json').read_text())
    categories=sorted({t['raw_semantic_label'] for t in data['targets']})
    first=root/'maps'/data['scene_domains'][0]['scene_id']/'map.json'
    spec=json.loads(first.read_text());model,tokenizer=load_siglip_text(spec['native_encoder'],'cuda')
    tokens=tokenizer(categories,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda')
    encodings=F.normalize(model(**tokens).pooler_output.float(),dim=-1).cpu().numpy()
    columns=dict(zip(categories,range(len(categories))));cases=[]
    for domain in data['scene_domains']:
        scene=domain['scene_id'];folder=root/'maps'/scene
        descriptors=np.load(folder/'descriptors.npy').astype(np.float32)
        descriptors/=np.maximum(np.linalg.norm(descriptors,axis=1,keepdims=True),1e-8)
        scores=descriptors@encodings.T
        membership=sparse.load_npz(folder/'membership.npz').tocsc()
        area=np.asarray(membership.sum(0)).ravel();available=np.flatnonzero(area>0)
        labels=np.load(package/domain['mesh_instance_ids_path'])
        old=np.load(root/'baseline_system/uqis'/scene/'query/registered/identity_evidence.npz')
        original_columns={q:i for i,q in enumerate(old['query_ids'])}
        for target in data['targets']:
            if target['scene_id']!=scene:continue
            category=target['raw_semantic_label'];qid=target['queries']['text']
            seed=int(available[np.argmax(scores[available,columns[category]])])
            actual=int(available[np.argmax(old['identity_response'][available,original_columns[qid]])])
            same_class=[target['instance_id']]+target['same_class_distractor_instance_ids']
            row=dict(scene_id=scene,query_id=qid,oracle_category=category)
            for name,index in [('original_expression',actual),('oracle_category_only',seed)]:
                vertices=membership[:,index].indices
                row[name]=dict(seed=index,target_fraction=float(np.mean(labels[vertices]==target['instance_id'])),
                    category_fraction=float(np.mean(np.isin(labels[vertices],same_class))))
            cases.append(row)
    summary={name:dict(category_majority=sum(r[name]['category_fraction']>.5 for r in cases),
        target_majority=sum(r[name]['target_fraction']>.5 for r in cases),queries=len(cases))
        for name in ['original_expression','oracle_category_only']}
    result=dict(automatic_method_result=False,category_names_from_evaluator=True,
        query_inputs_changed_for_diagnosis_only=True,summary=summary,cases=cases)
    (root/'category_confound.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
