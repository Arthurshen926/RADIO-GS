"""Coverage-domain audit of fixed predictions; no target labels or new tuning."""
import argparse
from pathlib import Path
import torch
from radio_gs.query_field.storage import load_field
from scripts.research.run_query_field_joint_exclusion import ROOT,read,write


def run(index, initial_arm="finite"):
    read(ROOT/'predictions_complete.json');job=read(ROOT/'prediction_config.json')['jobs'][index]
    field,manifest=load_field(job['field']);data=torch.load(ROOT/'predictions'/job['name']/'predictions.pt',weights_only=True)
    touched=torch.zeros(len(field.xyz),dtype=torch.bool)
    for v in field.coverage_factors.views:touched[v.index]=True
    available=field.observed|field.semantic_measure.supported
    assert torch.equal(available,data['observed']);rows=[]
    for qi,text in enumerate(data['texts']):
        initial=data['selected_by_control'][initial_arm][:,qi];selected=data['selected_by_control']['joint_coverage'][:,qi]
        removed=initial&~selected
        assert not bool((removed&~touched).any())
        rows.append(dict(text=text,initial_selected=int(initial.sum()),initial_observed=int((initial&field.observed).sum()),
            initial_semantic_only=int((initial&~field.observed&available).sum()),initial_factor_touched=int((initial&touched).sum()),
            removed=int(removed.sum()),removed_observed=int((removed&field.observed).sum()),
            removed_semantic_only=int((removed&~field.observed&available).sum()),
            remaining_untouched=int((selected&~touched).sum())))
    report=dict(scene=job['scene'],field_sha256=manifest['files']['field.pt']['sha256'],gaussians=len(field.xyz),
        observed=int(field.observed.sum()),available=int(available.sum()),factor_touched=int(touched.sum()),
        rows=rows,source_images_opened=False,target_labels_opened=False,domain_limit_not_proof_of_error=True)
    write(ROOT/'diagnostics'/('domain_'+job['scene']+'.json'),report)
    print({k:v for k,v in report.items() if k!='rows'},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
