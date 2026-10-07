"""Diagnose source exposure of inferred and reliable supports; never fit here."""
import argparse
import json
from pathlib import Path

import torch

from .observations import training_episode
from .storage import load_field, sha256


def compile_exposure(root, observed):
    """Reconstruct train-only exposure from frozen Gaussian contributions."""
    root=Path(root);manifest=json.loads((root/'observations.json').read_text())
    count=len(observed);mass=torch.zeros(count);eligible=torch.zeros(count,dtype=torch.bool)
    dominant=torch.zeros(count,dtype=torch.bool)
    for entry in manifest['frames']:
        if entry['split']!='train':continue
        path=root/f"view_{entry['frame']:05d}.pt"
        if sha256(path)!=entry['compiled']['sha256']:raise ValueError('training observation changed')
        view=torch.load(path,map_location='cpu',weights_only=True)
        operator=view['operator'].coalesce();pixels,rows=operator.indices()
        mass.scatter_add_(0,rows,operator.values()*view['raw_mass'][pixels])
        dominant[view['best_element'][view['best_element']>=0]]=True
        for index in range(len(view['masks'])):
            episode=training_episode(view,index,observed)
            if episode is not None:
                eligible[view['best_element'][episode[2]]]=True
    return dict(mass=mass,eligible=eligible,dominant=dominant)


@torch.no_grad()
def run(config_path,output,device='cuda'):
    config=json.loads(Path(config_path).read_text());output=Path(output);output.mkdir(parents=True,exist_ok=False)
    field,manifest=load_field(config['field'],device)
    exposure=compile_exposure(config['observations'],field.observed.cpu())
    rows=[]
    if config.get('reference_prediction'):
        previous=Path(config['reference_prediction'])
        receipt=json.loads((previous/'prediction.json').read_text())
        if receipt['field_sha256']!=manifest['files']['field.pt']['sha256']:
            raise ValueError('reference uses a different field')
        if sha256(previous/'predictions.pt')!=receipt['predictions_sha256']:raise ValueError('prediction changed')
        data=torch.load(previous/'predictions.pt',map_location='cpu',weights_only=True)
        for text,vector in zip(data['texts'],data['native_query_vectors']):
            identity=field.semantic.score(vector.to(device)).flatten()
            ids=field._candidates(identity).cpu()
            rows.append(dict(text=text,anchors=ids.tolist(),eligible=exposure['eligible'][ids].tolist(),
                dominant=exposure['dominant'][ids].tolist(),mass=exposure['mass'][ids].tolist()))
    def stats(ids):
        return dict(count=len(ids),eligible_fraction=float(exposure['eligible'][ids].float().mean()),
                    dominant_fraction=float(exposure['dominant'][ids].float().mean()),
                    median_mass=float(exposure['mass'][ids].median()),mean_mass=float(exposure['mass'][ids].mean()))
    sets=dict(observed=torch.where(field.observed.cpu())[0],training_eligible=torch.where(exposure['eligible'])[0])
    if rows:
        sets.update(text_first=torch.tensor([row['anchors'][0] for row in rows]),
                    text_all=torch.tensor([i for row in rows for i in row['anchors']]))
    torch.save(exposure,output/'training_exposure.pt')
    report=dict(config=config,field_sha256=manifest['files']['field.pt']['sha256'],
        observations_sha256=sha256(Path(config['observations'])/'observations.json'),
        exposure_sha256=sha256(output/'training_exposure.pt'),summary={key:stats(ids) for key,ids in sets.items()},rows=rows,
        training_eligibility_not_actual_sample_visits=True,benchmark_labels_opened=False,fit_performed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();torch.set_num_threads(4);run(a.config,a.output,a.device)
