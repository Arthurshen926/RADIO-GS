"""Post-prediction feature-domain audit; never edits a query or model."""
import argparse
from pathlib import Path
import numpy as np
import torch
from radio_gs.query_field.storage import load_field
from radio_gs.query_field.part_conditioning import native_region_probability
from radio_gs.querying.conditional_region_membership import FEATURE_NAMES
from scripts.research.run_query_field_region_scope import ROOT,read,write,receipt


def run(index):
    source=Path('output/learned_instance_additions');split=read(source/'design.json')
    x=np.concatenate([np.load(source/'training'/f'{s}.npz')['x'] for s in split['fit']])
    low,median,high=np.quantile(x,[.01,.5,.99],axis=0)
    j=read(ROOT/'prediction_config.json')['jobs'][index]
    field,_=load_field(j['field']);decoder=field.region_carrier.decoder(field.xyz)
    record=read(ROOT/'predictions'/j['name']/'prediction.json')
    data=torch.load(ROOT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    features=[];chosen=[];rows=[]
    for qi,d in enumerate(record['diagnostics']):
        if 'seed' not in d:continue
        ids,f,intersection=decoder.features.pairs(d['seed']);features.append(f)
        probability=native_region_probability(field.evidence,data['native_query_vectors'][qi],data['generic_negative_vectors'])
        admitted=np.isin(ids,d.get('admitted',[]));chosen.append(f[admitted])
        quality=field.region_carrier.quality(torch.from_numpy(f)).numpy()
        rows.append(dict(text=d['text'],seed=d['seed'],seed_members=float(decoder.features.area[d['seed']]),
            seed_native_probability=float(probability[d['seed']]),candidate_pairs=len(ids),
            admitted=[dict(region=int(ids[k]),native_probability=float(probability[ids[k]]),
                predicted_candidate_purity=float(quality[k,0]),predicted_visible_recall=float(quality[k,1]),
                predicted_added_purity=float(quality[k,2])) for k in np.flatnonzero(admitted)]))
    values={}
    for name,arrays in [('all_pairs',features),('admitted_pairs',chosen)]:
        a=np.concatenate(arrays) if arrays else np.zeros((0,16))
        values[name]=dict(pairs=len(a),features={key:dict(training_p01=float(low[k]),training_median=float(median[k]),training_p99=float(high[k]),
            actual_median=float(np.median(a[:,k])) if len(a) else None,
            below_training_p01=float((a[:,k]<low[k]).mean()) if len(a) else None,
            above_training_p99=float((a[:,k]>high[k]).mean()) if len(a) else None) for k,key in enumerate(FEATURE_NAMES)})
    report=dict(scene=j['scene'],post_hoc=True,target_labels_opened=False,model_changed=False,
        note='Reference uses original high-purity training seeds; current seeds use native text. Differences mix carrier, appearance and seed-selection shifts; not isolated causality.',
        distributions=values,rows=rows,field_manifest=receipt(Path(j['field'])/'manifest.json'),code=receipt(__file__))
    write(ROOT/'domain_diagnostics'/f'{j["scene"]}.json',report)
    print(j['scene'],{k:values['all_pairs']['features'][k] for k in ['native_appearance_cosine','seed_fraction_of_scene','candidate_fraction_of_scene','candidate_relative_radius']},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();torch.set_num_threads(4);run(a.index)
