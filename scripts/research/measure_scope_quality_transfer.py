"""Post-prediction quality regression on the new rooms, without refitting."""
import json
from pathlib import Path
import torch
from fit_independent_scope_reliability import samples,ROOT
from radio_gs.querying.supervised_scope_reliability import ScopeReliability

def main():
    torch.set_num_threads(4)
    result=json.loads((ROOT/'summary.json').read_text())
    predictor=ScopeReliability(ROOT/'scope_reliability.pt','cpu')
    train=[torch.load(p,weights_only=False)['targets'] for p in (ROOT/'training').glob('*.pt')]
    mean=torch.stack([v.mean(0) for v in train]).mean(0)
    rows={}
    for name in result['split']['validation']:
        data=samples(name,'cuda',ROOT/'configs/adapter_vote'/f'{name}.scene.json',
            ROOT/'evaluation/adapter_vote'/name/'query/category/typed_readout.pt',ROOT/f'{name}.evaluation.json')
        x,y=data['features'],data['targets'];pred=predictor(x)
        guesses={'learned':pred,'training_scene_mean':mean[None].expand_as(y),
            'detector_and_complete':torch.stack([x[:,0],torch.ones_like(x[:,0])],1)}
        rows[name]=dict(samples=len(x),metrics={key:dict(mae=(value-y).abs().mean(0).tolist(),mse=(value-y).square().mean(0).tolist()) for key,value in guesses.items()})
        torch.save(dict(**data,predicted_quality=pred),ROOT/f'{name}.quality.pt')
        print(name,rows[name],flush=True)
    macro={mode:{metric:torch.tensor([r['metrics'][mode][metric] for r in rows.values()]).mean(0).tolist() for metric in ['mae','mse']} for mode in guesses}
    (ROOT/'quality_transfer.json').write_text(json.dumps(dict(scenes=rows,macro=macro,targets=['mask_precision','visible_category_recall'],
        label_access='post-prediction only',checkpoint_updated=False),indent=2)+'\n')

if __name__=='__main__':main()
