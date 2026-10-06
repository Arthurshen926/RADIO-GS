"""Native generic-background contrast; fixed .5 decision, same evaluator."""
import argparse,json,shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.native_query_encoders import load_siglip_text

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);p.add_argument('--base',default='output/clean_writing_next46');a=p.parse_args();torch.set_num_threads(4)
    base=Path(a.base);data=load(base/'robust'/a.key/'field.pt');root=Path(data['root']);identity=json.loads((root/'native_query_encoder.json').read_text())
    model,tokenizer=load_siglip_text(identity['siglip_root'],'cuda');phrases=['object','things','stuff','texture'];tokens=tokenizer(phrases,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to('cuda');negative=F.normalize(model(**tokens).pooler_output.float(),dim=-1);del model
    query=load(root.parent.parent/'evaluation/text_online.encodings.pt')[0];q=F.normalize(query['encoding'].half().float().cuda(),dim=-1)
    parent=Path('output/taskwise_scope_composition_next38')/a.key;prior=load(parent/'anchored/pipeline_calibrated/scores.pt')['scores']['hierarchical_pipeline']
    for mode in ['alpha','gate','robust']:
        payload=load(base/mode/a.key/'field.pt');state=payload['dense_surface_field'];parts=[]
        for x in state['features'].split(4096):
            f=F.normalize(x.float().cuda(),dim=-1);margin=f@q.T-(f@negative.T).amax(1,keepdim=True);parts.append((20*margin).sigmoid().cpu())
        scores=torch.where(state['supported'][:,None],torch.cat(parts),prior)
        dest=base/('contrast_'+mode)/a.key;dest.mkdir(parents=True,exist_ok=True);torch.save(dict(texts=query['texts'],scores=scores),dest/'elements.pt');shutil.copyfile(parent/'field.pt',dest/'field.pt');shutil.copyfile(parent/'calibration.pt',dest/'calibration.pt')
        (dest/'readout.json').write_text(json.dumps(dict(negative_phrases=phrases,scale=20,threshold=.5,source_features=str(base/mode/a.key/'field.pt'),target_rgb_used=False,training=False,not_lerf_clip_reproduction=True),indent=2)+'\n')

if __name__=='__main__':main()
