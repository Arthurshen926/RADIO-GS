"""Native CLIP query-versus-generic-negative readout from the local VALA recipe."""
import argparse,json,shutil
from pathlib import Path
import torch
import open_clip
from torch.nn import functional as F

def load(p):return torch.load(p,map_location='cpu',weights_only=False)
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args();torch.set_num_threads(4)
 base=Path('output/clip_field_next58/mixture')/a.key;field=load(base/'native_field.pt');online=load(base/'typed_readout.pt')
 checkpoint='/root/.cache/huggingface/hub/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots/7288da5a0d6f0b51c4a2b27c624837a9236d0112/open_clip_pytorch_model.bin'
 model,_,_=open_clip.create_model_and_transforms('ViT-B-16',pretrained=checkpoint,precision='fp16',device='cuda');model.eval()
 negatives=['object','things','stuff','texture'];q=F.normalize(model.encode_text(open_clip.get_tokenizer('ViT-B-16')(online['texts']+negatives).cuda()).float(),dim=1);del model
 scores=[];n=len(online['texts'])
 for part in field['features'].split(8192):
  cosine=F.normalize(part.cuda().float(),dim=1)@q.T
  scores.append((10*(cosine[:,:n]-cosine[:,n:].amax(1,keepdim=True))).sigmoid().cpu())
 value=torch.cat(scores)*field['supported'];out=Path('output/clip_field_next58/contrastive')/a.key;out.mkdir(parents=True,exist_ok=True)
 for name in ['field.pt','calibration.pt']:shutil.copyfile(base/name,out/name)
 torch.save(dict(online,scores=value),out/'typed_readout.pt')
 (out/'fit.json').write_text(json.dumps(dict(negatives=negatives,scale=10,source_label_calibration=False,selection_unchanged=True,feature_precision='FP16 saved field',rule='sigmoid(10*(query cosine - max generic negative cosine))'),indent=2)+'\n')
if __name__=='__main__':main()
