"""Check every deployed element against explicit native decoding on source queries."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_decoder_readout import affine_decoder_cosine
from radio_gs.utils.immutable_artifacts import sha256_file

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:2');a=p.parse_args();torch.set_num_threads(4)
 if a.output.exists():raise FileExistsError(a.output)
 info=json.loads((a.root/'teachers.json').read_text());rec=info['query_file'];qp=a.root/rec['path'];assert sha256_file(qp)==rec['sha256'];queries=torch.load(qp,weights_only=False)['text'].float().to(a.device);results={}
 for name in a.runs:
  path=a.root/name/'field.pt';r=json.loads((path.parent/'report.json').read_text());assert sha256_file(path)==r['field_sha256'];d=torch.load(path,map_location=a.device,weights_only=False);f=ElementRetrievalField(**d['architecture']).to(a.device);f.load_state_dict(d['state_dict']);assert f.native_decoder is not None
  maximum=0.;posterior_max=0.
  with torch.no_grad():
   for z in f.latent.split(4096):
    exact=F.normalize(f.native_decoder(z),dim=-1)@F.normalize(queries,dim=-1).T;compact=affine_decoder_cosine(z,queries,f.native_decoder.weight,f.native_decoder.bias);maximum=max(maximum,float((exact-compact).abs().max()));posterior_max=max(posterior_max,float((f.cosine_logits(exact,'text').sigmoid()-f.cosine_logits(compact,'text').sigmoid()).abs().max()))
  if maximum>3e-5 or posterior_max>3e-5:raise ValueError('deployed native cosine differs from explicit decoding')
  results[name]={'field_sha256':r['field_sha256'],'all_elements_checked':len(f.latent),'source_queries_checked':len(queries),'maximum_cosine_error':maximum,'maximum_posterior_error':posterior_max,'semantic_accuracy_claim':False}
 a.output.write_text(json.dumps({'schema':'deployed_decoder_cosine_parity.v1','source_query_sha256':rec['sha256'],'runs':results},indent=2)+'\n');print(results)
if __name__=='__main__':main()
