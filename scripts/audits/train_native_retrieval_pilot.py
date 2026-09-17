"""Native-DINO image query response training on frozen geometry; no RADIO.

Small source diagnostic, not a text/prompt or object-mask benchmark.
"""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.utils.immutable_artifacts import sha256_file
def block_radio(event,args):
 if event=='open' and args and isinstance(args[0],str):
  if any(x in args[0].lower() for x in ('canonical_teacher_features','c-radio','c_radio_v')):
   raise RuntimeError('RADIO artifact forbidden')
sys.addaudithook(block_radio)
torch.set_num_threads(4);torch.manual_seed(7);device='cuda:0'
root=Path('/mnt/pool/sqy/results/RADIO-GS/output/frozen_native_dino_pilot_v2_20260915/figurines')
out=root/'image_retrieval64_support_v2.pth'
if out.exists():raise FileExistsError(out)
teacher=torch.load(root/'dino.pt',map_location='cpu',weights_only=False)
authority=json.loads((root/'responsibility.json').read_text())
meta=teacher['metadata']
assert sha256_file(root/'responsibility.json')==meta['exact_mpr_authority_sha256']
assert sha256_file(root/'source_frames.json')==meta['source_frame_manifest_sha256']
source_records={r['frame_idx']:r for r in json.loads((root/'source_frames.json').read_text())['frames']}
frame_receipts=[]
n=meta['num_gaussians'];h,w=meta['grid']
field=ElementRetrievalField(n,64,{'image':meta['feature_dim']}).to(device)
valid=teacher['valid_train'];features=teacher['features_train'].float()
sample=features[valid][::max(1,int(valid.sum())//4096)].to(device)
# Uncentered low-rank initialization preserves dot-product structure.
_,_,basis=torch.pca_lowrank(sample,q=64,center=False,niter=3)
with torch.no_grad():
 field.latent.copy_(features.to(device)@basis)
 field.query_adapters['image'].weight.copy_(basis.T)
 field.observed.copy_(valid.to(device))
train_frames=meta['train_frames'];val_frames=meta['validation_frames']
matrices={};maps={};supports={}
for r in authority['views']:
 frame=authority['frame_indices'][r['view_index']]
 path=root/r['relative_path']
 assert sha256_file(path)==r['sha256']
 hits=torch.load(path,map_location=device,weights_only=False)
 pid=hits['pixel_ids'].long();gid=hits['gaussian_ids'].long();weight=hits['base_weights'].float()
 mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight)
 matrix=torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce()
 matrices[frame]=matrix
 fp=root/'features'/f'frame_{frame:05d}.pt'
 cached=torch.load(fp,map_location='cpu',weights_only=False)
 assert cached['source_sha256']==source_records[frame]['source_sha256']
 assert cached['checkpoint_sha256']==meta['checkpoint_sha256']
 frame_receipts.append({'frame':frame,'sha256':sha256_file(fp)})
 feature=cached['feature'] if 'feature' in cached else cached['features']
 maps[frame]=feature.float().reshape(meta['feature_dim'],-1).T.to(device)
 support=torch.sparse.mm(matrix,field.observed.float()[:,None])[:,0]
 supports[frame]=(mass>.1)&(support>.5)
# Query locations selected without inspecting target responses.
query=maps[train_frames[0]][torch.linspace(0,h*w-1,16,device=device).long()].detach()
initial=field.latent.detach().clone()
def render(frame):
 score=field.retrieve(query,'image')['scores']
 known=field.observed.float()[:,None]
 support=torch.sparse.mm(matrices[frame],known)
 return torch.sparse.mm(matrices[frame],score*known)/support.clamp_min(1e-6)
def metric(frames):
 result=[]
 with torch.no_grad():
  for frame in frames:
   target=torch.sigmoid(10*(F.normalize(maps[frame],dim=1)@F.normalize(query,dim=1).T)-5)
   pred=render(frame);mask=supports[frame]
   result.append({'frame':frame,'support_coverage':float(mask.float().mean()),
    'response_mse':float((pred[mask]-target[mask]).square().mean())})
 return result
before={'train':metric(train_frames),'validation':metric(val_frames)}
optimizer=torch.optim.Adam(field.parameters(),lr=.002)
for step in range(48):
 frame=train_frames[step%len(train_frames)]
 target=torch.sigmoid(10*(F.normalize(maps[frame],dim=1)@F.normalize(query,dim=1).T)-5)
 pred=render(frame);mask=supports[frame]
 loss=F.binary_cross_entropy(pred[mask].clamp(1e-5,1-1e-5),target[mask])
 # Bound drift on rows whose source teacher supplied actual evidence.
 loss=loss+.1*(field.latent[field.observed]-initial[field.observed]).square().mean()
 optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
after={'train':metric(train_frames),'validation':metric(val_frames)}
payload={'schema':'radio_gs.element_retrieval_field.candidate.v1','architecture':{'num_elements':n,'latent_dim':64,'query_dims':{'image':meta['feature_dim']}},
 'state_dict':{k:v.detach().cpu() for k,v in field.state_dict().items()},
 'geometry_binding':authority['metadata']['frozen_geometry_binding'],
 'geometry_sha256':authority['metadata']['frozen_geometry_sha256'],
 'query_encoder':{'model':meta['model_name'],'sha256':meta['checkpoint_sha256'],'normalization':'l2_native_patch'},
 'training_teacher_sha256':sha256_file(root/'dino.pt'),'train_frames':train_frames,'validation_frames':val_frames}
torch.save(payload,out)
# Cold inference needs only scene state and compatible query encoder output.
del teacher,features,initial
cold=torch.load(out,map_location=device,weights_only=False)
restored=ElementRetrievalField(**cold['architecture']).to(device);restored.load_state_dict(cold['state_dict'])
assert torch.equal(restored.retrieve(query,'image')['scores'],field.retrieve(query,'image')['scores'])
report={'role':'native_image_query_response_training_diagnostic','steps':48,'radio_artifact_access_blocked':True,
 'field_and_query_adapter_trained':True,'cold_query_bit_exact':True,'before':before,'after':after,
 'checkpoint':str(out),'checkpoint_sha256':sha256_file(out),
 'source_feature_receipts':frame_receipts,'observation_contract':'fixed_geometry_post_visibility_known_evidence_normalized_v1',
 'limitations':['image patch queries only; text/prompt semantic supervision not connected',
 'teacher response targets, not object-mask ground truth','small fixed source split; no final test benchmark',
 'validation shares training-selected queries; unseen-query generalization untested']}
Path('paper/artifacts/native_image_retrieval_pilot_support_v2_20260915.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
