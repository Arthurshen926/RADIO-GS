"""Frozen-geometry, optional-teacher three-query development training."""
import argparse
import math
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField, load_element_retrieval_field
from radio_gs.training.retrieval_supervision import OptionalTeacherHeads, membership_loss, boundary_loss, relation_loss, sam_contour_loss, episode_membership_targets, soft_query_relation_loss, source_region_schedule, validate_episode_split, balanced_episode_coverage, query_neighborhood_loss, episode_order, teacher_element_features, symmetric_query_region_relation_loss
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.correspondence_audit import training_query_conflicts
from radio_gs.training.feature_coverage import supervised_feature_rows, CoveragePixelSampler
from radio_gs.training.signed_label_consistency import conflicting_signed_targets
from radio_gs.training.text_task_aggregation import canonical_text_tasks
from radio_gs.training.native_query_basis import native_query_basis
from radio_gs.training.context_text_supervision import context_text_tasks
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.native_space_alignment import native_region_alignment, native_element_alignment, project_native_direction_cone
from radio_gs.training.pipeline_contracts import validate_grounded_supervision, validate_training_axes, validate_contributions, validate_native_element_teacher


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--steps',type=int,default=180);p.add_argument('--latent-dim',type=int,default=64);p.add_argument('--native-semantic-dim',type=int);p.add_argument('--protect-native-semantic',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--native-decoder-readout',action='store_true');p.add_argument('--siglip-feature-weight',type=float,default=.1);p.add_argument('--native-response-weight',type=float,default=0.);p.add_argument('--source-fusion-initialization',type=Path);p.add_argument('--freeze-native-decoder',action='store_true');p.add_argument('--source-fusion-max-drift',type=float,default=0.);p.add_argument('--source-region-query-tasks',action='store_true');p.add_argument('--cross-view-image-only',action='store_true');p.add_argument('--reliable-correspondence',action='store_true');p.add_argument('--native-match-relations',action='store_true');p.add_argument('--native-space-alignment',action='store_true');p.add_argument('--allow-legacy-region-alignment',action='store_true');p.add_argument('--native-element-teacher',type=Path);p.add_argument('--initialize-native-elements',action='store_true');p.add_argument('--native-element-weight',type=float,default=.1);p.add_argument('--native-element-min-cosine',type=float,default=0.);p.add_argument('--native-element-only-siglip',action='store_true');p.add_argument('--calibrated-readout',action='store_true');p.add_argument('--native-query-basis',action='store_true');p.add_argument('--soft-text-only',action='store_true');p.add_argument('--context-text-evidence',type=Path);p.add_argument('--grounded-text-supervision',type=Path);p.add_argument('--allow-missing-text-tasks',action='store_true');p.add_argument('--allow-legacy-text-pseudo-labels',action='store_true');p.add_argument('--require-full-episode-coverage',action='store_true',default=True);p.add_argument('--allow-incomplete-episode-coverage',action='store_false',dest='require_full_episode_coverage');p.add_argument('--share-siglip-adapter',action='store_true');p.add_argument('--freeze-siglip-adapter',action='store_true');p.add_argument('--exclude-conflicting-labels',action='store_true',default=True);p.add_argument('--allow-conflicting-pseudo-labels',action='store_false',dest='exclude_conflicting_labels');p.add_argument('--guard-signed-targets',action='store_true',default=True);p.add_argument('--allow-conflicting-signed-targets',action='store_false',dest='guard_signed_targets');p.add_argument('--unit-teacher-features',action='store_true');p.add_argument('--unique-text-training-tasks',action='store_true');p.add_argument('--source-coverage',action='store_true',default=True);p.add_argument('--matched-source-regions',action='store_false',dest='source_coverage');p.add_argument('--source-membership',action='store_true');p.add_argument('--soft-query-relations',action='store_true');p.add_argument('--teachers',default='radio,dino,sam,siglip');p.add_argument('--shuffle-episodes',action='store_true');p.add_argument('--schedule-seed',type=int,default=1000);p.add_argument('--lr-schedule',choices=['constant','cosine'],default='constant');p.add_argument('--cover-teacher-pixels',action='store_true',default=True);p.add_argument('--random-teacher-pixels',action='store_false',dest='cover_teacher_pixels');p.add_argument('--cover-dense-teacher-views',action='store_true',default=True);p.add_argument('--matched-dense-teacher-views',action='store_false',dest='cover_dense_teacher_views');p.add_argument('--query-neighborhood-weight',type=float,default=0.);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();root=a.root;out=root/a.name
 implementation_paths=[Path(__file__),Path(__file__).parents[1]/'training/pipeline_contracts.py',Path(__file__).parents[1]/'training/reliable_correspondence.py',Path(__file__).parents[1]/'training/native_space_alignment.py',Path(__file__).parents[1]/'training/retrieval_supervision.py',Path(__file__).parents[1]/'models/element_retrieval_field.py',Path(__file__).parents[1]/'models/native_decoder_readout.py',Path(__file__).parents[1]/'models/signed_query_refinement.py']
 implementation_paths.append(Path(__file__).parents[1]/'training/native_response_retention.py')
 implementation_paths.append(Path(__file__).parents[1]/'training/source_fusion_initialization.py')
 implementation_paths.append(Path(__file__).parents[1]/'training/modality_admission.py')
 implementation_sources={str(path.resolve()):sha256_file(path) for path in implementation_paths}
 if out.exists():raise FileExistsError(out)
 torch.set_num_threads(4);torch.manual_seed(a.seed);device=a.device
 if a.protect_native_semantic and (a.native_semantic_dim is None or not a.calibrated_readout):raise ValueError('semantic protection requires partitioned coordinates')
 if a.native_semantic_dim is not None and (not a.source_fusion_initialization or not a.freeze_native_decoder or not 0<a.native_semantic_dim<a.latent_dim or a.source_fusion_max_drift or a.native_response_weight):raise ValueError('partitioned control requires fixed source decoder, structural suffix and no whole-vector constraints')
 if a.latent_dim < 1:raise ValueError('latent dimension must be positive')
 if not 0 <= a.query_neighborhood_weight <= 1:raise ValueError('query neighborhood weight must be in [0,1]')
 if a.query_neighborhood_weight and not a.share_siglip_adapter:raise ValueError('query neighborhood preservation requires shared native SigLIP space')
 names=set(filter(None,a.teachers.split(',')))
 if not names<={'radio','dino','sam','siglip'}:raise ValueError(names)
 if a.grounded_text_supervision is None and a.context_text_evidence is None and not a.soft_text_only and not a.allow_legacy_text_pseudo_labels:raise ValueError('explicit verified text supervision required; legacy crop-top1 labels require --allow-legacy-text-pseudo-labels')
 if a.soft_text_only and not a.native_match_relations:raise ValueError('soft text requires actual native query-region supervision')
 if a.native_space_alignment and not a.allow_legacy_region_alignment:raise ValueError('independent unit region alignment can be contradictory; use --native-element-teacher, or explicit --allow-legacy-region-alignment for historical controls')
 if not 0<=a.native_element_min_cosine<=1:raise ValueError('native direction limit outside [0,1]')
 if a.native_element_min_cosine and not a.initialize_native_elements:raise ValueError('native direction constraint requires native initialization')
 if a.native_element_only_siglip and (a.native_element_teacher is None or a.native_element_weight<=0 or a.native_match_relations or a.soft_query_relations):raise ValueError('element-only SigLIP requires its element loss and excludes legacy region losses')
 if a.initialize_native_elements and a.native_element_teacher is None:raise ValueError('native initialization needs a source element teacher')
 if not 0<=a.native_element_weight<=1:raise ValueError('native element weight outside [0,1]')
 if a.native_element_teacher is not None:
  if not a.native_query_basis or 'siglip' not in names or a.native_space_alignment:raise ValueError('element teacher requires native query basis and SigLIP; independent region alignment is incompatible')
 if a.source_fusion_initialization and (not a.native_decoder_readout or a.native_element_teacher or a.native_query_basis):raise ValueError('source fusion initialization requires the native decoder and excludes other initialization contracts')
 if not 0<=a.source_fusion_max_drift<1 or (a.source_fusion_max_drift and not a.freeze_native_decoder):raise ValueError('bounded source updates require fixed native decoder and radius in [0,1)')
 if a.freeze_native_decoder and not a.source_fusion_initialization:raise ValueError('frozen native decoder requires source initialization')
 if not math.isfinite(a.native_response_weight) or a.native_response_weight<0 or (a.native_response_weight and not a.native_decoder_readout):raise ValueError('native response retention requires coupled decoder and nonnegative finite weight')
 if not math.isfinite(a.siglip_feature_weight) or a.siglip_feature_weight<=0:raise ValueError('SigLIP feature weight must be finite and positive')
 if a.native_decoder_readout and (not a.share_siglip_adapter or 'siglip' not in names or a.native_query_basis or a.freeze_siglip_adapter or a.native_element_teacher or a.unit_teacher_features or a.query_neighborhood_weight or a.soft_query_relations or a.native_match_relations):raise ValueError('decoder-coupled control requires its native SigLIP head and shared encoder, excludes independent latent-space constraints')
 if a.native_query_basis:
  if not a.share_siglip_adapter:raise ValueError('native query basis requires shared native text/image space')
  a.freeze_siglip_adapter=True
 if a.native_match_relations and ('siglip' not in names or not a.share_siglip_adapter):raise ValueError('native matching relations require SigLIP teacher and shared adapter')
 if a.native_space_alignment and (not a.native_query_basis or 'siglip' not in names or not a.source_coverage):raise ValueError('native alignment requires frozen source basis, SigLIP and complete source-region coverage')
 if a.source_membership and not a.source_coverage:raise ValueError('source membership requires explicit source coverage')
 if a.soft_query_relations and ('siglip' not in names or not a.source_coverage):raise ValueError('soft query relations require enabled SigLIP and source coverage')
 if 'radio' not in names:
  import sys
  def forbid_radio(event,args):
   if event=='open' and isinstance(args[0],(str,bytes)) and ('teacher_radio.pt' in str(args[0]) or 'canonical_teacher_features' in str(args[0])):
    raise RuntimeError('RADIO access forbidden in this ablation')
  sys.addaudithook(forbid_radio)
 info=json.loads((root/'teachers.json').read_text());receipt=json.loads((root/'episodes.json').read_text())
 if sha256_file(root/'episodes.pt')!=receipt['sha256']:raise ValueError('episode identity')
 data=torch.load(root/'episodes.pt',map_location='cpu',weights_only=False)
 grounded_text=None
 grounded_text_sha256=None
 if a.grounded_text_supervision is not None:
  if a.context_text_evidence is not None or a.soft_text_only or a.unique_text_training_tasks:raise ValueError('grounded text uses its own canonical tasks; choose one text supervision policy')
  grounded_receipt=json.loads(a.grounded_text_supervision.with_suffix('.json').read_text())
  grounded_text_sha256=sha256_file(a.grounded_text_supervision)
  if grounded_text_sha256!=grounded_receipt['sha256']:raise ValueError('grounded text identity')
  grounded_text=torch.load(a.grounded_text_supervision,weights_only=False)
  if grounded_text.get('schema')!='grounded_text_supervision.v2' or grounded_text['episode_sha256']!=receipt['sha256'] or grounded_text['teacher_manifest_sha256']!=data['teacher_manifest_sha256'] or grounded_text['geometry_binding']!=data['geometry_binding'] or grounded_text['queries']!=data['queries']['queries'] or grounded_text['train_frames']!=info['train_frames']:raise ValueError('grounded text supervision binding differs')
  for e in grounded_text['tasks']:
   if e['split']!='train' or e['modality']!='text' or info['frames'][e['target_view']] not in info['train_frames']:raise ValueError('grounded text contains nontraining task')
 context_evidence_sha256=None
 context_task_count=None
 context_replacement=[]
 if a.context_text_evidence is not None:
  if a.soft_text_only:raise ValueError('context hard text and soft-text-only are distinct controls')
  context_receipt=json.loads(a.context_text_evidence.with_suffix('.json').read_text())
  context_evidence_sha256=sha256_file(a.context_text_evidence)
  if context_evidence_sha256!=context_receipt['sha256']:raise ValueError('context text evidence identity')
  context=torch.load(a.context_text_evidence,weights_only=False)
  if context.get('schema')!='source_context_text_evidence.v1' or context['episode_sha256']!=receipt['sha256'] or context['teacher_manifest_sha256']!=data['teacher_manifest_sha256'] or context['geometry_binding']!=data['geometry_binding']:raise ValueError('context text evidence bindings differ')
  if context['queries']!=data['queries']['queries']:raise ValueError('context text vocabulary differs')
  if context['native_encoder_identity']!=json.loads((root/'native_query_encoder.json').read_text()):raise ValueError('context native encoder differs')
  labels=[-1]*len(data['proposal_views']);accepted=[False]*len(labels);seen_regions=set()
  for j,q in enumerate(context['regions'].tolist()):
   if q in seen_regions or not 0<=q<len(labels) or info['frames'][int(data['proposal_views'][q])] not in info['train_frames']:raise ValueError('context evidence includes duplicate or nontraining region')
   seen_regions.add(q);labels[q]=int(context['labels'][j,0]);accepted[q]=bool(context['accepted'][j])
  context_replacement=context_text_tasks(data['episodes'],data['proposal_views'],labels,accepted)
  context_task_count=len(context_replacement)
  if not context_replacement:raise ValueError('context evidence has no verified text task; do not silently restore old labels')
 training_indexed=[(i,e) for i,e in enumerate(data['episodes']) if e['split']=='train']
 if grounded_text is not None:training_indexed=[(i,e) for i,e in training_indexed if e['modality']!='text']+[(len(data['episodes'])+i,e) for i,e in enumerate(grounded_text['tasks'])]
 if a.context_text_evidence is not None:training_indexed=[(i,e) for i,e in training_indexed if e['modality']!='text']+context_replacement
 if a.unique_text_training_tasks:training_indexed=canonical_text_tasks(training_indexed)
 correspondence_admission=None
 if a.reliable_correspondence:
  from radio_gs.training.reliable_correspondence import reliable_correspondence_tasks
  training_indexed,correspondence_admission=reliable_correspondence_tasks(training_indexed,data['region_probability'].float(),data['proposal_views'],data['seen'])
 source_query_coverage=None
 if a.source_region_query_tasks:
  from radio_gs.training.reliable_correspondence import source_region_query_tasks
  training_indexed,source_query_coverage=source_region_query_tasks(training_indexed,data,[info['frames'].index(f) for f in info['train_frames']])
 image_query_admission=None
 if a.cross_view_image_only:
  if a.source_membership:raise ValueError('source membership would reintroduce same-view image tasks')
  from radio_gs.training.reliable_correspondence import cross_view_image_tasks_only
  training_indexed,image_query_admission=cross_view_image_tasks_only(training_indexed,data['proposal_views'])
 training_episodes=[e for _,e in training_indexed]
 from radio_gs.training.modality_admission import admitted_training_modalities
 training_counts={m:sum(e['modality']==m for e in training_episodes) for m in ('text','image','prompt')}
 if a.allow_missing_text_tasks and (grounded_text is None or not grounded_text.get('empty_text_supervision_explicitly_allowed')):raise ValueError('explicit grounded empty-text receipt required')
 modalities=admitted_training_modalities(training_counts,allow_missing_text=a.allow_missing_text_tasks)
 episode_coverage=balanced_episode_coverage({m:training_counts[m] for m in modalities},a.steps)
 episode_coverage['unavailable_modalities']=[m for m,n in training_counts.items() if n==0]
 episode_coverage['all_modality_task_counts']=training_counts
 if a.require_full_episode_coverage and not episode_coverage['complete']:raise ValueError('incomplete query supervision budget: '+str(episode_coverage))
 if sha256_file(root/'teachers.json')!=data['teacher_manifest_sha256']:raise ValueError('teacher manifest identity')
 if sha256_file(root/'responsibility.json')!=info['authority_sha256']:raise ValueError('authority identity')
 auth=json.loads((root/'responsibility.json').read_text());n=auth['num_gaussians'];h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width']
 pipeline_checks=validate_training_axes(data,auth,info)
 validate_episode_split(data['episodes'],data['proposal_views'],info['frames'],info['train_frames'],info['validation_frames'])
 if grounded_text is not None:
  pipeline_checks['grounded']=validate_grounded_supervision(grounded_text,frames=info['frames'],train_frames=info['train_frames'],proposal_views=data['proposal_views'],num_elements=n,num_pixels=h*w,queries=data['queries']['queries'],allow_empty=a.allow_missing_text_tasks)
  if sha256_file(Path(grounded_text['grounding_path']))!=grounded_text['grounding_sha256']:raise ValueError('grounding source identity differs')
 geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
 geometry.assert_binding(data['geometry_binding'])
 teachers={};opened=[]
 for name in sorted(names):
  if name=='siglip' and a.native_element_only_siglip:continue
  r=info['teacher_files'][name];path=root/r['path']
  if sha256_file(path)!=r['sha256']:raise ValueError(name+' identity')
  teachers[name]=torch.load(path,map_location='cpu',weights_only=False);opened.append(name)
 queries={k:data['queries'][k].float().to(device) for k in ('text','image','prompt')}
 architecture=dict(num_elements=n,latent_dim=a.latent_dim,query_dims={k:v.shape[1] for k,v in queries.items()},calibrated_readout=a.calibrated_readout,native_siglip_readout=a.native_decoder_readout)
 if a.native_semantic_dim is not None:architecture['native_semantic_dim']=a.native_semantic_dim
 if a.share_siglip_adapter:architecture['shared_query_groups']=[['text','image']]
 field=ElementRetrievalField(**architecture).to(device)
 if a.freeze_siglip_adapter:
  if not a.share_siglip_adapter:raise ValueError('frozen SigLIP control requires shared adapter')
  field.query_adapters['text'].requires_grad_(False)
 initial_siglip_projection=field.query_adapters['text'].weight.detach().clone()
 validate_episode_split(data['episodes'],data['proposal_views'],info['frames'],info['train_frames'],info['validation_frames'])
 trainviews=[info['frames'].index(f) for f in info['train_frames']]
 source_regions=source_region_schedule(data['proposal_views'],trainviews)
 prompt_valid=data['queries'].get('prompt_valid',torch.ones(len(data['proposal_views']),dtype=torch.bool))
 no_positive_regions={r['region'] for r in data.get('prompt_admission',{}).get('rejected_regions',[]) if r['reason']=='missing_positive_support'}
 if a.source_coverage and a.steps<len(source_regions):raise ValueError('insufficient budget for complete source-region supervision')
 covered_regions=set();covered_text_queries=set();dense_view_updates={v:0 for v in trainviews}
 element_teacher=None;element_teacher_sha256=None;element_teacher_rows=None;element_teacher_covered=None;element_teacher_steps=0;element_teacher_connected=False
 basis_energy=None
 if a.native_query_basis:
  bank=torch.cat([queries['text'],queries['image'][source_regions]],dim=0)
  basis,basis_energy=native_query_basis(bank.cpu(),a.latent_dim)
  with torch.no_grad():field.query_adapters['text'].weight.copy_(basis.to(device))
  initial_siglip_projection=field.query_adapters['text'].weight.detach().clone()
 field.observed.copy_(data['seen'][trainviews].any(0).to(device))
 if a.native_element_teacher is not None:
  element_receipt=json.loads(a.native_element_teacher.with_suffix('.json').read_text());element_teacher_sha256=sha256_file(a.native_element_teacher)
  if element_receipt['sha256']!=element_teacher_sha256:raise ValueError('element teacher hash differs')
  element_teacher=torch.load(a.native_element_teacher,weights_only=False)
  if element_teacher['episode_sha256']!=receipt['sha256'] or element_teacher['teacher_manifest_sha256']!=data['teacher_manifest_sha256'] or element_teacher['geometry_binding']!=data['geometry_binding'] or element_teacher['train_frames']!=info['train_frames']:raise ValueError('element teacher binding differs')
  pipeline_checks['native_element_teacher']=validate_native_element_teacher(element_teacher,num_elements=n,dimensions=a.latent_dim,query_dimension=queries['text'].shape[1],observed=field.observed)
  # SVD signs/order belong to the teacher artifact; equivalent independently
  # computed subspaces need not have identical coordinates. Use one authority.
  with torch.no_grad():field.query_adapters['text'].weight.copy_(element_teacher['basis'].to(device))
  initial_siglip_projection=field.query_adapters['text'].weight.detach().clone()
  element_teacher['directions']=F.normalize(element_teacher['directions'].float().to(device),dim=-1)
  element_teacher_rows=torch.where(element_teacher['valid'].to(device))[0];element_teacher_covered=torch.zeros(n,dtype=torch.bool,device=device)
  if a.native_element_weight>0 and a.steps*1024<len(element_teacher_rows):raise ValueError('budget insufficient to cover native element teacher')
  if a.native_element_only_siglip:opened.append('siglip_element_teacher')
  if a.initialize_native_elements:
   with torch.no_grad():field.latent[element_teacher_rows]=element_teacher['directions'][element_teacher_rows]*(.01*math.sqrt(a.latent_dim))
 geometric_observed=field.observed.clone()
 feature_update_counts=torch.zeros(n,dtype=torch.int32,device=device)
 native_neighborhood=None
 if a.query_neighborhood_weight:
  native_neighborhood=torch.cat([queries['text'],queries['image'][source_regions]],dim=0).detach()
 # Reset each auxiliary head independently so an ablation does not perturb others.
 heads=OptionalTeacherHeads(a.latent_dim-(a.native_semantic_dim or 0),{'radio':1280,'dino':768,'siglip':1536},{k:(a.siglip_feature_weight if k=='siglip' else .1) if k in names and not (k=='siglip' and a.native_element_only_siglip) else 0 for k in ('radio','dino','sam','siglip')}).to(device)
 for index,name in enumerate(('radio','dino','siglip')):
  if name in heads.heads:
   torch.manual_seed(100+index);heads.heads[name].reset_parameters()
 if a.native_decoder_readout:
  if a.native_semantic_dim is None:
   with torch.no_grad():field.native_decoder.load_state_dict(heads.heads['siglip'].state_dict())
  heads.heads['siglip']=field.native_decoder
 fusion_initialization=None;fusion_initialization_sha256=None;initialization_trace=[]
 if a.source_fusion_initialization:
  from radio_gs.training.source_fusion_initialization import apply_source_fusion,project_source_update
  ir=json.loads(a.source_fusion_initialization.with_suffix('.json').read_text());fusion_initialization_sha256=sha256_file(a.source_fusion_initialization)
  if fusion_initialization_sha256!=ir['sha256']:raise ValueError('source initialization identity differs')
  fusion_initialization=torch.load(a.source_fusion_initialization,weights_only=False,map_location='cpu')
  if fusion_initialization['schema']!='source_region_native_fusion_initialization.v1' or fusion_initialization['episode_sha256']!=receipt['sha256'] or fusion_initialization['teacher_manifest_sha256']!=data['teacher_manifest_sha256'] or fusion_initialization['geometry_binding']!=data['geometry_binding'] or fusion_initialization['train_frames']!=info['train_frames'] or fusion_initialization['benchmark_data_used'] is not False:raise ValueError('source initialization contract differs')
  pipeline_checks['source_initialized_elements']=apply_source_fusion(field,fusion_initialization,freeze_decoder=a.freeze_native_decoder)
  fusion_valid_ids=torch.where(fusion_initialization['valid'])[0].to(device);fusion_valid_latent=fusion_initialization['latent'][fusion_valid_ids.cpu()].to(device)
  valid_ids=torch.where(fusion_initialization['valid'])[0];anchor_ids=valid_ids[torch.linspace(0,len(valid_ids)-1,min(2048,len(valid_ids))).long()].to(device)
  anchor_native=F.normalize(fusion_initialization['latent'][anchor_ids.cpu()].to(device)@fusion_initialization['basis'].to(device),dim=-1)
 if a.native_semantic_dim is not None:
  # Both controls use exactly the measured anchor support, not random semantic rows.
  field.observed.copy_(fusion_initialization['valid'].to(device));geometric_observed=field.observed.clone()
  initial_semantic=field.latent[:,:a.native_semantic_dim].detach().clone()
  if a.protect_native_semantic:
   write_mask=torch.ones_like(field.latent);write_mask[:,:a.native_semantic_dim]=0
   field.latent.register_hook(lambda gradient:gradient*write_mask)
 @torch.no_grad()
 def trace_initialization(step):
  if fusion_initialization is None:return
  current=F.normalize(field.native_decoder(field.latent[anchor_ids]),dim=-1);bank=F.normalize(queries['text'],dim=-1);ref=anchor_native@bank.T;actual=current@bank.T
  initialization_trace.append({'step':step,'sampled_source_elements':len(anchor_ids),'native_direction_cosine':float((current*anchor_native).sum(-1).mean()),'source_text_response_mae':float((actual-ref).abs().mean()),'source_text_top1_retained':float((actual.argmax(-1)==ref.argmax(-1)).float().mean())})
 trace_initialization(0)
 torch.manual_seed(777)
 masks=data['masks'].flatten(1).to(device);rp=data['region_probability'].float().to(device);mats=[];support=[]
 for record in auth['views']:
  path=root/record['relative_path']
  if sha256_file(path)!=record['sha256']:raise ValueError('view identity')
  hit=torch.load(path,map_location=device,weights_only=False);pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weights=hit['base_weights'].float()
  validate_contributions(pid,gid,weights,num_pixels=h*w,num_elements=n)
  mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weights)
  mat=torch.sparse_coo_tensor(torch.stack([pid,gid]),weights/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce()
  mats.append(mat);support.append(torch.sparse.mm(mat,field.observed.float()[:,None])[:,0])
 pixel_sampler=CoveragePixelSampler([torch.where(x>.5)[0] for x in support]) if a.cover_teacher_pixels else None
 dense={k:F.interpolate(teachers[k].float().to(device),size=(h,w),mode='bilinear',align_corners=False).flatten(2).transpose(1,2) for k in ('radio','dino') if k in names}
 if 'siglip' in names and not a.native_element_only_siglip:teachers['siglip']=teachers['siglip']['descriptors'].float().to(device)
 relation_bank=torch.cat([queries['text'],queries['image'][source_regions]],dim=0).detach() if a.native_match_relations else None
 native_response_bank=torch.cat([queries['text'],queries['image'][source_regions]],dim=0).detach() if a.native_response_weight else None
 native_response_regions={}
 if a.native_response_weight:
  from radio_gs.training.native_response_retention import region_element_weights,native_response_loss
  for ri in source_regions:
   vi=int(data['proposal_views'][ri]);native_response_regions[ri]=region_element_weights(mats[vi],support[vi],masks[ri],field.observed)
 relation_view_updates={v:0 for v in trainviews}
 query_conflicts=training_query_conflicts([e for e in training_episodes if 'grounded_text_task' not in e],rp) if a.exclude_conflicting_labels else {}
 def evidence(e):
  return episode_membership_targets(e,masks,rp,field.observed,support[e['target_view']])
 signed_target_conflicts=set()
 if a.guard_signed_targets:
  seed_cache={}
  for episode_index,e in training_indexed:
   if e['split']!='train' or e['modality']!='prompt':continue
   q=e['query_region'];v=int(data['proposal_views'][q])
   if q not in seed_cache:
    matrix=mats[v];indices=matrix.indices();weights=matrix.values();rows=[];values=[]
    for pixel in data['queries']['seed_pixels'][q].tolist():
     selected=indices[0]==pixel;rows.append(indices[1,selected]);values.append(weights[selected])
    seed_cache[q]=(rows,values)
   _,_,rowtarget,rowknown=evidence(e)
   if any(conflicting_signed_targets(*seed_cache[q],[1,0],rowtarget,rowknown,field.observed)):signed_target_conflicts.add(episode_index)
 guarded_episode_ids={id(e) for i,e in training_indexed if i in signed_target_conflicts}
 if grounded_text is not None:
  for key in ('positive_pixels','negative_pixels','positive_elements','negative_elements'):grounded_text[key]=grounded_text[key].to(device)
 def training_evidence(e):
  if 'grounded_text_task' in e:
   index=e['grounded_text_task'];qi=e['text_index'];vi=e['target_view']
   pos=grounded_text['positive_pixels'][index];neg=grounded_text['negative_pixels'][index]
   rp=grounded_text['positive_elements'][qi];rn=grounded_text['negative_elements'][qi]
   return pos.float(),(pos^neg)&(support[vi]>.5),rp.float(),(rp^rn)&field.observed&data['seen'][vi].to(device)
  target,known,rowtarget,rowknown=evidence(e)
  if id(e) in guarded_episode_ids:return target,torch.zeros_like(known),rowtarget,torch.zeros_like(rowknown)
  key=(e['modality'],e.get('text_index',e['query_region']))
  if key in query_conflicts:
   ambiguous=query_conflicts[key]&field.observed;v=e['target_view']
   ambiguous_mass=torch.sparse.mm(mats[v],ambiguous.float()[:,None])[:,0]/support[v].clamp_min(1e-8)
   rowknown=rowknown&~ambiguous
   # Supervision validity only: geometry, predictions and evaluation stay intact.
   known=known&(ambiguous_mass<=.01)
  return target,known,rowtarget,rowknown
 def response(e):
  qi=e.get('text_index',e['query_region']);q=queries[e['modality']][qi:qi+1]
  probs=field.retrieve(q,e['modality'])['scores'][:,0]
  v=e['target_view'];render=torch.sparse.mm(mats[v],(probs*field.observed)[:,None])[:,0]/support[v].clamp_min(1e-8)
  return probs,render
 @torch.no_grad()
 def evaluate():
  result={}
  for split in ('train','validation'):
   result[split]={}
   for modality in queries:
    values=[];losses=[];coverage=[]
    for e in data['episodes']:
     if e['split']!=split or e['modality']!=modality:continue
     _,render=response(e);target,known,_,_=evidence(e);pred=render>.5;truth=target.bool()
     union=((pred|truth)&known).sum();intersection=((pred&truth)&known).sum()
     values.append(float(intersection/union.clamp_min(1)));losses.append(float(membership_loss(render,target,known)));coverage.append(float(known.float().mean()))
    result[split][modality]={'known_support_pseudo_iou':sum(values)/len(values) if values else None,'membership_loss':sum(losses)/len(losses) if losses else None,'known_pixel_fraction':sum(coverage)/len(coverage) if coverage else None,'episodes':len(values)}
  return result
 @torch.no_grad()
 def evaluate_optimization_tasks():
  result={}
  for modality in queries:
   rows=[]
   for e in training_episodes:
    if e['modality']!=modality:continue
    _,render=response(e);target,known,_,_=training_evidence(e)
    prediction=render>.5;truth=target.bool();intersection=((prediction&truth)&known).sum();union=((prediction|truth)&known).sum()
    rows.append({'iou':float(intersection/union.clamp_min(1)),'membership_loss':float(membership_loss(render,target,known)),'known_pixels':int(known.sum())})
   result[modality]={'tasks':len(rows),'tasks_with_known_pixels':sum(r['known_pixels']>0 for r in rows),'abstained_tasks':sum(r['known_pixels']==0 for r in rows),'known_support_pseudo_iou':sum(r['iou'] for r in rows)/len(rows) if rows else None,'membership_loss':sum(r['membership_loss'] for r in rows)/len(rows) if rows else None}
  return result
 out.mkdir()
 before=evaluate();optimization_before=evaluate_optimization_tasks()
 calibration=list(field.readout_log_scale.parameters())+list(field.readout_bias.parameters());calibration_ids={id(x) for x in calibration}
 optimizer=torch.optim.Adam([{'params':[x for x in dict.fromkeys(list(field.parameters())+list(heads.parameters())) if id(x) not in calibration_ids],'lr':.003},{'params':calibration,'lr':.03}])
 bymod={m:[e for e in training_episodes if e['modality']==m] for m in queries};history=[];gradients={}
 for step in range(a.steps):
  if a.lr_schedule=='cosine':
   rate=.003*(.1+.9*.5*(1+math.cos(math.pi*step/max(a.steps-1,1))))
   for index,group in enumerate(optimizer.param_groups):group['lr']=rate*(10 if index==1 else 1)
  m=modalities[step%len(modalities)];items=bymod[m];iteration=step//len(modalities)
  order=episode_order(len(items),iteration//len(items),a.schedule_seed+step%len(modalities),a.shuffle_episodes)
  e=items[order[iteration%len(items)]];optimizer.zero_grad(set_to_none=True)
  probs,render=response(e);target,known,rowtarget,rowknown=training_evidence(e)
  loss2=membership_loss(render,target,known);loss3=membership_loss(probs,rowtarget,rowknown)
  if a.soft_text_only and m=='text':loss2=render.sum()*0;loss3=probs.sum()*0
  loss=loss2+loss3
  if a.native_decoder_readout and m in ('text','image') and step<3 and not a.protect_native_semantic:
   query_gradient=torch.autograd.grad(loss,field.native_decoder.weight if field.native_decoder.weight.requires_grad else field.latent,retain_graph=True,allow_unused=True)[0]
   if query_gradient is None or not query_gradient.norm()>0:raise ValueError('native query membership disconnected from decoder')
   gradients['native_query_'+m]=float(query_gradient.norm())
  if m=='prompt':
   qregion=e['query_region'];qview=int(data['proposal_views'][qregion]);pixels=data['queries']['seed_pixels'][qregion].to(device)
   # A signed pixel constrains the rendered mixture, not one arbitrary Gaussian.
   seed_response=torch.sparse.mm(mats[qview],(probs*field.observed)[:,None])[:,0]/support[qview].clamp_min(1e-8)
   valid_seed=support[qview][pixels]>.5
   if valid_seed.any():
    loss=loss+F.binary_cross_entropy(seed_response[pixels][valid_seed].clamp(1e-6,1-1e-6),torch.tensor([1.,0.],device=device)[valid_seed])
  auxiliary_region=e['target_region']
  if a.source_coverage:
   auxiliary_region=source_regions[step%len(source_regions)];covered_regions.add(auxiliary_region)
   source_view=int(data['proposal_views'][auxiliary_region])
   if a.source_membership and auxiliary_region not in no_positive_regions:
    source_modality=('image','prompt')[(step//len(source_regions))%2]
    if source_modality=='prompt' and not bool(prompt_valid[auxiliary_region]):source_modality='image'
    source_episode={'query_region':auxiliary_region,'target_region':auxiliary_region,
                    'target_view':source_view,'modality':source_modality,'negative_regions':[]}
    source_probs,source_render=response(source_episode)
    st,sk,srt,srk=evidence(source_episode)
    source_loss=.25*(membership_loss(source_render,st,sk)+membership_loss(source_probs,srt,srk))
    if source_modality=='prompt':
     pixels=data['queries']['seed_pixels'][auxiliary_region].to(device)
     valid_seed=support[source_view][pixels]>.5
     if valid_seed.any():
      source_loss=source_loss+.25*F.binary_cross_entropy(source_render[pixels][valid_seed].clamp(1e-6,1-1e-6),torch.tensor([1.,0.],device=device)[valid_seed])
    loss=loss+source_loss
  if a.protect_native_semantic and m in ('text','image') and step<3:
   cg=torch.autograd.grad(loss,field.readout_bias[m],retain_graph=True)[0]
   if not torch.isfinite(cg) or cg.abs()==0:raise ValueError('native membership calibration disconnected')
   gradients['native_calibration_'+m]=float(cg.abs())
  v=int(data['proposal_views'][auxiliary_region])
  teacher_elements=teacher_element_features(field.latent,field.observed,a.unit_teacher_features)
  latent=torch.sparse.mm(mats[v],teacher_elements)/support[v][:,None].clamp_min(1e-8)
  dense_view=trainviews[step%len(trainviews)] if a.cover_dense_teacher_views else v
  dense_latent=latent if dense_view==v else torch.sparse.mm(mats[dense_view],teacher_elements)/support[dense_view][:,None].clamp_min(1e-8)
  if dense:dense_view_updates[dense_view]+=1
  valid=torch.where(support[dense_view]>.5)[0]
  selected=(pixel_sampler.sample(dense_view,1024) if pixel_sampler is not None else valid[torch.randperm(len(valid),device=device)[:1024]]) if dense else valid[:0]
  aux={}
  for name in dense:
   aux[name]=heads.feature_loss(name,dense_latent[selected,a.native_semantic_dim or 0:],dense[name][dense_view,selected])
   aux[name+'_relation']=.02*relation_loss(dense_latent[selected[:128],a.native_semantic_dim or 0:],dense[name][dense_view,selected[:128]])
  if 'siglip' in names:
   region=masks[auxiliary_region]&(support[v]>.5)
   if not a.native_element_only_siglip:aux['siglip']=heads.feature_loss('siglip',latent[region].mean(0,keepdim=True),teachers['siglip'][auxiliary_region:auxiliary_region+1]) if region.any() else latent.sum()*0
   if a.native_space_alignment and region.any():
    directions=F.normalize(field.latent,dim=-1)*field.observed[:,None]
    mean_directions=torch.sparse.mm(mats[v],directions)/support[v][:,None].clamp_min(1e-8)
    aux['native_space_alignment']=.1*native_region_alignment(mean_directions[region].mean(0,keepdim=True),teachers['siglip'][auxiliary_region:auxiliary_region+1],field.query_adapters['text'].weight)
   if a.soft_query_relations:
    region_ids=torch.where(data['proposal_views']==v)[0].to(device)
    region_masks=masks[region_ids].float()*(support[v]>.5)[None]
    keep=region_masks.sum(1)>0;region_ids=region_ids[keep];region_masks=region_masks[keep]
    # Average normalized ELEMENT features first: these are actual mean query
    # cosines, not cosine of a newly normalized region descriptor.
    element_directions=F.normalize(field.latent,dim=-1)*field.observed[:,None]
    pixel_directions=torch.sparse.mm(mats[v],element_directions)/support[v][:,None].clamp_min(1e-8)
    region_directions=region_masks@pixel_directions/region_masks.sum(1,keepdim=True).clamp_min(1)
    text_directions=F.normalize(field.query_adapters['text'](queries['text']),dim=-1)
    student_scores=text_directions@region_directions.T
    teacher_scores=F.normalize(queries['text'],dim=-1)@F.normalize(teachers['siglip'][region_ids],dim=-1).T
    aux['siglip_query_relations']=.1*soft_query_relation_loss(student_scores,teacher_scores)
    covered_text_queries.update(range(len(queries['text'])))
  if 'sam' in names:
   aux['sam']=.1*sam_contour_loss(latent[:,a.native_semantic_dim or 0:],masks[auxiliary_region],support[v]>.5,h,w)
   aux['sam_membership_boundary']=render.sum()*0 if a.soft_text_only and m=='text' else .1*boundary_loss(render,target,known,h,w)
  if relation_bank is not None:
   rv=trainviews[step%len(trainviews)];relation_view_updates[rv]+=1
   region_ids=torch.where(data['proposal_views']==rv)[0].to(device)
   region_masks=masks[region_ids].float()*(support[rv]>.5)[None]
   keep=region_masks.sum(1)>0;region_ids=region_ids[keep];region_masks=region_masks[keep]
   directions=F.normalize(field.latent,dim=-1)*field.observed[:,None]
   pixels=torch.sparse.mm(mats[rv],directions)/support[rv][:,None].clamp_min(1e-8)
   region_directions=region_masks@pixels/region_masks.sum(1,keepdim=True).clamp_min(1)
   adapted=F.normalize(field.query_adapters['text'](relation_bank),dim=-1)
   native_scores=F.normalize(relation_bank,dim=-1)@F.normalize(teachers['siglip'][region_ids],dim=-1).T
   aux['native_match_relations']=.1*symmetric_query_region_relation_loss(adapted@region_directions.T,native_scores)
  if native_neighborhood is not None:
   aux['query_neighborhood']=a.query_neighborhood_weight*query_neighborhood_loss(field.query_adapters['image'](native_neighborhood),native_neighborhood)
  if element_teacher is not None and a.native_element_weight>0:
   er=element_teacher_rows[(torch.arange(1024,device=device)+step*1024)%len(element_teacher_rows)];element_teacher_covered[er]=True
   aux['native_element_alignment']=a.native_element_weight*native_element_alignment(field.latent[er],element_teacher['directions'][er]);element_teacher_steps+=1
   if not element_teacher_connected:
    check_gradient=torch.autograd.grad(aux['native_element_alignment'],field.latent,retain_graph=True,allow_unused=True)[0]
    if check_gradient is None:raise ValueError('native element teacher is disconnected from field')
    element_teacher_connected=True
  if a.native_response_weight and native_response_regions[auxiliary_region] is not None:
   ids,weights=native_response_regions[auxiliary_region]
   aux['native_response']=a.native_response_weight*native_response_loss(field.latent[ids],field.native_decoder,native_response_bank,teachers['siglip'][auxiliary_region],weights)
  if step<3 and a.freeze_native_decoder and not a.protect_native_semantic and 'siglip' in aux:
   gradient=torch.autograd.grad(aux['siglip'],field.latent,retain_graph=True,allow_unused=True)[0]
   if gradient is None:raise ValueError('frozen teacher loss disconnected from elements')
   gradients['teacher_siglip']=max(gradients.get('teacher_siglip',0),float(gradient.norm()))
  loss=loss+sum(aux.values())
  if step<3 and 'sam' in aux:
   gradient=torch.autograd.grad(aux['sam'],field.latent,retain_graph=True)[0]
   gradients['teacher_sam']=max(gradients.get('teacher_sam',0),float(gradient.norm()))
  loss.backward()
  if not torch.isfinite(loss):raise ValueError('nonfinite loss')
  feature_update_counts.add_(supervised_feature_rows(field.latent.grad,geometric_observed).int())
  adapter_gradient=field.query_adapters[m].weight.grad
  if adapter_gradient is not None:gradients[m]=max(gradients.get(m,0),float(adapter_gradient.norm()))
  for name,head in heads.heads.items():
   if head.weight.grad is not None:gradients['teacher_'+name]=max(gradients.get('teacher_'+name,0),float(head.weight.grad.norm()))
  optimizer.step()
  if a.source_fusion_max_drift:project_source_update(field.latent,fusion_valid_ids,fusion_valid_latent,a.source_fusion_max_drift)
  if step+1 in (1,16,64,256,1024,a.steps):trace_initialization(step+1)
  if a.native_element_min_cosine:
   with torch.no_grad():field.latent[element_teacher_rows]=project_native_direction_cone(field.latent[element_teacher_rows],element_teacher['directions'][element_teacher_rows],a.native_element_min_cosine)
  if step%30==0 or step==a.steps-1:
   row={'step':step,'learning_rate':optimizer.param_groups[0]['lr'],'modality':m,'loss':float(loss),'membership_2d':float(loss2),'membership_3d':float(loss3),'aux':{k:float(v) for k,v in aux.items()}};history.append(row);print(row,flush=True)
 if a.source_coverage and len(covered_regions)!=len(source_regions):raise ValueError('source-region supervision coverage incomplete')
 if element_teacher is not None and a.native_element_weight>0 and int(element_teacher_covered.sum())!=len(element_teacher_rows):raise ValueError('native element teacher coverage incomplete')
 native_retention=None
 if element_teacher is not None:
  with torch.no_grad():
   cosine=(F.normalize(field.latent[element_teacher_rows],dim=-1)*element_teacher['directions'][element_teacher_rows]).sum(-1);native_retention={'mean':float(cosine.mean()),'minimum':float(cosine.min()),'p05':float(cosine.quantile(.05))}
   if a.native_element_min_cosine and native_retention['minimum']<a.native_element_min_cosine-1e-5:raise ValueError('native feature retention constraint failed')
 required_queries=[m for m in queries if field.query_adapters[m].weight.requires_grad]
 if a.freeze_siglip_adapter and not torch.equal(initial_siglip_projection,field.query_adapters['text'].weight):raise ValueError('frozen query projection changed')
 if any(gradients.get(k,0)<=0 for k in required_queries+['teacher_'+name for name in names if not (name=='siglip' and (a.native_element_only_siglip or a.protect_native_semantic))]):raise ValueError('inactive required supervision gradient')
 if a.protect_native_semantic and not torch.equal(initial_semantic,field.latent[:,:a.native_semantic_dim]):raise ValueError('protected native semantic coordinates changed')
 after_training_visibility=evaluate()
 # Visibility permits training writes; only evidence-trained rows are exported
 # as semantic observations. No random untouched latent is a valid map feature.
 field.observed.copy_(geometric_observed & (feature_update_counts>0))
 support=[torch.sparse.mm(mat,field.observed.float()[:,None])[:,0] for mat in mats]
 after=evaluate();optimization_after=evaluate_optimization_tasks();geometry.assert_binding(data['geometry_binding'])
 encoder=json.loads((root/'native_query_encoder.json').read_text())
 payload={'schema':'radio_gs.element_retrieval_field.candidate.v1','semantic_validity_contract':'visible_and_supervised_v1','architecture':architecture,'state_dict':{k:v.cpu() for k,v in field.state_dict().items()},'geometry_binding':data['geometry_binding'],'query_encoder':encoder,'query_context':{'prompt_world_normalization':data['queries']['prompt_world_normalization'],'prompt_raster_size':[h,w]},'training_query_asset_binding':{'query_file_sha256':info['query_file']['sha256'],'episode_sha256':receipt['sha256'],'grounded_text_supervision_sha256':grounded_text_sha256,'context_text_evidence_sha256':context_evidence_sha256,'native_element_teacher_sha256':element_teacher_sha256}}
 if any(sha256_file(Path(path))!=digest for path,digest in implementation_sources.items()):raise RuntimeError('training implementation changed during execution; refusing artifact publication')
 counts_path=out/'feature_update_counts.pt';torch.save(feature_update_counts.cpu(),counts_path)
 path=out/'field.pt';torch.save(payload,path)
 cold,_=load_element_retrieval_field(path,expected_sha256=sha256_file(path),geometry=geometry,expected_query_encoder=encoder,device=device)
 cold_error={m:float((cold.retrieve(queries[m][:1],m)['scores']-field.retrieve(queries[m][:1],m)['scores']).abs().max()) for m in queries}
 report={'schema':'three_query_four_teacher_development.v1','implementation_sources':implementation_sources,'pipeline_checks':pipeline_checks,'correspondence_admission':correspondence_admission,'source_query_coverage':source_query_coverage,'image_query_admission':image_query_admission,'source_fusion_initialization_sha256':fusion_initialization_sha256,'native_semantic_dim':a.native_semantic_dim,'protect_native_semantic':a.protect_native_semantic,'teacher_roles':{'siglip':'fixed source native anchor' if a.protect_native_semantic else 'native feature reconstruction','radio':'dense reconstruction','dino':'dense reconstruction','sam':'mask-derived contour and response boundary; not native feature reconstruction'},'teacher_roles_enabled_only':sorted(names),'freeze_native_decoder':a.freeze_native_decoder,'source_fusion_max_drift':a.source_fusion_max_drift,'initialization_trace':initialization_trace,'correctness_policy':{'exclude_query_conflicts':a.exclude_conflicting_labels,'guard_signed_targets':a.guard_signed_targets,'independent_source_region_coverage':a.source_coverage,'legacy_text_explicitly_allowed':a.allow_legacy_text_pseudo_labels},'implementation_sha256':sha256_file(Path(__file__)),'supervision_implementation_sha256':sha256_file(Path(__file__).parents[1]/'training/retrieval_supervision.py'),'conflicting_training_labels_excluded':a.exclude_conflicting_labels,'guard_signed_targets':a.guard_signed_targets,'unit_teacher_features':a.unit_teacher_features,'unique_text_training_tasks':a.unique_text_training_tasks,'training_task_local_indices':[i for i,e in training_indexed],'training_task_source_episode_indices':[e.get('source_episode_indices',[i] if i<len(data['episodes']) else []) for i,e in training_indexed],'signed_target_conflict_episode_indices':sorted(signed_target_conflicts),'conflicting_query_element_pairs':sum(int(v.sum()) for v in query_conflicts.values()),'source_coverage_enabled':a.source_coverage,'source_membership_enabled':a.source_membership,'soft_query_relations_enabled':a.soft_query_relations,'source_regions_available':len(source_regions),'source_regions_supervised':len(covered_regions),'soft_text_queries_supervised':len(covered_text_queries),'episode_coverage':episode_coverage,'steps':a.steps,'latent_dim':a.latent_dim,'seed':a.seed,'native_decoder_readout':a.native_decoder_readout,'siglip_feature_weight':a.siglip_feature_weight,'native_response_weight':a.native_response_weight,'native_match_relations':a.native_match_relations,'native_space_alignment':a.native_space_alignment,'native_element_teacher_sha256':element_teacher_sha256,'native_element_only_siglip':a.native_element_only_siglip,'native_element_supervision_steps':element_teacher_steps,'native_element_supervision_connected':element_teacher_connected,'native_element_min_cosine':a.native_element_min_cosine,'native_retention':native_retention,'initialize_native_elements':a.initialize_native_elements,'native_element_weight':a.native_element_weight if element_teacher is not None else 0.,'native_element_teacher_covered':int(element_teacher_covered.sum()) if element_teacher_covered is not None else 0,'calibrated_readout':a.calibrated_readout,'readout_calibration':{m:{'scale':float(v.clamp(-4.,6.).exp()),'bias':float(field.readout_bias[m])} for m,v in field.readout_log_scale.items()},'native_query_basis':a.native_query_basis,'soft_text_only':a.soft_text_only,'context_text_evidence_sha256':context_evidence_sha256,'grounded_text_supervision_sha256':grounded_text_sha256,'grounded_text_training_counts':grounded_text['counts'] if grounded_text is not None else None,'context_text_training_task_count':context_task_count,'context_text_training_tasks':[e for _,e in context_replacement],'text_membership_monitor_is_not_training_loss':a.soft_text_only,'native_basis_energy_retained':basis_energy,'native_relation_view_updates':{info['frames'][v]:count for v,count in relation_view_updates.items()},'shuffle_episodes':a.shuffle_episodes,'schedule_seed':a.schedule_seed,'lr_schedule':a.lr_schedule,'cover_teacher_pixels':a.cover_teacher_pixels,'cover_dense_teacher_views':a.cover_dense_teacher_views,'dense_teacher_view_updates':{info['frames'][v]:count for v,count in dense_view_updates.items()},'teacher_pixel_coverage':[{'valid':len(v),'visited':int(k.sum())} for v,k in zip(pixel_sampler.valid_pixels,pixel_sampler.covered)] if pixel_sampler is not None else None,'feature_update_counts_sha256':sha256_file(counts_path),'query_neighborhood_weight':a.query_neighborhood_weight,'query_neighborhood_inputs':'fixed vocabulary and training-view image queries only','siglip_adapter_frozen':a.freeze_siglip_adapter,'siglip_projection_change_norm':float((field.query_adapters['text'].weight-initial_siglip_projection).norm()),'enabled_auxiliary_teachers':sorted(names),'opened_teacher_files':opened,'before':before,'after':after,'optimization_tasks_before':optimization_before,'optimization_tasks_after':optimization_after,'legacy_text_monitor_uses_old_labels':grounded_text is not None or a.context_text_evidence is not None,'after_training_visibility':after_training_visibility,'semantic_validity_contract':'visible and nonzero semantic feature gradient during training; not a convergence guarantee','training_coverage':{'geometrically_visible_elements':int(data['seen'][trainviews].any(0).sum()),'training_domain_elements':int(geometric_observed.sum()),'native_anchor_domain_used':a.native_semantic_dim is not None,'semantically_trained_elements':int(field.observed.sum()),'excluded_untrained_visible_elements':int((geometric_observed & ~field.observed).sum()),'excluded_outside_training_domain':int(data['seen'][trainviews].any(0).sum())-int(geometric_observed.sum()),'mean_feature_update_steps_on_valid':float(feature_update_counts[field.observed].float().mean())},'gradient_norm_max':gradients,'history':history,'cold_encoded_query_max_error':cold_error,'field_sha256':sha256_file(path),'geometry_binding_unchanged':True,'benchmark_ground_truth':False,'evaluation_scope':'heldout query AND target views; teacher pseudo labels; unknown pixels excluded','sam_label_provenance_remains_when_sam_aux_disabled':True,'native_online_query_encoder_cold_test':False}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(after),flush=True)
if __name__=='__main__':main()
