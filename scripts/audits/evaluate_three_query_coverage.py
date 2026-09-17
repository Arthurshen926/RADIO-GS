"""Matched evaluation on immutable partial source labels; no benchmark claims."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.models.element_retrieval_field import load_element_retrieval_field
from radio_gs.models.signed_query_refinement import condition_signed_query, retrieve_latent_query
from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
from radio_gs.training.retrieval_supervision import episode_membership_targets
from radio_gs.training.retrieval_metrics import partial_retrieval_metrics, summarize_partial_episodes, coverage_aware_retrieval_metrics
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.training.source_view_split import episode_in_validation_frames
from radio_gs.training.text_task_aggregation import canonical_text_tasks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--field-root',type=Path,help='Load runs from another root with the same geometry and encoder')
    p.add_argument('--reference-run',type=Path,help='Intersect evaluation support with this hash-verified run')
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:2')
    p.add_argument('--refine-prompts',action='store_true')
    p.add_argument('--minimal-prompt-refinement',action='store_true')
    p.add_argument('--unique-text-tasks',action='store_true',default=True)
    p.add_argument('--correspondence-weighted-text',action='store_false',dest='unique_text_tasks')
    p.add_argument('--validation-frame-ids',type=int,nargs='+',help='Require BOTH query and target view in this validation subset')
    a=p.parse_args();root=a.root;device=a.device;torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError(a.output)
    receipt=json.loads((root/'episodes.json').read_text())
    if sha256_file(root/'episodes.pt')!=receipt['sha256']:raise ValueError('episode identity')
    data=torch.load(root/'episodes.pt',weights_only=False)
    info=json.loads((root/'teachers.json').read_text())
    selected_frames=set(a.validation_frame_ids) if a.validation_frame_ids is not None else set(info['validation_frames'])
    if not selected_frames or not selected_frames<=set(info['validation_frames']):raise ValueError('evaluation subset must contain validation frames only')
    evaluation_episodes=[(i,e) for i,e in enumerate(data['episodes']) if episode_in_validation_frames(e,data['proposal_views'],info['frames'],selected_frames)]
    raw_episode_count=len(evaluation_episodes)
    if a.unique_text_tasks:evaluation_episodes=canonical_text_tasks(evaluation_episodes)
    if sha256_file(root/'teachers.json')!=data['teacher_manifest_sha256']:raise ValueError('teacher manifest identity')
    if sha256_file(root/'responsibility.json')!=info['authority_sha256']:raise ValueError('authority identity')
    auth=json.loads((root/'responsibility.json').read_text());h=auth['metadata']['feature_height'];w=auth['metadata']['feature_width'];n=auth['num_gaussians']
    geometry=FrozenGaussianGeometry.load(a.geometry,expected_sha256=auth['metadata']['frozen_geometry_sha256'],device=device)
    identity=json.loads((root/'native_query_encoder.json').read_text())
    matrices=[]
    for record in auth['views']:
        path=root/record['relative_path']
        if sha256_file(path)!=record['sha256']:raise ValueError('view identity')
        hit=torch.load(path,map_location=device,weights_only=False)
        pid=hit['pixel_ids'].long();gid=hit['gaussian_ids'].long();weight=hit['base_weights'].float()
        mass=torch.zeros(h*w,device=device).scatter_add_(0,pid,weight)
        matrices.append(torch.sparse_coo_tensor(torch.stack([pid,gid]),weight/mass[pid].clamp_min(1e-8),(h*w,n)).coalesce())
    masks=data['masks'].flatten(1).to(device);rp=data['region_probability'].to(device)
    hard_train_words={e['text_index'] for e in data['episodes'] if e['split']=='train' and e['modality']=='text'}
    results={}
    reference=None
    if a.reference_run:
        reference_report=json.loads((a.reference_run/'report.json').read_text())
        reference,_=load_element_retrieval_field(a.reference_run/'field.pt',expected_sha256=reference_report['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=device)
    with torch.no_grad():
        for run in a.runs:
            training=json.loads(((a.field_root or root)/run/'report.json').read_text())
            field,_=load_element_retrieval_field((a.field_root or root)/run/'field.pt',expected_sha256=training['field_sha256'],geometry=geometry,expected_query_encoder=identity,device=device)
            support=[torch.sparse.mm(mat,field.observed.float()[:,None])[:,0] for mat in matrices]
            evaluation_observed=field.observed if reference is None else field.observed & reference.observed
            evaluation_support=[torch.sparse.mm(mat,evaluation_observed.float()[:,None])[:,0] for mat in matrices]
            rows=[]
            for index,e in evaluation_episodes:
                m=e['modality'];qi=e.get('text_index',e['query_region']);v=e['target_view']
                query=data['queries'][m][qi:qi+1].float().to(device)
                response=field.retrieve(query,m)['scores'][:,0]
                rendered=torch.sparse.mm(matrices[v],(response*field.observed)[:,None])[:,0]/support[v].clamp_min(1e-8)
                target,known,row_target,row_known=episode_membership_targets(e,masks,rp,evaluation_observed,evaluation_support[v])
                positive_masks=masks[e.get('positive_regions',[e['target_region']])].any(0)
                negative_masks=masks[e['negative_regions']].any(0) if e['negative_regions'] else torch.zeros_like(positive_masks)
                all_known=positive_masks ^ negative_masks
                entry={'image_all_labels':coverage_aware_retrieval_metrics(rendered,positive_masks,all_known,support[v]>.5), 'episode_index':index,'source_episode_indices':e.get('source_episode_indices',[index]),'modality':m,'query_region':e['query_region'],'target_view':v,
                       'image':partial_retrieval_metrics(rendered,target,known,support[v]>.5),
                       'elements':partial_retrieval_metrics(response,row_target,row_known,field.observed)}
                if m=='prompt' and a.refine_prompts:
                    qregion=e['query_region'];qview=int(data['proposal_views'][qregion])
                    pixels=data['queries']['seed_pixels'][qregion].tolist();matrix=matrices[qview]
                    indices=matrix.indices();weights=matrix.values();seed_rows=[];seed_weights=[]
                    for pixel in pixels:
                        selected=indices[0]==pixel;seed_rows.append(indices[1,selected]);seed_weights.append(weights[selected])
                    refinement=condition_signed_query(field,query,seed_rows,seed_weights,[1.,0.],stop_when_satisfied=a.minimal_prompt_refinement)
                    refined=retrieve_latent_query(field,refinement['latent_query'])['scores'][:,0]
                    refined_render=torch.sparse.mm(matrices[v],(refined*field.observed)[:,None])[:,0]/support[v].clamp_min(1e-8)
                    entry['image_refined_all_labels']=coverage_aware_retrieval_metrics(refined_render,positive_masks,all_known,support[v]>.5)
                    entry['image_refined']=partial_retrieval_metrics(refined_render,target,known,support[v]>.5)
                    entry['elements_refined']=partial_retrieval_metrics(refined,row_target,row_known,field.observed)
                    entry['signed_evidence']={k:val.tolist() if torch.is_tensor(val) else val for k,val in refinement.items() if k!='latent_query'}
                if m=='text':entry.update(text=data['queries']['queries'][qi],hard_training_label_seen=qi in hard_train_words)
                rows.append(entry)
            metrics={}
            for modality in ('text','image','prompt'):
                subset=[row for row in rows if row['modality']==modality]
                metrics[modality]={}
                domains=['image','image_all_labels','elements']+(['image_refined','image_refined_all_labels','elements_refined'] if modality=='prompt' and a.refine_prompts else [])
                for domain in domains:
                    metrics[modality][domain]=summarize_partial_episodes([row[domain] for row in subset])
            results[run]={'refinement_availability':{'requested':sum(r['modality']=='prompt' for r in rows) if a.refine_prompts else 0,'available':sum(r.get('signed_evidence',{}).get('available',False) for r in rows),'unavailable':sum(r.get('signed_evidence',{}).get('available') is False for r in rows)},'field_sha256':training['field_sha256'],'macro':metrics,'episodes':rows}
            print(run,{m:metrics[m]['image']['known_iou'] for m in metrics},flush=True)
    result={'schema':'partial_source_retrieval_evaluation.v1','episode_sha256':receipt['sha256'],'benchmark_ground_truth':False,'all_label_policy':'explicit nonconflicting 2D labels; unsupported predictions are zero; independent of reference support',
            'evaluation_status':'development evaluation; holdout novelty recorded by experiment protocol','selected_validation_frames':sorted(selected_frames),'unique_text_tasks':a.unique_text_tasks,'raw_episode_count':raw_episode_count,'evaluated_task_count':len(evaluation_episodes),'reference_field_sha256':reference_report['field_sha256'] if reference is not None else None,'field_root':str(a.field_root or root),'prompt_refinement_uses_only_authorized_source_seeds':a.refine_prompts,'minimal_prompt_refinement':a.minimal_prompt_refinement,'unknown_response_mass_is_not_false_positive_rate':True,'runs':results}
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
