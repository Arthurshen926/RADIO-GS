"""Four independent query modalities on one frozen observation map.

The semantic control reads native region features. Point scope uses regions
containing a geometric anchor; instance scope keeps native identity separate
from cross-view physical correspondence. Maps may declare local membership
weights compiled from other-view native evidence. Rankings and explicit sets
are separate outputs, not interchangeable probability maps.
No source image, source depth, mask decoder or target annotation is loaded here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import sparse
from scipy.spatial import cKDTree
import torch
from torch.nn import functional as F

from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
from .observation_readout import native_identity, project_observations, corroborate_identity, mean_native_identity_ranks
from .instance_observation_scope import InstanceObservationScope
from .observation_hypotheses import hypothesis_identity, anchor_track
from .anchor_observation_evidence import anchor_membership


def local_surface(xyz, tree):
    """A continuous local carrier from the same public, unlabelled vertices."""
    distance,neighbors=tree.query(xyz,k=12)
    offset=xyz[neighbors]-xyz[:,None,:]
    covariance=np.einsum('nki,nkj->nij',offset,offset)
    _,vectors=np.linalg.eigh(covariance)
    return vectors[:,:,0],distance[:,3]


def geometric_anchor(xyz, query, tree, surface):
    if query['modality']=='point_3d':
        distance,index=tree.query(query['point_world_xyz'])
        return int(index),dict(distance_m=float(distance),anchor_rule='nearest_unlabelled_map_vertex')
    pose=np.asarray(query['camera_to_world']);k=np.asarray(query['camera_intrinsics'])
    normal,radius=surface
    direction=pose[:3,:3]@np.linalg.solve(k,np.r_[query['positive_pixel_uv'],1.])
    direction/=np.linalg.norm(direction)
    denominator=normal@direction
    travel=np.divide(np.sum(normal*(xyz-pose[:3,3]),axis=1),denominator,
                     out=np.full(len(xyz),np.inf),where=np.abs(denominator)>1e-5)
    candidates=np.flatnonzero((travel>0)&np.isfinite(travel))
    hit=pose[:3,3]+travel[candidates,None]*direction
    inside=np.linalg.norm(hit-xyz[candidates],axis=1)<=radius[candidates]
    candidates,hit=candidates[inside],hit[inside]
    if len(candidates):
        first=int(np.argmin(travel[candidates]));index=int(candidates[first])
        return index,dict(anchor_rule='first_ray_intersection_with_local_mesh_surfel',
            intersection_world_xyz=hit[first].tolist(),vertex_to_intersection_m=float(np.linalg.norm(hit[first]-xyz[index])),
            rgb_read=False,depth_read=False)
    camera=(xyz-pose[:3,3])@pose[:3,:3]
    ids=np.flatnonzero(camera[:,2]>.001);points=camera[ids]
    uv=points@k.T;uv=uv[:,:2]/uv[:,2,None]
    angular=np.linalg.norm(uv-np.asarray(query['positive_pixel_uv']),axis=1)
    hit=np.flatnonzero(angular<=2.)
    # The radius only covers the raster footprint of discrete mesh vertices.
    local=hit[np.argmin(points[hit,2])] if len(hit) else int(np.argmin(angular))
    index=int(ids[local])
    return index,dict(pixel_distance=float(angular[local]),depth_m=float(points[local,2]),surface_intersection_missing=True,
        anchor_rule='frontmost_mesh_vertex_within_2_pixels_else_nearest_ray',rgb_read=False,depth_read=False)


def transfer_anchor_scope(membership, visibility, region_frames, anchor):
    """One geometric hop to observations where the anchor itself is invisible.

    Require support for >= half the smaller scope and >= quarter the larger.
    This prevents an arbitrarily large containing region from inheriting the
    anchor. No semantic similarity between disjoint objects creates an edge.
    """
    seeds=membership[anchor].indices
    counts=np.bincount(region_frames[seeds],minlength=visibility.shape[1])
    weights=np.zeros(membership.shape[1],np.float32)
    weights[seeds]=1/np.maximum(counts[region_frames[seeds]],1)
    strength=(counts>0).astype(np.float32)
    area=np.asarray(membership.sum(0)).reshape(-1)
    overlap=(membership.T@membership[:,seeds]).tocoo()
    r,s,shared=overlap.row,seeds[overlap.col],overlap.data
    keep=(counts[region_frames[r]]==0)&(shared>=.5*np.minimum(area[r],area[s]))&(shared>=.25*np.maximum(area[r],area[s]))
    r,s,shared=r[keep],s[keep],shared[keep]
    evidence=np.zeros_like(weights)
    np.maximum.at(evidence,r,shared/np.maximum(area[r],area[s]))
    total=np.bincount(region_frames,weights=evidence,minlength=visibility.shape[1])
    new_strength=np.zeros_like(strength)
    np.maximum.at(new_strength,region_frames,evidence)
    weights+=evidence/np.maximum(total[region_frames],1e-8)*new_strength[region_frames]
    strength+=new_strength
    known=np.asarray(visibility@strength).reshape(-1)
    positive=np.asarray(membership@weights).reshape(-1)
    value=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
    return np.clip(value,0,1),dict(seed_views=int((counts>0).sum()),transferred_views=int((new_strength>0).sum()),
        transferred_regions=int((evidence>0).sum()),query_supported_vertices=int((known>0).sum()),
        matching='one-hop physical region overlap; half-smaller/quarter-larger support',semantic_space_changed=False)


def public_queries(path, scene):
    if path.is_file():
        value=json.loads(path.read_text())
        rows=value['queries'] if 'queries' in value else [value]
        return [q for q in rows if q['scene_id']==scene],path.parent
    rows=[]
    for modality in ['text','image','point_2d','point_3d']:
        rows.extend(q for q in json.loads((path/'queries'/f'{modality}.json').read_text())['queries'] if q['scene_id']==scene)
    return rows,path


@torch.inference_mode()
def run(map_root, public, output, device='cuda',scope_transfer=False,instance_scope=False):
    started=time.monotonic();torch.set_num_threads(4)
    spec=json.loads((map_root/'map.json').read_text());scene=spec['scene_id']
    xyz=np.load(map_root/'mesh_xyz.npy');tree=cKDTree(xyz);surface=local_surface(xyz,tree)
    descriptors=np.load(map_root/'descriptors.npy').astype(np.float32)
    descriptors/=np.maximum(np.linalg.norm(descriptors,axis=1,keepdims=True),1e-8)
    transfer=sparse.load_npz(map_root/'transfer.npz')
    membership=sparse.load_npz(map_root/'membership.npz');visibility=sparse.load_npz(map_root/'visibility.npz')
    region_frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(map_root/'region_offsets.npy')))
    positive_membership=sparse.load_npz(map_root/spec['positive_membership']) if spec.get('positive_membership') else None
    surface_consistency=None
    if spec.get('surface_consistency'):
        from .surface_consistency import SurfaceConsistency
        settings=spec['surface_consistency']
        surface_consistency=SurfaceConsistency(sparse.load_npz(map_root/settings['graph']),settings['strength'],settings['solver'],
            score_rule=settings.get('score_rule','unary'))
    scope_reader=InstanceObservationScope(membership,visibility,region_frames,positive_membership,surface_consistency,
        spec.get('scope_exclusion_policy','visible_complement')) if instance_scope else None
    witnessed_reader=None
    if instance_scope and spec.get('scope_witness_rule'):
        from .witnessed_observation_scope import WitnessedObservationScope
        witnessed_reader=WitnessedObservationScope(membership,visibility,region_frames,
            sparse.load_npz(map_root/spec['scope_witnesses']),spec['scope_witness_rule'])
    entity_reader=None
    if instance_scope and spec.get('entity_scope'):
        from .entity_observation_scope import EntityObservationScope
        entity=spec['entity_scope']
        entity_frames=np.repeat(np.arange(visibility.shape[1]),np.diff(np.load(map_root/entity['offsets'])))
        entity_reader=EntityObservationScope(membership,region_frames,
            sparse.load_npz(map_root/entity['membership']),visibility,entity_frames)
    conditional_reader=None
    if instance_scope and spec.get('conditional_membership'):
        from .conditional_region_membership import ConditionalRegionMembership
        from .conditional_element_membership import ConditionalElementMembership
        reader_class=(ConditionalElementMembership if spec['conditional_membership'].get('decoder')=='element'
                      else ConditionalRegionMembership)
        local={}
        if spec['conditional_membership'].get('local_appearance'):
            local=dict(local_appearance=np.load(map_root/'element_appearance.npy',mmap_mode='r'),
                       local_observation=np.load(map_root/'element_appearance_observation.npy'))
        if spec['conditional_membership'].get('decoder')=='element':local['surface_consistency']=surface_consistency
        conditional_reader=reader_class(membership,visibility,region_frames,xyz,
            np.load(map_root/'appearance_descriptors.npy'),spec['conditional_membership']['checkpoint'],**local)
    direct_anchor_reader=None
    if instance_scope and spec.get('direct_anchor_membership'):
        from .direct_anchor_membership import DirectAnchorMembership
        direct_anchor_reader=DirectAnchorMembership(membership,visibility,region_frames,xyz,
            descriptors,transfer,spec['direct_anchor_membership']['checkpoint'])
    query_member_reader=None
    if instance_scope and spec.get('query_conditioned_membership'):
        from .query_conditioned_membership import QueryConditionedMembership
        query_member_reader=QueryConditionedMembership(membership,visibility,region_frames,xyz,
            descriptors,transfer,spec['query_conditioned_membership']['checkpoint'])
    supported=transfer.getnnz(axis=1)>0
    queries,public=public_queries(public,scene)
    encodings={}; anchors={}; details={};appearance_encodings={};query_patches={};query_patch_weights={}
    dual_image=spec.get('image_identity_fusion')=='equal_rank'
    text_queries=[q for q in queries if q['modality']=='text']
    referent_plans={};phrase_encodings={};language_report=None
    if text_queries and spec.get('text_identity'):
        missing=[q for q in text_queries if 'referent_plan' not in q]
        if missing:
            from radio_gs.models.referent_language import parse_referents
            plans,language_report=parse_referents([q['expression'] for q in missing],device=device)
            for q,plan in zip(missing,plans):q['referent_plan']=plan
        else:
            language_report=dict(precomputed_query_only_plans=True,images_read=False,
                additional_encoder='Qwen/Qwen3-VL-8B-Instruct',annotations_read=False)
        referent_plans={q['query_id']:q['referent_plan'] for q in text_queries}
        output.mkdir(parents=True,exist_ok=True)
        (output/'referent_plans.json').write_text(json.dumps(dict(plans=referent_plans,
            language_preprocessing=language_report),indent=2)+'\n')
    lexical_features=None
    if text_queries:
        model,tokenizer=load_siglip_text(spec['native_encoder'],device)
        tokens=tokenizer([q['expression'] for q in text_queries],padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
        encoded=model(**tokens)
        result=F.normalize(encoded.pooler_output.float(),dim=-1).cpu().numpy()
        if spec.get('referent_grounding'):
            text_hidden=encoded.last_hidden_state.cpu()
            text_valid=(tokens['input_ids']!=tokenizer.pad_token_id).cpu()
        if spec.get('lexical_identity'):
            from .lexical_identity_reader import lexical_query_features
            lexical_features=lexical_query_features([q['expression'] for q in text_queries],encoded,
                tokenizer,model,spec['lexical_identity']['role_checkpoint'],device)
        del encoded
        for q,value in zip(text_queries,result):encodings[q['query_id']]=value
        if referent_plans:
            phrases=list(dict.fromkeys(phrase for plan in referent_plans.values()
                for phrase in [plan['target']]+[r['reference'] for r in plan['relations']]))
            tokens=tokenizer(phrases,padding='max_length',max_length=64,truncation=True,return_tensors='pt').to(device)
            result=F.normalize(model(**tokens).pooler_output.float(),dim=-1).cpu().numpy()
            phrase_encodings=dict(zip(phrases,result))
        del model;torch.cuda.empty_cache()
    image_queries=[q for q in queries if q['modality']=='image']
    if image_queries:
        if spec.get('image_native'):
            from radio_gs.models.native_region_appearance import NativeRegionAppearance
            encoder=NativeRegionAppearance(spec['image_native']['encoder'],device=device,repository=spec['image_native']['repository'])
        else:encoder=NativeSiglip2Runtime.load(Path(spec['native_encoder']),device=torch.device(device))
        for q in image_queries:
            image=Image.open(public/q['crop_rgb_path']).convert('RGB')
            if spec.get('image_native'):
                if spec.get('image_local_match'):
                    local,valid=encoder.patches(image)
                    query_patches[q['query_id']]=local.cpu().numpy()
                    query_patch_weights[q['query_id']]=valid.cpu().numpy()
                    encodings[q['query_id']]=F.normalize(valid[None]@local,dim=-1)[0].cpu().numpy()
                else:encodings[q['query_id']]=encoder.encode(image)[0].cpu().numpy()
                appearance_encodings[q['query_id']]=encodings[q['query_id']]
            else:
                image=image.resize((384,384),Image.Resampling.BICUBIC)
                pixels=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()[None]/255
                encodings[q['query_id']]=encoder.encode(pixels)[0].cpu().numpy()
        del encoder;torch.cuda.empty_cache()
        if dual_image:
            encoder=NativeSiglip2Runtime.load(Path(spec['native_encoder']),device=torch.device(device))
            for q in image_queries:
                image=Image.open(public/q['crop_rgb_path']).convert('RGB').resize((384,384),Image.Resampling.BICUBIC)
                pixels=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()[None]/255
                encodings[q['query_id']]=encoder.encode(pixels)[0].cpu().numpy()
            del encoder;torch.cuda.empty_cache()
    for q in queries:
        qid=q['query_id']
        if q['modality'].startswith('point_'):
            anchor,detail=geometric_anchor(xyz,q,tree,surface);anchors[qid]=anchor
            value=np.asarray(transfer[anchor]@descriptors).reshape(-1)
            encodings[qid]=value/max(float(np.linalg.norm(value)),1e-8)
            details[qid]=dict(anchor_vertex=anchor,anchor_has_feature=bool(supported[anchor]),**detail)
    native_columns=[i for i,q in enumerate(queries) if q['modality']!='image' or not spec.get('image_native') or dual_image]
    response=np.empty((len(descriptors),len(queries)),np.float32)
    if native_columns:
        values=np.stack([encodings[queries[i]['query_id']] for i in native_columns])
        response[:,native_columns]=native_identity(descriptors,values)
    if spec.get('image_native') and image_queries:
        image_columns=[i for i,q in enumerate(queries) if q['modality']=='image']
        appearance=np.load(map_root/spec['image_native']['descriptors']).astype(np.float32)
        appearance/=np.maximum(np.linalg.norm(appearance,axis=1,keepdims=True),1e-8)
        image_values=np.stack([appearance_encodings[queries[i]['query_id']] for i in image_columns])
        appearance_response=native_identity(appearance,image_values)
        if not dual_image:response[:,image_columns]=appearance_response
    local_match_response=None
    if spec.get('image_local_match') and image_queries:
        from .local_appearance_match import match_regions
        local=spec['local_appearance']
        matched=match_regions(np.load(map_root/local['features'],mmap_mode='r'),
            sparse.load_npz(map_root/local['region_weights']),np.load(map_root/local['offsets']),
            np.load(map_root/'region_offsets.npy'),
            [query_patches[q['query_id']] for q in image_queries],
            [query_patch_weights[q['query_id']] for q in image_queries],device=device)
        local_match_response=matched[spec['image_local_match']]
    identity_response=response
    hypotheses=sparse.load_npz(map_root/spec['scope_hypotheses']) if spec.get('scope_hypotheses') else None
    tracks=sparse.load_npz(map_root/spec['instance_tracks']) if spec.get('instance_tracks') else None
    track_response=hypothesis_identity(tracks,response) if tracks is not None else None
    if hypotheses is not None:
        identity_response=hypothesis_identity(hypotheses,response)
    if spec.get('member_identity') and native_columns:
        member=np.load(map_root/spec['member_identity']).astype(np.float32)
        member/=np.maximum(np.linalg.norm(member,axis=1,keepdims=True),1e-8)
        coverage=np.load(map_root/spec['member_identity_coverage'])
        identity_response=identity_response.copy()
        identity_response[:,native_columns]=corroborate_identity(response[:,native_columns],native_identity(member,values),coverage)
    if dual_image and image_queries:
        identity_response=identity_response.copy()
        available=np.flatnonzero(membership.getnnz(axis=0)>0)
        image_evidence=local_match_response if local_match_response is not None else appearance_response
        identity_response[:,image_columns]=mean_native_identity_ranks(response[:,image_columns],image_evidence,available)
    if text_queries and spec.get('text_context_identity'):
        text_columns=[i for i,q in enumerate(queries) if q['modality']=='text']
        channel='unmasked' if spec['text_context_identity']=='unmasked' else 'context'
        context=np.load(map_root/spec['regional_context'][channel]).astype(np.float32)
        context/=np.maximum(np.linalg.norm(context,axis=1,keepdims=True),1e-8)
        context_response=context@np.stack([encodings[queries[i]['query_id']] for i in text_columns],axis=1)
        identity_response=identity_response.copy()
        if spec['text_context_identity']=='unmasked':
            identity_response[:,text_columns]=context_response
        else:
            available=np.flatnonzero(membership.getnnz(axis=0)>0)
            identity_response[:,text_columns]=mean_native_identity_ranks(response[:,text_columns],context_response,available)
    referent_details={}
    if referent_plans:
        from .referent_identity import ReferentIdentity,native_logit_scale
        resolver=ReferentIdentity(membership,xyz,native_logit_scale(spec['native_encoder']))
        phrase_response={phrase:descriptors@value for phrase,value in phrase_encodings.items()}
        identity_response=identity_response.copy()
        for i,q in enumerate(queries):
            if q['modality']!='text':continue
            plan=referent_plans[q['query_id']]
            relations=[(r['predicate'],r['reference'],phrase_response[r['reference']])
                for r in plan['relations']] if spec['text_identity']!='target' else []
            preserve_expression=spec['text_identity'] in ['native_relations','relation_support']
            base=response[:,i] if preserve_expression else phrase_response[plan['target']]
            energy,detail=resolver.rank(base,relations,positive_only=spec['text_identity']=='relation_support')
            identity_response[:,i]=energy
            referent_details[q['query_id']]=dict(**detail,target_phrase=plan['target'],
                unresolved=plan['unresolved'],role_plan=plan,
                full_expression_identity_retained=preserve_expression,
                identity_rule=('original full expression' if preserve_expression else 'native target phrase')+
                    ' with '+('marginal geometric landmark evidence' if relations else 'no landmark execution'))
    lexical_report=None
    if text_queries and spec.get('lexical_identity'):
        from .lexical_identity_reader import LexicalIdentityReader
        lexical_reader=LexicalIdentityReader(spec['lexical_identity']['checkpoint'],descriptors,membership,device)
        lexical_scores,lexical_report=lexical_reader.rank(lexical_features)
        text_columns=[i for i,q in enumerate(queries) if q['modality']=='text']
        identity_response=identity_response.copy();identity_response[:,text_columns]=lexical_scores
        del lexical_reader;torch.cuda.empty_cache()
    grounding_report=None
    if text_queries and spec.get('referent_grounding'):
        from .scene_referent_grounding import SceneReferentIdentity
        grounder=SceneReferentIdentity(spec['referent_grounding']['checkpoint'],descriptors,
            membership,visibility,region_frames,xyz,device)
        text_columns=[i for i,q in enumerate(queries) if q['modality']=='text']
        pooled=np.stack([encodings[q['query_id']] for q in text_queries])
        grounded,grounding_report=grounder.rank(text_hidden,text_valid,pooled)
        identity_response=identity_response.copy();identity_response[:,text_columns]=grounded
        del grounder;torch.cuda.empty_cache()
    if spec.get('save_identity_evidence'):
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'identity_evidence.npz',
            query_ids=np.asarray([q['query_id'] for q in queries]),
            native_response=response,identity_response=identity_response)
    semantic=np.clip((np.asarray(project_observations(transfer,response))+1)*.5,0,1)
    semantic[~supported]=0
    records=[]
    for i,q in enumerate(queries):
        qid=q['query_id'];score=semantic[:,i].astype(np.float32)
        if qid in anchors and not supported[anchors[qid]]:score[:]=0
        alternate=score
        if qid in anchors:
            alternate,scope=anchor_membership(membership,visibility,region_frames,anchors[qid],positive_membership)
            details[qid].update(scope)
        predictions=[('native_semantic',score),('anchor_comembership',alternate)]
        if instance_scope:
            seed_rank,scope_rank=alternate,alternate
            selections={name:alternate>.5 for name in ['identity_region','identity_scope']}
            if tracks is not None:
                track_detail={}
                if qid in anchors:
                    track,track_detail=anchor_track(tracks,membership,visibility,region_frames,anchors[qid])
                else:track=int(np.argmax(track_response[:,i]))
                if track is not None:
                    weights=tracks[track].toarray().reshape(-1)
                    members=np.flatnonzero(weights)
                    seed=int(members[np.argmax(response[members,i])])
                    confidence=np.bincount(region_frames,weights=weights,minlength=visibility.shape[1]).astype(np.float32)
                    control,scope_rank,selections,identity_detail=scope_reader.decode(response[:,i],score,seed,weights,confidence)
                    seed_rank=control
                    if qid in anchors:
                        seed_rank=alternate;selections['identity_region']=alternate>.5
                        positive=np.asarray(membership@weights).reshape(-1)
                        known=np.asarray(visibility@confidence).reshape(-1)
                        scope_rank=np.divide(positive,known,out=np.zeros_like(positive),where=known>0)
                        identity_detail.update(ranking_rule='positive fraction in selected track observations',
                            absent_scope_policy='zero rank without track observations; not verified object absence')
                    identity_detail.update(track_id=track,track_count=tracks.shape[0],
                        identity_rule='expected native cosine across a view-exclusive physical track' if qid not in anchors else 'positive geometric anchor',
                        scope_rule='mutual geometric edges; view-exclusive track; equal per-view strict majority',**track_detail)
                    details.setdefault(qid,{})['instance_identity']=identity_detail
            elif qid not in anchors:
                seed=None
                if hypotheses is not None:
                    available=np.flatnonzero(scope_reader.area>0)
                    seed=int(available[np.argmax(identity_response[available,i])])
                seed_rank,scope_rank,selections,identity_detail=scope_reader.infer(identity_response[:,i],score,seed=seed)
                if dual_image and q['modality']=='image':
                    selected_seed=identity_detail['seed_region']
                    identity_detail.update(seed_native_similarity=float(response[selected_seed,i]),
                        seed_appearance_similarity=float(appearance_response[selected_seed,image_columns.index(i)]),
                        seed_identity_percentile=float(identity_response[selected_seed,i]),
                        identity_rule='equal mean of independent native SigLIP and DINO candidate ranks; semantic field unchanged')
                    if local_match_response is not None:
                        identity_detail.update(local_appearance_rule=spec['image_local_match'],
                            seed_local_match_score=float(local_match_response[selected_seed,image_columns.index(i)]),
                            identity_rule='equal native SigLIP and local DINO correspondence ranks; original semantic field and physical scope')
                if qid in referent_details:
                    selected_seed=identity_detail['seed_region']
                    identity_detail.update(**referent_details[qid],
                        seed_native_similarity=float(response[selected_seed,i]),
                        seed_identity_energy=float(identity_response[selected_seed,i]))
                if q['modality']=='text' and spec.get('text_context_identity'):
                    selected_seed=identity_detail['seed_region']
                    identity_detail.update(seed_native_similarity=float(response[selected_seed,i]),
                        text_context_identity=spec['text_context_identity'],
                        identity_rule=('native unmasked tight crop' if spec['text_context_identity']=='unmasked' else
                            'equal ranks of masked object and expanded visual context; original semantic field unchanged'))
                if hypotheses is not None:
                    identity_detail.update(seed_native_similarity=float(response[seed,i]),
                        hypothesis_identity=float(identity_response[seed,i]),
                        hypothesis_count=int((scope_reader.area>0).sum()),
                        identity_rule='expected native cosine across physical scope hypotheses',
                        region_native_argmax=int(available[np.argmax(response[available,i])]))
                if spec.get('member_identity'):
                    seed=identity_detail['seed_region']
                    identity_detail.update(seed_native_similarity=float(response[seed,i]),
                        seed_corroborated_similarity=float(identity_response[seed,i]),
                        other_view_member_coverage=float(coverage[seed]))
                if q['modality']=='text' and grounding_report is not None:
                    selected_seed=identity_detail['seed_region']
                    identity_detail.update(seed_native_similarity=float(response[selected_seed,i]),
                        seed_grounded_logit=float(identity_response[selected_seed,i]),
                        identity_rule='native full-expression logit plus independently trained referring identity correction',
                        referent_grounding=grounding_report)
                if q['modality']=='text' and lexical_report is not None:
                    identity_detail.update(identity_rule=lexical_report['identity_rule'],
                        lexical_identity=lexical_report)
                details[qid]={'instance_identity':identity_detail}
            if witnessed_reader is not None:
                scope_rank,selected,witness_detail=witnessed_reader.infer(identity_response[:,i],score,anchors.get(qid))
                selections['identity_scope']=selected
                details.setdefault(qid,{})['witnessed_scope']=witness_detail
            if entity_reader is not None:
                entity_rank,entity_selected,entity_detail=entity_reader.infer(identity_response[:,i],score,anchors.get(qid))
                if entity_detail['applied']:
                    scope_rank=entity_rank;selections['identity_scope']=entity_selected
                details.setdefault(qid,{})['entity_scope']=entity_detail
            if conditional_reader is not None:
                rank,selected,detail=conditional_reader.infer(identity_response[:,i],score,anchors.get(qid))
                if detail['applied']:scope_rank=rank;selections['identity_scope']=selected
                details.setdefault(qid,{})['conditional_membership']=detail
            if direct_anchor_reader is not None and qid in anchors:
                scope_rank,selected,detail=direct_anchor_reader.infer(anchors[qid])
                selections['identity_scope']=selected
                details.setdefault(qid,{})['direct_anchor_membership']=detail
            if direct_anchor_reader is not None and q['modality'] in spec.get('inferred_anchor_membership',[]):
                available=np.flatnonzero(scope_reader.area>0)
                seed=int(available[np.argmax(identity_response[available,i])])
                scope_rank,selected,detail=direct_anchor_reader.infer_region(seed)
                selections['identity_scope']=selected
                details.setdefault(qid,{})['inferred_anchor_membership']=detail
            if query_member_reader is not None and q['modality'] in spec['query_conditioned_membership']['applies_to']:
                scope_rank,selected,detail=query_member_reader.infer(identity_response[:,i],response[:,i])
                selections['identity_scope']=selected
                details.setdefault(qid,{})['query_conditioned_membership']=detail
            predictions.extend([('identity_region',seed_rank),('identity_scope',scope_rank)])
            for name,selected in selections.items():
                folder=output/name/'selections';folder.mkdir(parents=True,exist_ok=True)
                np.save(folder/f'{qid}.npy',selected.astype(bool))
        if scope_transfer:
            transferred=score
            if qid in anchors:
                transferred,transfer_detail=transfer_anchor_scope(membership,visibility,region_frames,anchors[qid])
                details[qid]['scope_transfer']=transfer_detail
            predictions.append(('anchor_correspondence',transferred))
        for name,prediction in predictions:
            folder=output/name/'predictions';folder.mkdir(parents=True,exist_ok=True)
            np.save(folder/f'{qid}.npy',prediction.astype(np.float32))
        records.append(dict(query_id=qid,modality=q['modality'],**details.get(qid,{})))
    report=dict(scene_id=scene,queries=len(queries),query_details=records,map=str(map_root),
        map_manifest_sha256=hashlib.sha256((map_root/'map.json').read_bytes()).hexdigest(),
        runtime_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        map_bytes=spec['map_bytes'],map_count=1,source_rgb_read=False,source_depth_read=False,
        source_masks_read=False,labels_read=False,query_rgb_read=len(image_queries),
        reference_rgb_for_point_queries=False,query_order_independent=True,
        native_semantic_output='linear native cosine in [0,1]; no fitted mask threshold',
        structural_output='conditional anchor-region co-membership; uncalibrated membership',
        scope_transfer=scope_transfer,instance_scope=instance_scope,
        local_membership=spec.get('local_membership'),
        conditional_membership=spec.get('conditional_membership'),
        direct_anchor_membership=spec.get('direct_anchor_membership'),
        query_conditioned_membership=spec.get('query_conditioned_membership'),
        surface_consistency=spec.get('surface_consistency'),
        member_identity=spec.get('member_identity_build'),
        scope_hypotheses=spec.get('scope_hypotheses_build'),
        instance_tracks=spec.get('instance_tracks_build'),
        image_native=spec.get('image_native'),
        image_identity_fusion=spec.get('image_identity_fusion'),
        image_local_match=spec.get('image_local_match'),
        scope_witness_rule=spec.get('scope_witness_rule'),
        entity_scope=spec.get('entity_scope'),
        text_identity=spec.get('text_identity'),language_preprocessing=language_report,
        text_context_identity=spec.get('text_context_identity'),
        referent_grounding=grounding_report,
        lexical_identity=lexical_report,
        inferred_anchor_membership=spec.get('inferred_anchor_membership'),
        independent_selection_output=instance_scope,
        selection_rule=('query-conditioned complete-carrier membership with uncertain estimated identity; original actual points' if query_member_reader is not None else
            'complete-carrier membership from provided points or estimated regional anchors' if spec.get('inferred_anchor_membership') else
            'actual-anchor element classifier for points; original native identity/scope for text and image' if direct_anchor_reader is not None else
            'independently trained conditional membership; otherwise freshly computed base scope' if conditional_reader is not None else
            'offline entity scope with multi-view support; otherwise freshly computed base scope' if entity_reader is not None else
            'third-view witnessed scope '+spec['scope_witness_rule'] if witnessed_reader is not None else
            'seed mask or confidence-weighted visible-observation strict majority; point anchor membership > .5'),
        seconds=time.monotonic()-started)
    receipts=output/'runs';receipts.mkdir(parents=True,exist_ok=True)
    (receipts/f'{scene}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(scene,'completed',len(queries),'independent queries on one frozen map',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--map',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope-transfer',action='store_true')
    p.add_argument('--instance-scope',action='store_true')
    a=p.parse_args();run(a.map,a.public,a.output,scope_transfer=a.scope_transfer,instance_scope=a.instance_scope)


if __name__=='__main__':main()
