"""CPU differential audit on retained inputs, NOT a fresh benchmark evaluation.

Every declared scene is visited. Source observations and UQIS anchors are
deterministically sampled; no GT, original query image, or old prediction is read.
Only checks, hashes and counts persist. The full five-output regression is a
separate GPU run and must never be inferred from this component report.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch
from torch.nn import functional as F

from radio_gs import evidence_core as core
from radio_gs.evidence_core.store import EvidenceStore
from radio_gs.querying.direct_anchor_membership import DirectAnchorEvidence
from radio_gs.querying.instance_observation_scope import InstanceObservationScope
from radio_gs.querying.local_element_fusion import compile_local_evidence


def frozen_module(name):
    root=Path(__file__).resolve().parents[2]
    path=root/'docs/history/2026-10-18-evidence-core-baseline/radio_gs/querying'/(name+'.py')
    spec=importlib.util.spec_from_file_location('radio_gs.querying._audit_frozen_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@contextmanager
def profile(name):
    old = os.environ.get('RADIO_GS_EVIDENCE_CORE')
    os.environ['RADIO_GS_EVIDENCE_CORE'] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('RADIO_GS_EVIDENCE_CORE', None)
        else:
            os.environ['RADIO_GS_EVIDENCE_CORE'] = old


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def source_bindings():
    repository=Path(__file__).resolve().parents[2]
    paths=[Path(__file__).resolve()]
    for folder in ['radio_gs/evidence_core','radio_gs/querying','radio_gs/pipeline']:
        paths.extend((repository/folder).glob('*.py'))
    paths.extend((repository/'docs/history/2026-10-18-evidence-core-baseline/radio_gs/querying').glob('*.py'))
    return {str(p.relative_to(repository)):sha(p) for p in sorted(paths)}


def compare(old, new):
    a = old.detach().numpy() if isinstance(old, torch.Tensor) else np.asarray(old)
    b = new.detach().numpy() if isinstance(new, torch.Tensor) else np.asarray(new)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        raise AssertionError('shared core differs from legacy in shape, dtype or values')
    return dict(entries=int(a.size), max_abs_error=0., exact=True)


def audit_uqis(job, extended=False):
    manifest = Path(job['scene']); root = manifest.parent
    store = EvidenceStore.open(manifest)
    m, v, frames = store.registered_relations()
    m = m.tocsr(); v = v.tocsr()
    supported = np.flatnonzero(m.getnnz(axis=1))
    anchors = supported[np.linspace(0, len(supported)-1, 3, dtype=int)]
    vertices = np.unique(np.r_[np.linspace(0, m.shape[0]-1, min(2048, m.shape[0]), dtype=int), anchors])
    reader = DirectAnchorEvidence(m, v, frames, np.load(root/'mesh_xyz.npy'),
                                  np.load(root/'descriptors.npy'), sparse.load_npz(root/'transfer.npz'))
    import joblib
    checkpoint = store.specification['direct_anchor_membership']['checkpoint']
    saved = joblib.load(checkpoint)
    scope = InstanceObservationScope(m, v, frames)
    checks = []
    contracts={}
    extra_assets=[]
    if extended:
        from radio_gs.querying.direct_anchor_membership import DirectAnchorMembership
        from radio_gs.querying.registered_observation_query import public_queries
        from radio_gs.evidence_core.query_contracts import validate_public_query,authorized_crop_path
        original=frozen_module('direct_anchor_membership')
        with profile('legacy'):
            before=original.view_balanced_transfer(m,np.load(root/'region_offsets.npy'))
        after=core.view_balanced_transfer(m,np.load(root/'region_offsets.npy'))
        checks.append(dict(kind='view_balanced_transfer',**compare(before.data,after.data)))
        compare(before.indices,after.indices);compare(before.indptr,after.indptr)
        old_head=original.DirectAnchorMembership(m,v,frames,reader.xyz,reader.descriptors,reader.transfer,checkpoint)
        new_head=DirectAnchorMembership(m,v,frames,reader.xyz,reader.descriptors,reader.transfer,checkpoint,chunk_size=8192)
        for anchor in anchors:
            with profile('legacy'):old_result=old_head.infer(int(anchor))
            with profile('shared_v1'):new_result=new_head.infer(int(anchor))
            for i,key in enumerate(['full_member_probability','full_member_selection']):
                checks.append(dict(kind=key,anchor=int(anchor),**compare(old_result[i],new_result[i])))
        queries,public=public_queries(Path(job['queries']),job['name'])
        extra_assets.extend((public/'queries').glob('*.json'))
        for query in queries:
            validate_public_query(query)
            if query['modality']=='image':extra_assets.append(authorized_crop_path(public,query))
        contracts=dict(queries_validated=len(queries),normalization='view_balanced_nested_regions',
            region_view_provenance=True,independent_visibility=True,member_head_full_carrier=True,
            temporary_member_features_bytes=dict(full=int(m.shape[0]*15*4),chunked=int(min(8192,m.shape[0])*15*4)),
            memory_numbers='feature array sizes only, not measured process peak RSS')
    for anchor in anchors:
        with profile('legacy'):
            old = reader.encode(int(anchor), vertices)
        with profile('shared_v1'):
            new = reader.encode(int(anchor), vertices)
        checks.append(dict(kind='15_member_features', anchor=int(anchor), **compare(old, new)))
        checks.append(dict(kind='retained_classifier_probability', anchor=int(anchor),
                           **compare(saved['estimator'].predict_proba(old), saved['estimator'].predict_proba(new))))
        seed = int(m[anchor].indices[0])
        # Fixed identity and ranking isolate the scope operation from encoders.
        identity = np.zeros(m.shape[1], np.float32); identity[seed] = 1.
        rank = np.linspace(0, 1, m.shape[0], dtype=np.float32)
        with profile('legacy'):
            old_scope = scope.infer(identity, rank, seed)
        with profile('shared_v1'):
            new_scope = scope.infer(identity, rank, seed)
        for k in [0, 1]:
            checks.append(dict(kind='scope_rank', **compare(old_scope[k], new_scope[k])))
        for key in old_scope[2]:
            checks.append(dict(kind=key, **compare(old_scope[2][key], new_scope[2][key])))
    return dict(checks=checks, assets=[manifest, root/'membership.npz', root/'visibility.npz',
        root/'region_offsets.npy', root/'descriptors.npy', root/'transfer.npz', root/'mesh_xyz.npy', Path(checkpoint),*extra_assets],
        sample=dict(anchors=anchors.tolist(), vertices=len(vertices)),map_contract=contracts,
        limits=('Three fixed anchors, full carrier for chunked member readout; identity encoders not rerun.' if extended else
                'All carrier vertices for text scope; sampled member features/head. Identity encoders not rerun.'))


def audit_scannet(job, extended=False):
    scene = read(job['scene'])
    views = scene['source_evidence']['views'][:2]
    scene = dict(scene, source_evidence=dict(scene['source_evidence'], views=views))
    records = [load(x['identity']) for x in views]
    hits = [load(x['contributions']) for x in views]
    n = max(int(h['gaussian_ids'].max()) for h in hits)+1
    texts = records[0]['queries']
    # Declared fixture priors exercise the actual source compiler, not the native
    # teacher encoders. They are not fitted to retained answers or target labels.
    prior = torch.linspace(-1, 1, len(texts)).expand(n, -1).clone()
    auxiliary = -prior
    observed = torch.ones(n, dtype=torch.bool)
    with profile('legacy'):
        old = compile_local_evidence(prior, observed, scene, texts, auxiliary, capture_observations=True)
    with profile('shared_v1'):
        new = compile_local_evidence(prior, observed, scene, texts, auxiliary, capture_observations=True)
    checks = []
    for key in old:
        if isinstance(old[key], torch.Tensor):
            checks.append(dict(kind=key, **compare(old[key], new[key])))
    if not checks:
        raise AssertionError('no source compiler tensors compared')
    assets=[Path(job['scene'])]+[Path(x[k]) for x in views for k in ['identity','contributions']]
    contracts=[]
    if extended:
        from radio_gs.querying.native_observation_prior import compiled_observation_prior
        from radio_gs.evidence_core.store import CompiledRegionEvidence
        from radio_gs.evidence_core.provenance import attach_provenance
        from radio_gs.evidence_core.gaussian_observations import compile_gaussian_view
        # Real retained hits and a fixed prefix of source masks test the build
        # algebra. These are NOT the native 256-wide build inputs or a rebuild.
        for record,hit in zip(records,hits):
            masks=torch.cat([r['masks'] for r in record['instances']])[:3]
            m=(masks>.5).float().flatten(1)
            p,e=hit['pixel_ids'].long(),hit['gaussian_ids'].long()
            weights=hit['base_weights'].float();count=m.sum(0)
            adjoint=torch.sparse_coo_tensor(torch.stack([e,p]),weights*(count[p]>0),(n,m.shape[1])).coalesce()
            expected=torch.sparse.mm(adjoint,(m/count.clamp_min(1)).T)
            expected_mass=torch.sparse.mm(adjoint,torch.ones(m.shape[1],1))
            new_view=compile_gaussian_view(dict(pixel_ids=p,gaussian_ids=e,weights=weights),m,n,retain_visibility=True)
            checks.append(dict(kind='gaussian_build_region_contributions',frame=record['frame'],
                               **compare(expected,new_view.region_contributions)))
            checks.append(dict(kind='gaussian_build_supported_mass',frame=record['frame'],
                               **compare(expected_mass,new_view.supported_mass)))
        original=frozen_module('native_observation_prior')
        auxiliary_scene=read(scene['local_auxiliary_scene']);assets.append(Path(scene['local_auxiliary_scene']))
        auxiliary_scene['native_observation_reliability']=scene['native_observation_reliability']
        auxiliary_scene.pop('observation_weighting',None)
        for spec in [scene,auxiliary_scene]:
            path=Path(spec['native_observation_field']);assets.append(path);payload=load(path)
            query=torch.eye(len(texts),payload['descriptors'].shape[1]);fallback=torch.zeros(payload['shape'][0],len(texts))
            with profile('legacy'):
                before=original.compiled_observation_prior(spec,query,payload['supported'],fallback,field=payload)[0]
            with profile('shared_v1'):
                after=compiled_observation_prior(spec,query,payload['supported'],fallback,field=payload)[0]
            checks.append(dict(kind='actual_compiled_prior',field=str(path),**compare(before,after)))
            if spec.get('native_observation_provenance'):
                reference=spec['native_observation_provenance'];assets.append(Path(reference['path']))
                payload=attach_provenance(payload,path,reference,source_frames=spec.get('source_frames'))
            store=CompiledRegionEvidence.from_payload(payload)
            contracts.append(dict(field=str(path),normalization=store.normalization,
                region_view_provenance=store.region_frames is not None,
                independent_visibility=store.independent_visibility_available))
    return dict(checks=checks, assets=assets,map_contract=contracts,
                sample=dict(source_frames=[x['frame'] for x in views], classes=len(texts)),
                limits='Two source views, actual contributions/masks, fixture priors. Native teachers and benchmark evaluator not rerun.')


def audit_source_domains(job):
    scene = read(job['scene']); checks = []; assets = [Path(job['scene'])]
    frames = scene['source_frames'][:2]
    if job['benchmark'] == 'nvos':
        tracks = load(job['reference_tracks']); assets.append(Path(job['reference_tracks']))
        frames = [f for f in scene['source_frames'] if f in tracks['frames'] and tracks['frames'][f]['object_logit'] > 0][:2]
        h, w = scene['source_fit_hw']; radius = scene['boundary_radius']
    else:
        h, w = 192, 256; radius = 1
    for frame in frames:
        if job['benchmark'] == 'nvos':
            record = tracks['frames'][frame]
            target = F.interpolate(record['logits'].float()[None, None], size=(h, w), mode='bilinear', align_corners=False).sigmoid()
            strength_p = 1.; strength_n = float(record.get('negative_reliability', 1.))
            rows = []
        else:
            path = Path(job['source_evidence'])/f'identity_{frame}.pt'; assets.append(path)
            record = load(path); rows = record['instances']
            target = torch.stack([F.interpolate(r['masks'].float().amax(0)[None,None], size=(h,w),
                mode='bilinear', align_corners=False)[0,0] if len(r['scores']) else torch.zeros(h,w) for r in rows])[:,None]
            strength_p = strength_n = torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0. for r in rows])[None]
        binary = (target > .5).float(); kernel = radius*2+1
        p = (-F.max_pool2d(-binary,kernel,1,radius)>.5).flatten(1).T
        n = (F.max_pool2d(binary,kernel,1,radius)<.5).flatten(1).T
        for qi, row in enumerate(rows):
            unknown = row.get('unknown_masks')
            if unknown is not None and len(unknown):
                n[:,qi] &= ~(F.interpolate(unknown.float().amax(0)[None,None],size=(h,w),mode='bilinear',align_corners=False)[0,0].flatten()>.5)
        # A deterministic domain fixture includes unseen pixels; actual Gaussian
        # projection needs a GPU and belongs to the full inference gate.
        visible = (torch.arange(h*w)%7 != 0)[:,None]
        old = (p*strength_p+n*strength_n)*visible
        domains = core.compile_domains(p,n,visible=visible,negative_origin='model_complement')
        pos, neg = core.signed_weights(domains,strength_p,strength_n)
        checks.append(dict(kind='signed_domain_weights',frame=frame,**compare(old,pos+neg)))
    if not checks:
        raise AssertionError('no active source observation audited')
    return dict(checks=checks, assets=assets, sample=dict(source_frames=frames),
                limits='Actual source masks, fixture visibility. Geometry, selected-set rendering and final scoring not rerun.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe',default='configs/pipeline/system_v10.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--extended',action='store_true',help='Full-carrier chunk checks and actual compiled priors, still no benchmark scoring')
    args = parser.parse_args(); torch.set_num_threads(4)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('audit reports are immutable: '+str(output))
    start = time.monotonic(); rows = []; hashes = {}; sources=source_bindings();recipe_hash=sha(args.recipe)
    for job in read(args.recipe)['jobs']:
        started = time.monotonic()
        try:
            if job['benchmark'] in ('uqis','scannet'):
                result={'uqis':audit_uqis,'scannet':audit_scannet}[job['benchmark']](job,extended=args.extended)
            else:result=audit_source_domains(job)
            for path in result.pop('assets'):
                if str(path) not in hashes:
                    hashes[str(path)] = sha(path)
            row = dict(benchmark=job['benchmark'],name=job['name'],status='passed',**result)
        except Exception as error:
            row = dict(benchmark=job['benchmark'],name=job['name'],status='failed',error=repr(error))
        row['seconds'] = time.monotonic()-started; rows.append(row)
        print(row['benchmark'],row['name'],row['status'],row.get('error',''),flush=True)
    report = dict(schema='radiogs.shared_core_component_audit.v1',recipe=args.recipe,recipe_sha256=recipe_hash,
        fresh_benchmark_evaluation=False,ground_truth_read=False,historical_predictions_read=False,
        all_declared_jobs=len(rows),passed=sum(r['status']=='passed' for r in rows),rows=rows,
        input_sha256=hashes,trace=core.trace_summary(),seconds=time.monotonic()-start)
    report['source_sha256'] = sources
    report['sources_unchanged_during_audit'] = sources == source_bindings() and recipe_hash == sha(args.recipe)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    if report['passed'] != len(rows) or not report['sources_unchanged_during_audit']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
