"""Complete scope composition: capacity intervention and native identity control."""
import argparse
import copy
import gc
import subprocess
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.nn import functional as F
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.query_field.scope_topology import ScopeTopology, ScopeTopologyView, compile_scope_topology_view, assemble_scope
from radio_gs.query_field.native_text_device import NativeTextDevice
from radio_gs.query_field.extent_decision_linked import LinkedExtentDecision
from radio_gs.query_field.observations import source_sam_metadata
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from scripts.research.compile_query_field_extent_factors import phase_pixels
from scripts.research.run_query_field_relations import read, write, receipt, check, execute, lanes

ROOT = Path('output/query_field_20261011/scope_topology_v1')
PARENT_FIELDS = Path('output/query_field_20261011/extent_preparation_v1/fields')
PARENT = Path('output/query_field_20261011/extent_decision_v3')
CANONICAL = Path('output/query_field_20261011/canonical_subset_reference_v1')
PROTOCOL = Path('docs/experiments/2026-10-11-scope-topology-protocol.md')
ARMS = ['seed_addition', 'extent_discrete', 'matched_positive_scope', 'native_complete_scope']
CODE = [__file__, 'radio_gs/query_field/scope_topology.py', 'radio_gs/query_field/field.py',
        'radio_gs/query_field/storage.py', 'radio_gs/query_field/native_text_device.py',
        'radio_gs/query_field/scope_confidence.py', 'radio_gs/query_field/extent_preparation.py',
        'radio_gs/query_field/extent_decision.py', 'radio_gs/query_field/extent_decision_linked.py']


def prepare():
    ROOT.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(read(PARENT/'prediction_config.json'))
    cfg.update(semantic_controls=ARMS, status='complete field scope: native control and privileged capacity intervention')
    inputs = []
    for j in cfg['jobs']:
        inputs += [receipt(p) for p in [PARENT/'predictions'/j['name']/'predictions.pt',
                    PARENT/'predictions'/j['name']/'privileged_scope.pt', Path(j['construction']),
                    Path(j['observations'])/'observations.json', PARENT_FIELDS/j['scene']/'field/manifest.json',
                    CANONICAL/'inputs'/j['scene']/'response.pt']]
    write(ROOT/'prediction_config.json', cfg)
    write(ROOT/'design.json', dict(configuration=receipt(ROOT/'prediction_config.json'), protocol=receipt(PROTOCOL),
        code=[receipt(p) for p in CODE], inputs=inputs, max_passes=4, greedy_steps=16,
        implementation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        target_labels_used_before_scoring=False, default_promoted=False, full_five=False))


def design():
    d = read(ROOT/'design.json')
    for r in d['code'] + [d['configuration'], d['protocol']]: check(r)
    return d


@torch.no_grad()
def compile_scene(index):
    d = design(); j = read(ROOT/'prediction_config.json')['jobs'][index]
    out = ROOT/'fields'/j['scene']; out.mkdir(parents=True, exist_ok=False); began = time.monotonic()
    field, parent = load_field(PARENT_FIELDS/j['scene']/'field')
    construction = read(j['construction']); source = read(Path(j['observations'])/'observations.json')
    records = {r['frame']:r for r in source['frames']}; state = {}; bindings = []
    generator = torch.Generator().manual_seed(20261011)
    for vi, binding in enumerate(construction['source_records']):
        frame = binding['frame']; record = records[frame]
        assert frame in source['splits']['train_frames'] and frame not in source['splits']['development_frames']
        for key in ['region_cache','sam_cache']: check(record[key])
        raw_path = Path(j['observations'])/f'view_{frame:05d}.pt'
        assert sha256(raw_path) == record['compiled']['sha256']
        native = torch.load(record['region_cache']['path'], map_location='cpu', weights_only=False)
        sam = torch.load(record['sam_cache']['path'], map_location='cpu', weights_only=False)
        raw = torch.load(raw_path, map_location='cpu', weights_only=True)
        assert native['query_independent'] and native['frame'] == raw['frame'] == frame
        assert source_sam_metadata(sam['metadata'])[0] == record['rgb']['sha256']
        masks = torch.from_numpy(unpack_masks(sam['packed_masks'], sam['mask_shape'][1]).copy())[native['native_proposal_indices'].long()]
        assert torch.equal(F.interpolate(masks.float()[:,None], (96,128), mode='nearest-exact')[:,0].bool(), raw['masks'])
        assert torch.equal(native['descriptors'], raw['semantic_descriptors'])
        binary = F.interpolate(masks.float()[:,None].cuda(), (384,512), mode='bilinear', align_corners=False)[:,0].half() > .5
        retained = torch.zeros(len(binary), dtype=torch.bool); retained[binding['regions']] = True
        pixels = phase_pixels(vi).cpu()
        values = compile_scope_topology_view(binary.cpu(), native['descriptors'], retained, pixels)
        view = ScopeTopologyView(values)
        # Full raster is an independent implementation; include empty, full and random unions.
        conditions = torch.cat([torch.zeros(1,len(binary),dtype=torch.bool), torch.ones(1,len(binary),dtype=torch.bool),
                                torch.rand(8,len(binary),generator=generator) < .2])
        ray_tests = 0
        for accepted in conditions:
            target, positive, negative = view.compose(accepted)
            union = binary[accepted.cuda()].any(0).float()[None,None]
            p = (-F.max_pool2d(-union,3,1,1) > .5).flatten()[pixels.cuda()].cpu()
            n = (F.max_pool2d(union,3,1,1) < .5).flatten()[pixels.cuda()].cpu()
            if not accepted.any(): n.zero_()
            assert torch.equal(target, union.flatten()[pixels.cuda()].cpu())
            assert torch.equal(positive,p) and torch.equal(negative,n)
            ray_tests += len(pixels)
        state.update({f'views.{vi}.'+k:v for k,v in values.items()})
        bindings.append(dict(frame=frame, proposals=len(binary), retained=int(retained.sum()), atoms=len(values['incidence']),
                             rays=len(pixels), exact_composition_ray_tests=ray_tests,
                             region_cache=record['region_cache'], sam_cache=record['sam_cache'], compiled=receipt(raw_path)))
        print('compiled',j['scene'],vi,bindings[-1]['proposals'],bindings[-1]['atoms'],flush=True)
        del masks, binary, raw, sam, native
    field.scope_topology = ScopeTopology(state, field.extent_preparation)
    added = sum(v.numel()*v.element_size() for v in state.values())
    manifest = save_field(field,out/'field',dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
        queries_used=False,target_labels_used=False,source_records=bindings,
        retains_query_independent_segmentation_information=True,source_rasters_stored=False,
        raw_images_cameras_masks_read_at_query=False,default_decoder_changed=False))
    assert manifest['tensor_bytes'] == parent['tensor_bytes']+added
    # Cold reload verifies every inherited tensor, not just a subset of field parameters.
    cold, _ = load_field(out/'field'); previous = field.state_dict(); new = cold.state_dict()
    for key, value in previous.items(): assert torch.equal(value,new[key]),key
    parent_count = sum(not k.startswith('scope_topology.') for k in previous)
    write(out/'compilation.json',dict(complete=True,scene=j['scene'],added_bytes=added,bindings=bindings,
        unchanged_parent_tensors=parent_count,cold_reload_exact=True,field_sha256=manifest['files']['field.pt']['sha256'],
        seconds=time.monotonic()-began,queries_used=False,target_labels_used=False))


@torch.no_grad()
def greedy_scope(view, target, valid, *, retained=False, max_steps=16):
    """Privileged source-pseudo-scope capacity probe; never a standard adapter."""
    atoms = view.incidence.float(); center = view.neighborhood[:,4]
    total = torch.bincount(center, weights=valid.float(), minlength=len(atoms))
    truth = torch.bincount(center, weights=(target & valid).float(), minlength=len(atoms))
    truth_area = truth.sum(); chosen = torch.zeros(atoms.shape[1],device=atoms.device,dtype=torch.bool)
    if not bool(truth_area): return chosen, dict(single_iou=None, greedy_iou=None, selected=0)
    occupied = torch.zeros(len(atoms),device=atoms.device,dtype=torch.bool); inter = truth.new_zeros(())
    area = truth.new_zeros(()); best = truth.new_zeros(()); single = None
    for _ in range(max_steps):
        new = ~occupied
        gain_inter = atoms.T @ (truth * new)
        gain_area = atoms.T @ (total * new)
        score = (inter + gain_inter)/(area + gain_area + truth_area - inter - gain_inter).clamp_min(1)
        allowed = (~chosen) & (view.retained if retained else torch.ones_like(chosen))
        score[~allowed] = -1
        if single is None: single = float(score.max().clamp_min(0))
        index = int(score.argmax())
        if not bool(score[index] > best + 1e-7): break
        chosen[index] = True; occupied |= view.incidence[:,index]
        inter = (truth * occupied).sum(); area = (total * occupied).sum(); best = score[index]
    return chosen, dict(single_iou=single, greedy_iou=float(best), selected=int(chosen.sum()))


@torch.no_grad()
def predict(index):
    d = design(); cfg = read(ROOT/'prediction_config.json'); j = cfg['jobs'][index]
    out = ROOT/'predictions'/j['name']; out.mkdir(parents=True, exist_ok=False)
    field, manifest = load_field(ROOT/'fields'/j['scene']/'field')
    old = torch.load(PARENT/'predictions'/j['name']/'predictions.pt',weights_only=True)
    fixed = torch.load(PARENT/'predictions'/j['name']/'privileged_scope.pt',weights_only=True)
    response = torch.load(CANONICAL/'inputs'/j['scene']/'response.pt',weights_only=True)
    assert old['texts'] == fixed['texts'] == response['texts']
    for key,value in old['geometry'].items(): assert torch.equal(value,getattr(field,key))
    assert torch.equal(old['observed'],field.observed|field.semantic_measure.supported)
    assert fixed['source_frames'] == [r['frame'] for r in read(j['construction'])['source_records']]
    topology = field.scope_topology.cuda(); preparation = field.extent_preparation.cuda()
    native = NativeTextDevice(field,'cuda'); decoder = LinkedExtentDecision(field.extent_factors,len(field.xyz))
    members = {a:old['membership_by_control'][a] for a in ARMS[:2]}
    members.update({a:torch.zeros_like(response['semantic_probability']) for a in ARMS[2:]})
    evidence = {a:[] for a in ARMS[2:]}; diagnostics = []
    for qi,text in enumerate(old['texts']):
        began = time.monotonic()
        with patch('builtins.open',side_effect=AssertionError('query IO')), patch('io.open',side_effect=AssertionError('query IO')), patch('torch.load',side_effect=AssertionError('query load')):
            prior, _ = native(old['native_query_vectors'][qi], old['generic_negative_vectors'])
        error = float((prior-response['semantic_probability'][:,qi]).abs().max())
        bits = int(((prior>.5)!=(response['semantic_probability'][:,qi]>.5)).sum())
        assert error<=2e-6 and bits==0, (error,bits)
        prior = prior.cuda(); scopes = {a:[] for a in ARMS[2:]}; capacity=[]
        for vi,(view,pv) in enumerate(zip(topology.views,preparation.views)):
            ref = {k:fixed['views'][vi][k][:,qi].cuda() if k!='confidence' else fixed['views'][vi][k][qi].cuda()
                   for k in ('target','positive','negative','confidence')}
            accepted, cap = greedy_scope(view, ref['target']>.5, pv.valid, max_steps=d['greedy_steps'])
            _, retained_cap = greedy_scope(view, ref['target']>.5, pv.valid, retained=True,max_steps=d['greedy_steps'])
            with patch('builtins.open',side_effect=AssertionError('query IO')), patch('io.open',side_effect=AssertionError('query IO')), patch('torch.load',side_effect=AssertionError('query load')):
                y,p,_ = view.compose(accepted); p &= pv.valid
                # Only candidate positive/target changes; privileged negative/confidence are fixed
                # except removing contradictions newly introduced by replacement positives.
                scopes['matched_positive_scope'].append(dict(target=y,positive=p,negative=ref['negative'] & ~(y > .5),
                                                              confidence=ref['confidence']))
                a,c,probability = view.native_admission(old['native_query_vectors'][qi],old['generic_negative_vectors'])
                ns = assemble_scope(view,pv,prior,a,c)
            scopes['native_complete_scope'].append(ns)
            target = ref['target']>.5; predicted = ns['target']>.5
            union = int(((target|predicted)&pv.valid).sum()); intersection=int((target&predicted&pv.valid).sum())
            capacity.append(dict(view=vi, full=cap,retained=retained_cap,
                                 native_iou=intersection/union if union else None, native_accepted=int(a.sum()),
                                 source_positive_pixels=int(ref['positive'].sum()), source_confidence=float(ref['confidence']),
                                 replacement_positive_negative_conflicts=int((p&ref['negative']).sum()),
                                 replacement_target_negative_conflicts=int(((y>.5)&ref['negative']).sum()),
                                 accepted=accepted.cpu().tolist()))
        row = dict(query=text, native_prior_max_error=error,native_prior_selection_disagreements=bits,capacity=capacity,arms={})
        for arm in ARMS[2:]:
            ss=scopes[arm]
            with patch('builtins.open',side_effect=AssertionError('query IO')), patch('io.open',side_effect=AssertionError('query IO')), patch('torch.load',side_effect=AssertionError('query load')):
                initial,cost,info=preparation(prior,ss)
            cpu=[{k:v.cpu() for k,v in s.items()} for s in ss]
            result=initial.cpu(); active=info['active_views']
            if active:
                confidence=np.zeros(len(cpu))
                for vi in active: confidence[vi]=float(cpu[vi]['confidence'])
                # Warmed numba kernel may access its code cache; scope/native/preparation
                # functions above are the IO-denied boundary in this experiment.
                selected,report=decoder.solve((result>.5).numpy(),torch.cat([s['target'] for s in cpu]).numpy(),
                    torch.cat([s['positive'] for s in cpu]).numpy(),torch.cat([s['negative'] for s in cpu]).numpy(),
                    confidence,cost.cpu().numpy(),max_passes=d['max_passes'])
                result=torch.from_numpy(selected).float();info.update(report)
            assert not (result>.5)[~old['observed']].any()
            members[arm][:,qi]=result; evidence[arm].append(cpu);row['arms'][arm]=info
        row['seconds']=time.monotonic()-began;diagnostics.append(row)
        write(out/'queries'/f'{qi:03d}.json',row)
        print(j['scene'],qi,text,row['seconds'],{a:len(row['arms'][a]['active_views']) for a in ARMS[2:]},flush=True)
    old.update(scores={a:old['scores']['seed_addition'] for a in ARMS},membership_by_control=members,
               selected_by_control={a:v>.5 for a,v in members.items()},selected=members['native_complete_scope']>.5)
    torch.save(old,out/'predictions.pt');torch.save(evidence,out/'research_scopes.pt')
    write(out/'prediction.json',dict(configuration=cfg,job=j,queries=len(old['texts']),diagnostics=diagnostics,
        predictions_sha256=sha256(out/'predictions.pt'),research_scopes=receipt(out/'research_scopes.pt'),
        field_manifest=receipt(ROOT/'fields'/j['scene']/'field/manifest.json'),target_labels_opened=False,
        native_scope_source_file_access=False, matched_scope_privileged=True,default_promoted=False))


def complete(gpus):
    d=design(); cfg=read(ROOT/'prediction_config.json'); records=[]
    for j in cfg['jobs']:
        p=ROOT/'predictions'/j['name'];r=read(p/'prediction.json')
        check(dict(path=str(p/'predictions.pt'),sha256=r['predictions_sha256']))
        assert not r['target_labels_opened'];records.append(receipt(p/'prediction.json'))
    write(ROOT/'predictions_complete.json',dict(records=records,queries=68,controls=ARMS))
    def score(j,gpu):
        execute(['-m','radio_gs.query_field.public_lerf','score','--config',str(ROOT/'prediction_config.json'),
                 '--job',j['name'],'--output',str(ROOT/'predictions')],ROOT/'logs'/('score_'+j['scene']+'.log'),gpu)
    lanes(cfg['jobs'],gpus,score)
    per={a:{} for a in ARMS}
    for j in cfg['jobs']:
        m=read(ROOT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        old=read(PARENT/'predictions'/j['name']/'evaluation/evaluation.json')['metrics']
        for a in ARMS:
            for key in ['iou','ap','localization','observations']:assert m[a][key]==old['seed_addition'][key]
            if a in ARMS[:2]: assert m[a]['selected']==old[a]['selected']
            per[a][j['scene']]=m[a]['selected']['miou']
    write(ROOT/'summary.json',dict(complete=True,per_scene=per,macro={a:sum(v.values())/4 for a,v in per.items()},
        matched_positive_scope_privileged=True,native_complete_scope_source_free=True,
        predictions_before_scoring=True,default_promoted=False,whole_five_task=False))


def run(gpus):
    def job(i,gpu):
        execute(['-m','scripts.research.run_query_field_scope_topology','compile','--index',str(i)],ROOT/'logs'/f'compile_{i}.log',gpu)
        execute(['-m','scripts.research.run_query_field_scope_topology','predict','--index',str(i)],ROOT/'logs'/f'predict_{i}.log',gpu)
    lanes(list(range(4)),gpus,job)
    complete(gpus)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','compile','predict','run','complete']);p.add_argument('--index',type=int);p.add_argument('--gpus',default='0,5')
    a=p.parse_args();torch.set_num_threads(4)
    if a.stage=='prepare':prepare()
    elif a.stage=='compile':compile_scene(a.index)
    elif a.stage=='predict':predict(a.index)
    elif a.stage=='complete':complete(list(map(int,a.gpus.split(','))))
    else:run(list(map(int,a.gpus.split(','))))
