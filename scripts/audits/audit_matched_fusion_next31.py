"""Correct next30 pooling and isolate source overlap / geometry / cross-view loss."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.training.source_pixel_evidence import source_pixel_evidence_weights
from radio_gs.training.source_region_readback import region_render_weights
from radio_gs.utils.immutable_artifacts import sha256_file


def metrics(pred, truth):
    error = pred - truth
    return dict(mse=float(error.square().mean()),
        centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
        teacher_top1_agreement=float((truth.argmax(1)==pred.argmax(1)).float().mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keys', nargs='+', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = args.device
    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)) and any(x in str(values[0]) for x in (
                '.labels.ply', '/lerf_ovs/label/', 'category_evaluation', 'ranking_evaluation')):
            raise RuntimeError('Benchmark evidence forbidden in matched source audit')
    sys.addaudithook(guard)
    output = Path('output/matched_source_fusion_next31')
    output.mkdir(exist_ok=True)
    for key in args.keys:
        out = output/key
        out.mkdir(exist_ok=False)
        folder = Path('output/native_residual_basis_next30_v2')/key
        mr = json.loads((folder/'map/report.json').read_text())
        root = Path(mr['root'])
        info = json.loads((root/'teachers.json').read_text())
        auth = json.loads((root/'responsibility.json').read_text())
        assert sha256_file(root/'responsibility.json') == info['authority_sha256']
        def load(path, digest):
            assert sha256_file(path) == digest, path
            return torch.load(path, map_location='cpu', weights_only=False)
        teacher = load(root/info['teacher_files']['siglip']['path'], info['teacher_files']['siglip']['sha256'])
        ep = load(root/'episodes.pt', json.loads((root/'episodes.json').read_text())['sha256'])
        assert ep['teacher_manifest_sha256'] == sha256_file(root/'teachers.json')
        data = load(folder/'map/map.pt', mr['map_sha256'])
        assert ep['geometry_binding'] == data['geometry_binding']
        train = torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
        descriptors = F.normalize(teacher['descriptors'].float().to(device),dim=-1)
        train_desc = descriptors[train]
        queries = dict(text=F.normalize(teacher['text_embeddings'].float().to(device),dim=-1),
            image=train_desc[torch.linspace(0,len(train_desc)-1,min(512,len(train_desc))).long()])
        basis = data['basis'].to(device)
        features = data['features'].to(device)
        observed = data['observed'].to(device)
        n = len(observed)
        records = []
        duplicates = []
        predictions = {}
        truths = {}
        def collect(stage, name, value, truth):
            predictions.setdefault(stage, {}).setdefault(name, []).append(value.detach().cpu())
            truths.setdefault(stage, {}).setdefault(name, []).append(truth.detach().cpu())
        # Query-independent counts over every source view, including training.
        for vi, frame in enumerate(info['frames']):
            ids = torch.where(ep['proposal_views'] == vi)[0]
            masks = ep['masks'][ids].flatten(1).float().to(device)
            area = masks.sum(1)
            overlap = masks @ masks.T
            upper = torch.triu(torch.ones_like(overlap,dtype=torch.bool),diagonal=1)
            exact = upper & (overlap == area[:,None]) & (overlap == area[None])
            union = area[:,None]+area[None]-overlap
            containment = upper & (overlap/torch.minimum(area[:,None],area[None]).clamp_min(1) >= .95) & (overlap/union.clamp_min(1) < .95)
            cosine = descriptors[ids] @ descriptors[ids].T
            duplicates.append(dict(frame=frame, split='train' if frame in info['train_frames'] else 'validation',
                regions=len(ids), exact_duplicate_pairs=int(exact.sum()),
                exact_mask_and_descriptor_pairs=int((exact & (cosine > 1-1e-6)).sum()),
                containment_pairs=int(containment.sum()),
                containment_cosine_below_0_8=int((containment & (cosine < .8)).sum()),
                containment_cosine_below_0_5=int((containment & (cosine < .5)).sum())))
            del masks, overlap, cosine
        (out/'overlap.json').write_text(json.dumps(duplicates,indent=2)+'\n')
        print(key,'overlap',sum(r['exact_duplicate_pairs'] for r in duplicates),
            sum(r['exact_mask_and_descriptor_pairs'] for r in duplicates),flush=True)
        for frame in info['validation_frames']:
            vi = info['frames'].index(frame)
            ids = torch.where(ep['proposal_views'] == vi)[0]
            masks = ep['masks'][ids].flatten(1).to(device)
            desc = descriptors[ids]
            rec = auth['views'][vi]
            hit = load(root/rec['relative_path'], rec['sha256'])
            pid = hit['pixel_ids'].long().to(device)
            gid = hit['gaussian_ids'].long().to(device)
            contribution = hit['base_weights'].float().to(device)
            weights, supported, valid = region_render_weights(masks,pid,gid,contribution,observed)
            legacy = ep['region_probability'][ids][:, data['observed']].float().to(device)
            legacy_supported = legacy.sum(1)>1e-8
            legacy /= legacy.sum(1,keepdim=True).clamp_min(1e-8)
            for name,d in [('native64',64),('native_residual',len(basis))]:
                normalized = F.normalize(features[:,:d],dim=-1)
                corrected = weights @ normalized
                old = legacy @ normalized[observed]
                for mode,q in queries.items():
                    truth = desc @ q.T
                    query = q @ basis[:d].T
                    collect('legacy_'+name,mode,old[legacy_supported] @ query.T,truth[legacy_supported])
                    collect('matched_render_'+name,mode,corrected[supported] @ query.T,truth[supported])
            # Same-view region lift is a diagnostic using withheld view evidence,
            # never a deployable map or a train input.
            lift, _ = source_pixel_evidence_weights(masks,pid,gid,contribution,n,partition_overlap=True)
            mass = lift.sum(0)
            local_seen = observed & (mass>1e-6)
            _, _, local_valid = region_render_weights(masks,pid,gid,contribution,local_seen)
            common = valid & local_valid & masks.any(0)
            common_masks = masks & common[None]
            common_counts = common_masks.sum(1)
            common_supported = common_counts > 0
            pooling = common_masks.float() / common_counts[:,None].clamp_min(1)
            map_weights, check_supported, _ = region_render_weights(common_masks,pid,gid,contribution,observed)
            local_weights, _, _ = region_render_weights(common_masks,pid,gid,contribution,local_seen)
            assert torch.equal(common_supported,check_supported)
            partition = masks.float()/masks.sum(0).clamp_min(1)[None]
            pixel_native = F.normalize(partition.T @ desc,dim=-1)
            for mode,q in queries.items():
                truth = desc @ q.T
                collect('ladder_overlap_native',mode,(pooling @ pixel_native @ q.T)[common_supported],truth[common_supported])
            for name,d in [('native64',64),('native_residual',len(basis))]:
                projected = desc @ basis[:d].T
                independent = F.normalize(projected,dim=-1)
                pixel = F.normalize(partition.T @ projected,dim=-1)
                local_features = F.normalize(lift.T @ projected,dim=-1)
                stage_vectors = dict(independent=independent, overlap=pooling @ pixel,
                    single_view_geometry=local_weights @ local_features,
                    cross_view_frozen=map_weights @ F.normalize(features[:,:d],dim=-1))
                for mode,q in queries.items():
                    truth = desc @ q.T
                    query = q @ basis[:d].T
                    for stage,vectors in stage_vectors.items():
                        collect('ladder_'+stage+'_'+name,mode,(vectors @ query.T)[common_supported],truth[common_supported])
            records.append(dict(frame=frame,regions=len(ids),legacy_supported=int(legacy_supported.sum()),
                matched_render_supported=int(supported.sum()),common_ladder_supported=int(common_supported.sum()),
                region_supported_fraction=(common_counts/masks.sum(1).clamp_min(1)).cpu().tolist(),
                unsupported_region_ids=ids[~common_supported.cpu()].tolist()))
            print(key,frame,'regions',len(ids),'common',int(common_supported.sum()),flush=True)
            del lift, local_features, weights, map_weights, local_weights, legacy
        result = dict(key=key,source_only=True,target_gt_used=False,benchmark_queries_used=False,
            diagnostic_only=True,same_view_stages_access_withheld_view_for_diagnosis_only=True,
            protocol='Mean deployed normalized cosine on mask pixels with observed contribution fraction >0.5; ladder uses identical common supported pixels across all stages. Regions without support excluded and listed. Teachers are pseudo evidence, not truth.',
            observations=records,overlap=duplicates,
            metrics={stage:{name:metrics(torch.cat(values),torch.cat(truths[stage][name]))
                for name,values in modalities.items()} for stage,modalities in predictions.items()},
            map_report_sha256=sha256_file(folder/'map/report.json'),episode_sha256=sha256_file(root/'episodes.pt'),
            authority_sha256=info['authority_sha256'],implementation_sha256=sha256_file(__file__),
            readback_implementation_sha256=sha256_file('radio_gs/training/source_region_readback.py'))
        (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(key,'complete',flush=True)


if __name__ == '__main__':
    main()
