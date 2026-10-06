"""Source-heldout response preservation after fixed geometric region pooling."""
import argparse
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--keys', nargs='+', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    def guard(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)) and any(x in str(args[0]) for x in ('.labels.ply','/lerf_ovs/label/','category_evaluation','ranking_evaluation')):
            raise RuntimeError('Target evaluation forbidden in source fusion audit')
    sys.addaudithook(guard)
    for key in a.keys:
        folder = Path('output/native_residual_basis_next30_v2')/key
        mr = json.loads((folder/'map/report.json').read_text())
        root = Path(mr['root'])
        info = json.loads((root/'teachers.json').read_text())
        def load(path, digest):
            assert sha256_file(path)==digest
            return torch.load(path,map_location='cpu',weights_only=False)
        tr = info['teacher_files']['siglip']
        teacher = load(root/tr['path'],tr['sha256'])
        ep = load(root/'episodes.pt',json.loads((root/'episodes.json').read_text())['sha256'])
        assert ep['teacher_manifest_sha256']==sha256_file(root/'teachers.json')
        train = torch.tensor([info['frames'][int(v)] in info['train_frames'] for v in ep['proposal_views']])
        val = torch.tensor([info['frames'][int(v)] in info['validation_frames'] for v in ep['proposal_views']])
        x = F.normalize(teacher['descriptors'][train].cuda().float(),dim=-1)
        text = F.normalize(teacher['text_embeddings'].cuda().float(),dim=-1)
        image = x[torch.linspace(0,len(x)-1,min(512,len(x))).long()]
        target = F.normalize(teacher['descriptors'][val].cuda().float(),dim=-1)
        data = load(folder/'map/map.pt',mr['map_sha256'])
        observed = data['observed']
        mass = ep['region_probability'][val][:,observed].float().cuda()
        supported = mass.sum(1)>1e-8
        weights = mass[supported]/mass[supported].sum(1,keepdim=True)
        target = target[supported]
        metrics = {}
        for name,d in [('native64',64),('native_residual',len(data['basis']))]:
            # Average element cosine over a heldout region, not cosine of an
            # averaged element vector: these are different output contracts.
            pooled = weights @ F.normalize(data['features'][observed,:d].cuda(),dim=-1)
            basis = data['basis'][:d].cuda()
            metrics[name] = {}
            for mode,queries in [('text',text),('image',image)]:
                truth = target @ queries.T
                pred = pooled @ (queries @ basis.T).T
                error = pred-truth
                metrics[name][mode] = dict(mse=float(error.square().mean()),
                    centered_mse=float((error-error.mean(1,keepdim=True)).square().mean()),
                    teacher_top1_agreement=float((truth.argmax(1)==pred.argmax(1)).float().mean()))
        result = dict(key=key,source_only=True,benchmark_queries_used=False,target_gt_used=False,
            diagnostic_not_deployment_gate=True,validation_regions=int(val.sum()),supported_validation_regions=int(supported.sum()),
            metrics=metrics,map_report_sha256=sha256_file(folder/'map/report.json'),
            episode_sha256=sha256_file(root/'episodes.pt'),implementation_sha256=sha256_file(__file__))
        with (folder/'fusion_audit.json').open('x') as f:json.dump(result,f,indent=2)
        print(key,metrics,flush=True)


if __name__=='__main__':main()
