"""Compare unit-mean and normalized-pool native region supervision.

Both arms have identical full native capacity, source views, rendering,
initialization and optimization budget. No benchmark query or label is read.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key', required=True)
    p.add_argument('--mode', choices=['unit_mean', 'normalized_pool', 'retrieval_relations'], required=True)
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--output', default='output/region_native_next39')
    a = p.parse_args()
    torch.set_num_threads(4)
    data = load(Path('output/scope_query_modes_next34') / a.key / 'field.pt')
    root = Path(data['root'])
    parent = load(data['base_field'])
    state = parent['state_dict']
    n = len(state['latent'])
    sam = load(root / 'teacher_sam.pt')
    teacher = load(root / 'teacher_siglip.pt')
    target = F.normalize(teacher['descriptors'].float().cuda(), dim=1)
    info = json.loads((root / 'teachers.json').read_text())
    auth = json.loads((root / 'responsibility.json').read_text())
    observed = state['observed'].cuda()
    parts = {'train': [], 'validation': []}
    targets = {'train': [], 'validation': []}
    for vi, frame in enumerate(info['frames']):
        ids = torch.where(sam['view_indices'] == vi)[0]
        masks = sam['masks'][ids].flatten(1).float().cuda()
        hit = load(root / auth['views'][vi]['relative_path'])
        pid, gid = hit['pixel_ids'].long().cuda(), hit['gaussian_ids'].long().cuda()
        weight = hit['base_weights'].float().cuda() * observed[gid]
        mass = torch.zeros(masks.shape[1], device='cuda').scatter_add_(0, pid, weight)
        op = torch.sparse_coo_tensor(torch.stack([gid, pid]), weight / mass[pid].clamp_min(1e-8), (n, masks.shape[1])).coalesce()
        region = torch.sparse.mm(op, masks.T).T
        valid = region.sum(1) > 0
        region = region[valid]
        region /= region.sum(1, keepdim=True)
        split = 'train' if frame in info['train_frames'] else 'validation'
        parts[split].append(region.to_sparse().cpu())
        targets[split].append(target[ids.cuda()][valid].cpu())
        print(a.key, a.mode, 'region operator', frame, len(region), flush=True)
    operators = {k: torch.cat(v).coalesce().cuda() for k, v in parts.items()}
    targets = {k: torch.cat(v).cuda() for k, v in targets.items()}
    del sam, target, region, masks, op, parts
    train = operators['train']
    adjoint = train.transpose(0, 1).coalesce()
    mass = torch.sparse.mm(adjoint, torch.ones(train.shape[0], 1, device='cuda'))
    z = state['latent'][:, :64].cuda()
    initial = F.normalize(F.linear(z, state['native_decoder.weight'].cuda(), state['native_decoder.bias'].cuda()), dim=1)
    value = initial.clone()
    # A fixed source-only bank supervises which regions a query ranks highly.
    # Image/text banks receive equal weight despite their different counts.
    banks = [targets['train'], F.normalize(teacher['text_embeddings'].float().cuda(), dim=1)]
    distributions = [(targets['train'] @ bank.T / .03).softmax(dim=0) for bank in banks] if a.mode == 'retrieval_relations' else []
    history = []
    def measure(step):
        result = {'step': step}
        for split, operator in operators.items():
            pooled = torch.sparse.mm(operator, value)
            result[split] = dict(direction_cosine=float((F.normalize(pooled, dim=1) * targets[split]).sum(1).mean()), mean_norm=float(pooled.norm(dim=1).mean()), unit_mean_error=float((pooled-targets[split]).square().sum(1).mean()))
        history.append(result)
        print(a.key, a.mode, result, flush=True)
    measure(0)
    # Region-normalized evidence mass is the diagonal preconditioner. The same
    # weak original-direction prior and step size apply to both objectives.
    for step in range(a.steps):
        pooled = torch.sparse.mm(train, value)
        if a.mode == 'unit_mean':
            residual = pooled - targets['train']
        else:
            norm = pooled.norm(dim=1, keepdim=True).clamp_min(1e-6)
            unit = pooled / norm
            if a.mode == 'retrieval_relations':
                residual = torch.zeros_like(unit)
                for bank, distribution in zip(banks, distributions):
                    predicted = (unit @ bank.T / .03).softmax(dim=0)
                    residual += ((predicted-distribution) @ bank) * (len(unit)/len(bank)/len(banks))
            else:
                residual = unit - targets['train']
            residual = (residual - unit * (unit * residual).sum(1, keepdim=True)) / norm
        gradient = torch.sparse.mm(adjoint, residual) / mass.clamp_min(1e-8)
        gradient.add_(value-initial, alpha=.05)
        value = F.normalize(value - .1 * gradient, dim=1)
        if (step+1) % 20 == 0:
            measure(step+1)
    data['region_native_features'] = value.half().cpu()
    data['method'] = 'full-native region reconstruction: ' + a.mode
    out = Path(a.output) / a.mode / a.key
    out.mkdir(parents=True, exist_ok=True)
    torch.save(data, out / 'field.pt')
    report = dict(key=a.key, mode=a.mode, steps=a.steps, native_dimensions=value.shape[1], full_native_bytes=data['region_native_features'].numel()*2, history=history, geometry_frozen=True, source_frames=info['train_frames'], validation_frames=info['validation_frames'], benchmark_query_or_gt_used=False, selected_checkpoint='fixed final step', prior=.05, learning_rate=.1, source_query_banks=[len(bank) for bank in banks] if distributions else [], relation_temperature=.03 if distributions else None)
    (out / 'fit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
