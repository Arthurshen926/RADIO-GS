"""Fit scope memberships through the frozen source renderer, not adjoint votes.

Each scope uses only its associated source masks. Unobserved views do not
supply negatives. Teacher descriptors, correspondence and geometry stay fixed.
"""
import argparse
import json
from pathlib import Path
import torch


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', required=True)
    parser.add_argument('--source', default='output/scope_complete_next36')
    parser.add_argument('--output', default='output/rendered_scopes_next37')
    parser.add_argument('--steps', type=int, default=80)
    parser.add_argument('--evidence-root')
    args = parser.parse_args()
    torch.set_num_threads(4)
    source = Path(args.source) / args.key
    payload = load(source / 'field.pt')
    graph = payload['scope_hypergraph']
    construction = json.loads((source / 'construction.json').read_text())
    records = construction['records']
    root = Path(payload['root'])
    info = json.loads((root / 'teachers.json').read_text())
    auth = json.loads((root / 'responsibility.json').read_text())
    masks = load(root / 'episodes.pt')['masks']
    extra = Path('output/scope_hierarchy_full_next33') / args.key / 'small_teacher.pt'
    if not extra.exists():
        extra = Path('output/small_region_restoration_next32') / args.key / 'small_teacher.pt'
    if extra.exists():
        masks = torch.cat([masks, load(extra)['masks']])
    authority_root = root
    if args.evidence_root:
        evidence_root = Path(args.evidence_root)
        masks = load(evidence_root / 'masks.pt')['masks']
        authority_root = evidence_root / 'authority'
        auth = json.loads((authority_root / 'responsibility.json').read_text())
    n, count = graph['shape']
    pixels = masks.shape[-2] * masks.shape[-1]
    operators = {}
    for vi in sorted({v for record in records for v in record['views']}):
        hit = load(authority_root / auth['views'][vi]['relative_path'])
        pid, gid = hit['pixel_ids'].long().cuda(), hit['gaussian_ids'].long().cuda()
        weight = hit['base_weights'].float().cuda()
        mass = torch.zeros(pixels, device='cuda').scatter_add_(0, pid, weight)
        op = torch.sparse_coo_tensor(torch.stack([pid, gid]), weight / mass[pid].clamp_min(1e-8), (pixels, n)).coalesce()
        operators[vi] = (op, op.transpose(0, 1).coalesce(), mass > 1e-8)
    initial = torch.sparse_coo_tensor(graph['indices'], graph['weights'], graph['shape']).coalesce()
    out_rows, out_cols, out_values, statistics = [], [], [], []
    for start in range(0, count, 32):
        end = min(count, start + 32)
        indices = initial.indices()
        keep = (indices[1] >= start) & (indices[1] < end)
        index = indices[:, keep].clone(); index[1] -= start
        value = torch.sparse_coo_tensor(index, initial.values()[keep], (n, end-start)).to_dense().cuda()
        evidence = []
        denominator = torch.zeros_like(value)
        for vi, (op, adjoint, visible) in operators.items():
            local, region = [], []
            for j, record in enumerate(records[start:end]):
                if vi in record['views']:
                    local.append(j)
                    region.append(record['regions'][record['views'].index(vi)])
            if not local:
                continue
            columns = torch.tensor(local, device='cuda')
            target = masks[region].flatten(1).float().T.cuda() * visible[:, None]
            colmass = torch.sparse.mm(adjoint, visible.float()[:, None])
            denominator[:, columns] += colmass
            evidence.append((op, adjoint, visible, columns, target))
        def measure():
            mse, ious = [], []
            for op, _, visible, columns, target in evidence:
                prediction = torch.sparse.mm(op, value[:, columns])
                mse.extend(((prediction-target).square().sum(0)/visible.sum().clamp_min(1)).tolist())
                binary = (prediction > .5) & visible[:, None]
                truth = target > .5
                ious.extend(((binary & truth).sum(0)/(binary | truth).sum(0).clamp_min(1)).tolist())
            return {'mse': sum(mse)/len(mse), 'iou': sum(ious)/len(ious), 'pairs': len(ious)}
        before = measure()
        # A^T 1 bounds A^T A for a nonnegative row-stochastic renderer.
        # Accumulate all associated views before the projected gradient step.
        for step in range(args.steps):
            gradient = torch.zeros_like(value)
            for op, adjoint, _, columns, target in evidence:
                residual = target - torch.sparse.mm(op, value[:, columns])
                gradient[:, columns] += torch.sparse.mm(adjoint, residual)
            value.add_(.95 * gradient / denominator.clamp_min(1e-8)).clamp_(0, 1)
        after = measure()
        row, col = torch.where(value > 1e-5)
        out_rows.append(row.cpu()); out_cols.append(col.cpu()+start); out_values.append(value[row, col].cpu())
        statistics.append({'start': start, 'end': end, 'before': before, 'after': after})
        print(args.key, statistics[-1], flush=True)
    graph['indices'] = torch.stack([torch.cat(out_rows), torch.cat(out_cols)])
    graph['weights'] = torch.cat(out_values)
    payload['method'] = 'source-render optimized scope memberships; frozen native semantics'
    construction['membership_refit'] = {
        'source_views': 'each record views only',
        'steps': args.steps,
        'validation_iou_describes': 'original correspondence footprints, before membership refit',
        'quality_weights': 'unchanged original correspondence confidence',
        'evidence_root': args.evidence_root,
        'source_mask_shape': list(masks.shape[-2:]),
    }
    output = Path(args.output)/args.key
    output.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output/'field.pt')
    (output/'construction.json').write_text(json.dumps(construction, indent=2)+'\n')
    report = {'key': args.key, 'steps': args.steps, 'source': str(source), 'batches': statistics,
              'geometry_frozen': True, 'teacher_coordinates_frozen': True, 'target_gt_used': False}
    (output/'membership_fit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
