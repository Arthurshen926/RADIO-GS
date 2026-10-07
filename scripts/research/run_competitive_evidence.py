"""Fixed frozen exports, signed teacher diagnostics and unsigned LERF audit."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from radio_gs.query_field.contracts import QueryPacket
from radio_gs.query_field.evaluate import choose_pixel, mask_iou
from radio_gs.query_field.geometry import selected_alpha
from radio_gs.query_field.observations import partial_mask_target
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.query_field.support_training import TrainingRays
from scripts.research.prepare_competitive_evidence import ROOT, CONFIG, write


RULES = ['product', 'competitive_pooled', 'competitive_rays']


def verify_launch():
    launch = json.loads((ROOT/'launch.json').read_text())
    for item in launch['code_config_protocol']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('sealed code/configuration changed: '+item['path'])


def package(study, scene):
    models = [m for m in study['models'] if m['scene'] == scene]
    field, parent = load_field(models[0]['parent_field'])
    for model in models:
        field.negative_rule = model['negative_rule']
        manifest = save_field(field, model['field'], dict(
            parent_field_sha256=parent['files']['field.pt']['sha256'],
            observations_sha256=parent['provenance']['observations_sha256'],
            operation='Frozen signed competition; every state tensor retained',
            extra_3d_labels=False, benchmark_queries_used=False))
        reloaded, _ = load_field(model['field'])
        exact = all(torch.equal(value, reloaded.state_dict()[key]) for key, value in field.state_dict().items())
        if not exact or manifest['tensor_bytes'] != parent['tensor_bytes']:
            raise ValueError('frozen export changed tensors or capacity')
        del reloaded
        write(Path(model['output'])/'package.json', dict(all_state_tensors_exact=exact,
              field_sha256=manifest['files']['field.pt']['sha256'],
              parent_sha256=parent['files']['field.pt']['sha256'], tensor_bytes=manifest['tensor_bytes']))
        print('packaged', model['name'], flush=True)


@torch.no_grad()
def diagnostic(study, scene, device):
    job = next(j for j in study['diagnostics'] if j['scene'] == scene)
    field, exported = load_field(job['parent_field'], device)
    root = Path(job['observations'])
    manifest = json.loads((root/'observations.json').read_text())
    rows = []
    height, width = manifest['raster_hw']
    for entry in manifest['frames']:
        if entry['split'] != 'development':
            continue
        path = root/f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry['compiled']['sha256']:
            raise ValueError('teacher observation changed')
        view = torch.load(path, map_location='cpu', weights_only=True)
        rays = TrainingRays(view['operator'])
        operator = view['operator'].to(device)
        best = view['best_element']
        known = (best >= 0) & field.observed.cpu()[best.clamp_min(0)]
        camera = SimpleNamespace(intrinsic=view['intrinsic'], world_to_camera=view['world_to_camera'],
                                 height=view['original_hw'][0], width=view['original_hw'][1])
        for index, mask in enumerate(view['masks']):
            pixel = choose_pixel(mask)
            _, negative_mask = partial_mask_target(view['masks'], index, view['scales_m'])
            candidates = torch.where(negative_mask.flatten() & known)[0]
            positions = torch.linspace(0, len(candidates)-1,
                                       min(study['negative_pixels_per_teacher_region'], len(candidates))).round().long()
            negatives = candidates[positions]
            positive = negative = negative_rays = None
            error = None
            try:
                if pixel is None:
                    raise ValueError('empty source mask')
                positive = rays.support([pixel], device)
                if len(negatives):
                    negative = rays.support(negatives, device)
                    negative_rays = rays.support(negatives, device, preserve_rays=True)
            except ValueError as exc:
                error = str(exc)
            unsigned = None
            for rule in RULES:
                row = dict(frame=view['frame'], region=index, rule=rule, pixel=pixel,
                           negative_pixels=negatives.tolist(), status='ok',
                           actual_subset_iou=0., projected_membership_iou=0.)
                field.negative_rule = rule
                try:
                    if error:
                        raise ValueError(error)
                    current_unsigned = field.query(QueryPacket('point_2d', positive=positive))
                    if unsigned is None:
                        unsigned = current_unsigned.membership_scores
                    if not torch.equal(unsigned, current_unsigned.membership_scores):
                        raise RuntimeError('unsigned teacher query changed')
                    result = field.query(QueryPacket('point_2d', positive=positive,
                        negative=negative_rays if rule == 'competitive_rays' else negative))
                    probability = torch.sparse.mm(operator, result.membership_scores[:, None])[:, 0]
                    target = mask.flatten().to(device)
                    row['projected_membership_iou'] = mask_iou(probability > field.threshold, target)
                    alpha = selected_alpha(field, result.selected, camera, height, width, manifest['world_units_per_metre'])
                    row['actual_subset_iou'] = mask_iou(alpha > .5, target)
                    row['selected'] = int(result.selected.sum())
                except ValueError as exc:
                    row.update(status='unsupported', reason=str(exc))
                rows.append(row)
            if index % 20 == 0:
                print('signed diagnostic', scene, view['frame'], index, flush=True)
        del operator
    summary = {}
    for rule in RULES:
        subset = [r for r in rows if r['rule'] == rule]
        summary[rule] = dict(cases=len(subset), failed=sum(r['status'] != 'ok' for r in subset),
            missing_negative=sum(not r['negative_pixels'] for r in subset),
            **{key: sum(r[key] for r in subset)/len(subset) for key in ['actual_subset_iou', 'projected_membership_iou']})
    write(ROOT/f'{scene}_signed_teacher.json', dict(summary=summary, rows=rows,
          field_sha256=exported['files']['field.pt']['sha256'], observations_sha256=sha256(root/'observations.json'),
          unsigned_membership_exact=True, benchmark_result=False, independent_final_test=False))
    print(summary, flush=True)


@torch.no_grad()
def unsigned_lerf(study, scene, device):
    job = next(j for j in study['unsigned_lerf'] if j['scene'] == scene)
    previous = Path(job['reference_prediction'])
    record = json.loads((previous/'prediction.json').read_text())
    if sha256(previous/'predictions.pt') != record['predictions_sha256']:
        raise ValueError('frozen LERF payload changed')
    data = torch.load(previous/'predictions.pt', map_location='cpu', weights_only=True)
    field, manifest = load_field(job['field'], device)
    for key, value in data['geometry'].items():
        if not torch.equal(getattr(field, key).cpu(), value):
            raise ValueError('geometry changed')
    if not torch.equal(field.observed.cpu(), data['observed']):
        raise ValueError('observed carrier changed')
    background = field.semantic.score(data['generic_negative_vectors']).max(1).values
    rows = []
    for rule in RULES:
        field.negative_rule = rule
        for column, (text, vector) in enumerate(zip(data['texts'], data['native_query_vectors'])):
            result = field.query(QueryPacket('text', task='union', semantic=vector.to(device)))
            exact = dict(membership=torch.equal(result.membership_scores.cpu(), data['scores']['shared_membership'][:, column]),
                         selected=torch.equal(result.selected.cpu(), data['selected'][:, column]),
                         native=torch.equal(torch.sigmoid(10*(result.semantic_scores-background)).cpu(),
                                            data['scores']['native_generic_relevance'][:, column]))
            if not all(exact.values()):
                raise RuntimeError(f'unsigned LERF changed: {scene}/{rule}/{text}/{exact}')
            rows.append(dict(rule=rule, text=text, exact=exact))
        print('unsigned exact', scene, rule, len(data['texts']), flush=True)
    write(ROOT/f'{scene}_unsigned_lerf.json', dict(rows=rows,
          field_sha256=manifest['files']['field.pt']['sha256'],
          reference_prediction_sha256=record['predictions_sha256'], target_labels_opened=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['package', 'diagnostic', 'unsigned'])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    verify_launch()
    study = json.loads((CONFIG/'competitive_evidence_study.json').read_text())
    if args.stage == 'package':
        package(study, args.scene)
    else:
        (diagnostic if args.stage == 'diagnostic' else unsigned_lerf)(study, args.scene, args.device)


if __name__ == '__main__':
    main()
