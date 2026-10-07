"""Post-study teacher diagnostic: visible inferred support versus member extent.

Uses complete source-development masks only, never public target annotations.
Mask agreement of an anchor's visible footprint is not whole-object 3D identity.
No support, threshold, learned tensor or deployed prediction is selected here.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from radio_gs.query_field.adapters import pixel_support
from radio_gs.query_field.contracts import QueryPacket, Support
from radio_gs.query_field.evaluate import choose_pixel, mask_iou
from radio_gs.query_field.geometry import selected_alpha
from radio_gs.query_field.observations import erode, partial_mask_target
from radio_gs.query_field.storage import load_field, sha256
from scripts.research.prepare_competitive_evidence import ROOT, CONFIG, write


@torch.no_grad()
def audit(scene, device):
    study = json.loads((CONFIG/'competitive_evidence_study.json').read_text())
    job = next(j for j in study['diagnostics'] if j['scene'] == scene)
    field, exported = load_field(job['parent_field'], device)
    root = Path(job['observations'])
    manifest = json.loads((root/'observations.json').read_text())
    reference_path = Path(job['parent_field']).parents[1]/'development_trained.json'
    reference = json.loads(reference_path.read_text())
    old = {(r['frame'], r['region'], r['mode']): r for r in reference['rows'] if r['scale_condition'] == 'default'}
    rows = []
    height, width = manifest['raster_hw']
    for entry in manifest['frames']:
        if entry['split'] != 'development':
            continue
        path = root/f"view_{entry['frame']:05d}.pt"
        if sha256(path) != entry['compiled']['sha256']:
            raise ValueError('teacher observation changed')
        view = torch.load(path, map_location='cpu', weights_only=True)
        operator = view['operator'].to(device).coalesce()
        raw = torch.sparse_coo_tensor(operator.indices(),
            operator.values()*view['raw_mass'].to(device)[operator.indices()[0]], operator.shape).coalesce()
        adjoint = raw.transpose(0, 1).coalesce()
        total = torch.sparse.mm(adjoint, torch.ones(height*width, 1, device=device))[:, 0]
        camera = SimpleNamespace(intrinsic=view['intrinsic'], world_to_camera=view['world_to_camera'],
                                 height=view['original_hw'][0], width=view['original_hw'][1])
        for index, mask in enumerate(view['masks']):
            target = mask.flatten().to(device)
            inside = torch.where(erode(mask).flatten())[0]
            if not len(inside):
                inside = torch.where(mask.flatten())[0]
            positions = torch.linspace(0, len(inside)-1, min(16, len(inside))).round().long()
            image_packet = QueryPacket('image', semantic=view['semantic_descriptors'][index].to(device),
                                       local=view['local_pixels'][inside[positions]].to(device))
            image_result = field.query(image_packet)
            anchor = int(image_result.candidate_indices[image_result.candidate_weights.argmax()])
            inside_mass = torch.sparse.mm(adjoint, target.float()[:, None])[:, 0]
            ratio = float(inside_mass[anchor]/total[anchor]) if total[anchor] > 0 else None
            status = 'not_visible' if ratio is None else 'target_supported' if ratio >= .5 else 'off_target_visible'
            pixel = choose_pixel(mask)
            row = dict(frame=view['frame'], region=index, image_anchor=anchor,
                       anchor_visible_target_mass_fraction=ratio, anchor_status=status,
                       image_iou=0., point_2d_iou=0., point_3d_iou=0., unsupported=[])
            packets = dict(region_image=image_packet)
            try:
                if pixel is None:
                    raise ValueError('empty teacher region')
                packets['point_2d'] = QueryPacket('point_2d', positive=pixel_support(operator, pixel))
            except ValueError:
                row['unsupported'].append('point_2d')
            known_anchor = -1 if pixel is None else int(view['best_element'][pixel])
            if known_anchor >= 0:
                packets['point_3d'] = QueryPacket('point_3d', exact_positive=True,
                    positive=Support(torch.tensor([known_anchor], device=device), torch.ones(1, device=device)))
            else:
                row['unsupported'].append('point_3d')
            for mode, key in [('region_image', 'image_iou'), ('point_2d', 'point_2d_iou'), ('point_3d', 'point_3d_iou')]:
                if mode in packets:
                    try:
                        result = image_result if mode == 'region_image' else field.query(packets[mode])
                        alpha = selected_alpha(field, result.selected, camera, height, width, manifest['world_units_per_metre'])
                        selected = alpha > .5
                        row[key] = mask_iou(selected, target)
                        if mode == 'region_image':
                            _, negative = partial_mask_target(view['masks'], index, view['scales_m'])
                            extra = selected & ~target
                            row['image_false_added_pixels'] = int(extra.sum())
                            row['image_known_negative_false_added_pixels'] = int((extra & negative.flatten().to(device)).sum())
                    except ValueError:
                        row['unsupported'].append(mode)
                if row[key] != old[(view['frame'], index, mode)]['actual_subset_iou']:
                    raise RuntimeError(f'parent teacher diagnostic did not replay: {scene}/{view["frame"]}/{index}/{mode}')
            rows.append(row)
            if index % 20 == 0:
                print('identity/extent audit', scene, view['frame'], index, flush=True)
    summary = {}
    for status in ['all', 'target_supported', 'off_target_visible', 'not_visible']:
        group = rows if status == 'all' else [r for r in rows if r['anchor_status'] == status]
        summary[status] = dict(cases=len(group),
            **{key: sum(r[key] for r in group)/len(group) if group else None
               for key in ['image_iou', 'point_2d_iou', 'point_3d_iou']},
            false_added_pixels=sum(r['image_false_added_pixels'] for r in group),
            known_negative_false_added_pixels=sum(r['image_known_negative_false_added_pixels'] for r in group))
    write(ROOT/f'{scene}_identity_extent_audit.json', dict(rows=rows, summary=summary,
          source_only_poststudy_diagnostic=True, public_target_labels_opened=False,
          field_sha256=exported['files']['field.pt']['sha256'], all_parent_teacher_ious_exact=True,
          reference_development_sha256=sha256(reference_path),
          interpretation='Visible footprint majority is a per-view source-mask association diagnostic, not complete 3D identity or an oracle upper bound.',
          code_sha256=sha256(__file__)))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, choices=['figurines', 'ramen', 'fern'])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    audit(args.scene, args.device)
