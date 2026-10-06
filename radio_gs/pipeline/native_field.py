"""One native semantic map for text, image and registered point queries.

Only text uses the stored text calibration. Image and point results are native
similarity rankings, not calibrated masks or independently decoded object sets.
Int8 storage preserves the encoder coordinates instead of learning a new query
projection. It is an optional deployment representation, not a new teacher.
"""
import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from radio_gs.querying.observation_readout import native_identity


def read_spec(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else value


def feature_rows(field, rows, device):
    if field.get('storage') == 'native_int8_per_element':
        return (field['feature_codes'][rows].to(device=device, dtype=torch.float32)
                * field['feature_scale'][rows].to(device=device, dtype=torch.float32))
    return field['features'][rows].to(device=device, dtype=torch.float32)


def feature_bytes(field):
    names = ['feature_codes', 'feature_scale'] if field.get('storage') == 'native_int8_per_element' else ['features']
    return sum(field[k].numel() * field[k].element_size() for k in names)


@torch.inference_mode()
def compress_field(scene, output):
    scene = read_spec(scene)
    field = torch.load(scene['native_field'], map_location='cpu', weights_only=False)
    features = field['features']
    codes = torch.empty_like(features, dtype=torch.int8)
    scale = torch.empty((len(features), 1), dtype=torch.float32)
    squared_error = 0.
    maximum_error = 0.
    for start in range(0, len(features), 8192):
        rows = slice(start, start + 8192)
        original = features[rows].float()
        step = original.abs().amax(1, keepdim=True) / 127
        step = torch.where(step > 0, step, torch.ones_like(step))
        quantized = (original / step).round().clamp(-127, 127).to(torch.int8)
        codes[rows], scale[rows] = quantized, step
        error = original - quantized.float() * step
        squared_error += float(error.square().sum())
        maximum_error = max(maximum_error, float(error.abs().max()))
    result = {k: v for k, v in field.items() if k != 'features'}
    result.update(storage='native_int8_per_element', feature_codes=codes, feature_scale=scale,
                  feature_dimension=features.shape[1], source_field=scene['native_field'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(result, output / 'native_field.pt')
    deployed_scene = dict(scene, native_field=str(output / 'native_field.pt'))
    (output / 'scene.json').write_text(json.dumps(deployed_scene, indent=2) + '\n')
    report = dict(elements=len(features), dimension=features.shape[1],
                  old_feature_bytes=feature_bytes(field), deployed_feature_bytes=feature_bytes(result),
                  maximum_coordinate_error=maximum_error,
                  mean_squared_coordinate_error=squared_error / features.numel(),
                  learned_query_projection=False, encoder_coordinates_changed=False,
                  storage_only=True, source_rgb_or_labels_read=False,
                  scope='feature payload only; geometry, encoders, source RGB and caches are additional costs')
    (output / 'compression.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


@torch.inference_mode()
def score_field(field, encodings, device='cuda'):
    values = []
    query = encodings.to(device=device, dtype=torch.float32)
    for start in range(0, len(field['supported']), 8192):
        rows = slice(start, start + 8192)
        values.append(native_identity(feature_rows(field, rows, device), query).cpu())
    return torch.cat(values) * field['supported'].reshape(-1, 1)


@torch.inference_mode()
def score_scene_field(scene, field, encodings, device='cuda'):
    """Optional same-coordinate scope distribution, shared by all query types.

    The secondary map is explicit deployment state. Late matching is nonlinear
    in the query and is not advertised as one compact feature vector.
    """
    if not scene.get('native_scope_field'):
        return score_field(field, encodings, device)
    scope = torch.load(scene['native_scope_field'], map_location='cpu', weights_only=False)
    query = encodings.to(device=device, dtype=torch.float32)
    values = []
    for start in range(0, len(field['supported']), 8192):
        rows = slice(start, start + 8192)
        a = feature_rows(field, rows, device)
        b = feature_rows(scope, rows, device)
        if scene['scope_readout'] == 'early_mean':
            value = F.normalize(a + b, dim=-1) @ query.T
        else:
            temperature = 20.
            logits = torch.stack([a @ query.T, b @ query.T])
            value = (torch.logsumexp(logits * temperature, dim=0) - math.log(2)) / temperature
        values.append(value.cpu())
    return torch.cat(values) * field['supported'].reshape(-1, 1)


@torch.inference_mode()
def point_encoding(scene, field, query, device):
    from radio_gs.geometry.frozen_gaussian import FrozenGaussianGeometry
    binding = field['geometry']
    geometry = FrozenGaussianGeometry.load(binding['path'], expected_sha256=binding['sha256'], device=device)
    support = field['supported'].reshape(-1).to(device)
    vectors, details = [], []
    if query['modality'] == 'prompt_3d':
        if query['geometry'] != binding:
            raise ValueError('3D query and map must use the same geometry coordinates')
        points = query.get('world_points', [query.get('world_xyz')])
        labels = query.get('point_labels', [1] * len(points))
        for point in points:
            xyz = torch.as_tensor(point, device=device)
            distances = (geometry.xyz - xyz).square().sum(-1)
            distances[~support] = torch.inf
            index = int(distances.argmin())
            vectors.append(feature_rows(field, [index], device)[0])
            details.append(dict(primitive=index, anchor_distance=float(distances[index].sqrt())))
    else:
        from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
        from radio_gs.rendering.contribution_compositor import rasterize_single_view_contributions
        prior = MogeGeometryPrior(scene['camera_report'], scene['camera_report_sha256'])
        camera = prior.cameras[query['reference_frame']]
        height, width = scene.get('source_fit_hw', [192, 256])
        intrinsic = camera.intrinsic.to(device).clone()
        intrinsic[0] *= width / camera.width
        intrinsic[1] *= height / camera.height
        renderer = SimpleNamespace(scaled_intrinsics=lambda w, h: intrinsic,
                                   near_plane=.001 * prior.scale, far_plane=100 * prior.scale)
        hit = rasterize_single_view_contributions(geometry, renderer, camera.world_to_camera.to(device),
                                                 height=height, width=width, include_depths=False)
        points = query['points_normalized']
        labels = query.get('point_labels', [1] * len(points))
        for x, y in points:
            px, py = min(int(x * width), width - 1), min(int(y * height), height - 1)
            active = (hit['pixel_ids'] == py * width + px) & support[hit['gaussian_ids'].long()]
            ids, weights = hit['gaussian_ids'][active].long(), hit['weights'][active].float()
            if not len(ids) or float(weights.sum()) == 0:
                raise ValueError('Clicked ray has no observed map feature; no prompt identity can be read')
            vectors.append((feature_rows(field, ids.cpu(), device) * weights[:, None]).sum(0) / weights.sum())
            details.append(dict(pixel=[px, py], contributing_elements=len(ids)))
    labels = torch.tensor(labels, device=device).bool()
    if not labels.any():
        raise ValueError('A map point query needs a positive identity anchor')
    vectors = F.normalize(torch.stack(vectors), dim=-1)
    encoding = vectors[labels].mean(0)
    if (~labels).any():
        encoding = encoding - vectors[~labels].mean(0)
    return F.normalize(encoding[None], dim=-1), dict(anchors=details, point_labels=labels.int().tolist(),
        interpretation='native feature direction; signed contrast is ranking only, not enforced mask membership')


@torch.inference_mode()
def run_map_query(scene, queries, output, device='cuda'):
    start = time.monotonic()
    torch.set_num_threads(4)
    scene, queries = read_spec(scene), read_spec(queries)
    if isinstance(queries, dict):
        queries = queries['queries'] if 'queries' in queries else [queries]
    field = torch.load(scene['native_field'], map_location='cpu', weights_only=False)
    calibration_kind = scene.get('calibration', {}).get('kind', 'stored_text_calibration')
    calibrated_text = not (calibration_kind.startswith('none') or calibration_kind.startswith('uncalibrated'))
    text_indices = [i for i, q in enumerate(queries) if q.get('modality', 'text') == 'text']
    encodings, details = {}, {}
    if text_indices:
        from radio_gs.models.native_query_encoders import load_siglip_text
        model, tokenizer = load_siglip_text(scene['text_encoder'], device)
        tokens = tokenizer([queries[i]['text'] for i in text_indices], padding='max_length', max_length=64,
                           truncation=True, return_tensors='pt').to(device)
        native = F.normalize(model(**tokens).pooler_output.float(), dim=-1)
        native = F.normalize(native.half().float(), dim=-1)
        for i, value in zip(text_indices, native):
            encodings[i] = value[None]
            details[i] = dict(encoder=scene['text_encoder'], output='stored text calibration' if calibrated_text else 'native cosine rank')
        del model
    vision = None
    for i, query in enumerate(queries):
        modality = query.get('modality', 'text')
        if modality == 'text':
            continue
        if modality == 'image':
            if vision is None:
                from radio_gs.scripts.build_multiscale_sam_mask_aligned_crop_summary_teacher import NativeSiglip2Runtime
                vision = NativeSiglip2Runtime.load(Path(scene['text_encoder']), device=torch.device(device))
            image = Image.open(query['image']).convert('RGB')
            if query.get('foreground_mask'):
                rgb = np.array(image, copy=True)
                rgb[np.asarray(Image.open(query['foreground_mask']).convert('L')) <= 127] = 127
                image = Image.fromarray(rgb)
            image = image.resize((384, 384), Image.Resampling.BICUBIC)
            pixels = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float()[None] / 255
            encodings[i] = F.normalize(vision.encode(pixels).float(), dim=-1)
            details[i] = dict(encoder=scene['text_encoder'], query_image=query['image'],
                              foreground_mask_supplied=bool(query.get('foreground_mask')), output='native cosine rank')
        elif modality in ('prompt_2d', 'prompt_3d'):
            encodings[i], details[i] = point_encoding(scene, field, query, device)
        else:
            raise ValueError('Map-native query modalities: text, image, prompt_2d, prompt_3d')
    encoded = torch.cat([encodings[i] for i in range(len(queries))])
    cosine = score_scene_field(scene, field, encoded, device)
    payload = dict(queries=queries, texts=[q.get('text', q.get('id', str(i))) for i, q in enumerate(queries)],
                   encodings=encoded.cpu(), cosine=cosine, geometry=field['geometry'],
                   observed=field['supported'].reshape(-1),
                   output_contract='native ranking; calibrated response only for text; no selected-set inference',
                   text_query_indices=text_indices, native_space=field['native_space'],
                   matching_function=scene.get('scope_readout', 'native_cosine'),
                   native_scope_field=scene.get('native_scope_field'))
    if text_indices and calibrated_text:
        probability = (math.exp(field['log_scale']) * cosine[:, text_indices] + field['bias']).sigmoid()
        probability *= field['supported'].reshape(-1, 1)
        payload['semantic_probability' if len(text_indices) == len(queries) else 'text_semantic_probability'] = probability
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output / 'response.pt')
    report = dict(scene=scene['scene'], queries=queries, query_details=details,
                  native_field=scene['native_field'], feature_bytes=feature_bytes(field),
                  feature_storage=field.get('storage', 'float16'), source_rgb_read=False,
                  target_rgb_read=False, ground_truth_read=False, historical_predictions_read=False,
                  query_rgb_read=any(q.get('modality') == 'image' for q in queries),
                  subset_inference=False, seconds=time.monotonic() - start)
    if scene.get('native_scope_field'):
        extra = torch.load(scene['native_scope_field'], map_location='cpu', weights_only=False)
        report.update(native_scope_field=scene['native_scope_field'], matching_function=scene['scope_readout'],
                      total_feature_bytes=feature_bytes(field) + feature_bytes(extra),
                      point_anchor='primary native field', compact_single_vector=False)
    (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['compress', 'query'])
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = (compress_field(args.scene, args.output) if args.action == 'compress' else
              run_map_query(args.scene, args.queries, args.output, args.device))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
