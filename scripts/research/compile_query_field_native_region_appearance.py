"""Construction-only native mask pooling with exact existing region bindings."""
import argparse
import gc
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from radio_gs.models.native_region_appearance import NativeRegionAppearance, CHECKPOINT
from radio_gs.query_field.observations import source_sam_metadata
from radio_gs.query_field.storage import load_field, save_field, sha256
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from scripts.research.run_query_field_relations import read, write, receipt, check

ROOT = Path('output/query_field_20261011/native_region_appearance_v1')
PARENT = Path('output/query_field_20261011/binary_region_scope_v2')
VARIANTS = ('native_448', 'native_full')


def bound_masks(region, sam, indices, raw_masks):
    """Compose retained relation IDs with native proposal IDs, never SAM row IDs."""
    ids = torch.as_tensor(indices, dtype=torch.long)
    proposals = region['native_proposal_indices'].long()
    if len(ids) != len(ids.unique()) or bool(((ids < 0) | (ids >= len(proposals))).any()):
        raise ValueError('invalid retained region IDs')
    masks = torch.from_numpy(unpack_masks(sam['packed_masks'], sam['mask_shape'][1]).copy())[proposals[ids]]
    low = F.interpolate(masks[:, None].float(), raw_masks.shape[-2:], mode='nearest-exact')[:, 0].bool()
    if not torch.equal(low, raw_masks[ids]):
        raise ValueError('source mask binding changed')
    return masks


@torch.no_grad()
def run(index):
    j = read(PARENT / 'prediction_config.json')['jobs'][index]
    out = ROOT / 'construction' / j['scene']; out.mkdir(parents=True, exist_ok=False)
    field, parent = load_field(j['field'])
    source_path = Path(j['observations']) / 'observations.json'
    source = read(source_path); construction = read(j['construction'])
    original = read(Path(j['construction']).parent / 'field/manifest.json')
    if original['provenance']['observations_sha256'] != sha256(source_path):
        raise ValueError('source observation receipt changed')
    for key in ['configuration', 'scene_configuration', 'dino_checkpoint']:
        check(source[key])
    if sha256(CHECKPOINT) != source['dino_checkpoint']['sha256']:
        raise ValueError('appearance encoder changed')
    config = read(source['configuration']['path']); scene = read(source['scene_configuration']['path'])
    assert config['dino_width'] == 448
    assert source['world_units_per_metre'] == field.world_units_per_metre
    train = set(source['splits']['train_frames']); dev = set(source['splits']['development_frames'])
    rows = {r['frame']: r for r in source['frames']}
    assert len(construction['source_records']) == int(field.region_carrier.view_count) == len(train)
    teacher = NativeRegionAppearance(device='cuda')
    descriptors = {v: [] for v in VARIANTS}; bindings = []; offset = 0
    for vi, binding in enumerate(construction['source_records']):
        frame = binding['frame']; r = rows[frame]; ids = torch.tensor(binding['regions'])
        assert frame in train and frame not in dev and r['split'] == 'train'
        for key in ['rgb', 'region_cache', 'sam_cache']:
            check(r[key])
        path = Path(j['observations']) / f'view_{frame:05d}.pt'
        assert sha256(path) == r['compiled']['sha256']
        raw = torch.load(path, map_location='cpu', weights_only=True)
        region = torch.load(r['region_cache']['path'], map_location='cpu', weights_only=False)
        sam = torch.load(r['sam_cache']['path'], map_location='cpu', weights_only=False)
        assert region['query_independent'] and region['encoder']['path'] == scene['text_encoder']
        assert Path(region['sam_path']).resolve() == Path(r['sam_cache']['path']).resolve()
        assert source_sam_metadata(sam['metadata'])[0] == r['rgb']['sha256']
        assert raw['frame'] == region['frame'] == frame
        stop = offset + len(ids)
        assert torch.equal(field.evidence.views[offset:stop], torch.full((len(ids),), vi))
        assert torch.equal(field.region_carrier.frames[offset:stop], field.evidence.views[offset:stop])
        assert torch.equal(field.evidence.scales[offset:stop], raw['scales_m'][ids])
        assert torch.equal(field.evidence.region_semantic[offset:stop], raw['semantic_descriptors'][ids])
        masks = bound_masks(region, sam, ids, raw['masks'])
        image = Image.open(r['rgb']['path']).convert('RGB')
        assert (image.height, image.width) == tuple(masks.shape[-2:])
        resized = image.resize((448, round(image.height * 448 / image.width)), Image.Resampling.LANCZOS)
        small = F.interpolate(masks[:, None].float(), (resized.height, resized.width), mode='nearest-exact')[:, 0].numpy()
        descriptors['native_448'].append(teacher.encode(resized, small).float().cpu())
        descriptors['native_full'].append(teacher.encode(image, masks.numpy()).float().cpu())
        bindings.append(dict(frame=frame, regions=binding['regions'], native_proposals=region['native_proposal_indices'][ids].tolist(),
            rgb=r['rgb'], sam_cache=r['sam_cache'], region_cache=r['region_cache'], compiled_sha256=r['compiled']['sha256'],
            original_hw=[image.height, image.width], matched_hw=[resized.height, resized.width], exact_downsampled_mask_binding=True))
        offset = stop
        del raw, region, sam, masks
        print('appearance', j['scene'], vi + 1, '/', len(train), flush=True)
    assert offset == len(field.region_carrier.frames)
    del teacher; torch.cuda.empty_cache(); gc.collect()
    # Every other tensor must survive serialization exactly; validate each cold export.
    original_state = field.state_dict(); old_descriptor = field.region_carrier.descriptors
    for variant in VARIANTS:
        value = torch.cat(descriptors[variant])
        assert value.shape == old_descriptor.shape and value.dtype == old_descriptor.dtype
        assert bool(torch.isfinite(value).all()) and bool((value.norm(dim=1) > .99).all())
        field.region_carrier.descriptors = value
        target = ROOT / variant / j['scene'] / 'field'
        exported = save_field(field, target, dict(parent_field_sha256=parent['files']['field.pt']['sha256'],
            observations=receipt(source_path), construction=receipt(j['construction']), checkpoint=receipt(CHECKPOINT),
            compiler=receipt(__file__), native_encoder_code=receipt('radio_gs/models/native_region_appearance.py'),
            source_records=bindings, changed_tensor='region_carrier.descriptors', pooling='native area-weighted patch tokens then normalize',
            variant=variant, matched_resolution=variant == 'native_448', input_resize='LANCZOS RGB / nearest-exact masks' if variant == 'native_448' else 'none',
            query_free=True, target_labels_used=False, benchmark_queries_used=False, training_steps=0,
            additional_3d_supervision='unchanged historical quality model:24 independent ScanNet training rooms', default_changed=False))
        assert exported['tensor_bytes'] == parent['tensor_bytes']
        cold, _ = load_field(target); new = cold.state_dict()
        assert set(new) == set(original_state)
        for key, previous in original_state.items():
            assert torch.equal(new[key], value if key == 'region_carrier.descriptors' else previous), key
        write(target.parent / 'compilation.json', dict(complete=True, variant=variant, scene=j['scene'],
            changed_tensor='region_carrier.descriptors', other_tensors_exact=len(new) - 1,
            tensor_bytes_unchanged=True, cold_reloaded=True, source_frames=len(bindings), regions=len(value),
            descriptor_cosine_to_old_quantiles=np.quantile((old_descriptor * value).sum(1).numpy(), [0, .5, 1]).tolist(),
            manifest=receipt(target / 'manifest.json'), parent_manifest=receipt(Path(j['field']) / 'manifest.json')))
        del cold, new; gc.collect()
        print('exported', variant, j['scene'], flush=True)
    write(out / 'complete.json', dict(complete=True, source_records=bindings, variants=list(VARIANTS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--index', type=int, required=True)
    args = parser.parse_args(); torch.set_num_threads(4); run(args.index)
