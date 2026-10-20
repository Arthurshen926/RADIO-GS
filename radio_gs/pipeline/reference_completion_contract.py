"""Explicit signed-reference/image-access contract, distinct from map-only crops."""
import hashlib
from pathlib import Path

ASSETS = {'reference_rgb', 'reference_mask', 'positive_scribble', 'negative_scribble', 'target_rgb'}
PERMISSIONS = dict(reference_rgb=True, target_rgb=True, positive_scribble=True,
                   negative_scribble=True, reference_gt=False, target_gt=False)
KEYS = {'schema', 'policy', 'native_wh', 'solver_steps', 'threshold', 'footprint_backend',
        'points_per_sign', 'gpu_memory_fraction', 'target_frame', 'inputs', 'permissions',
        'image_checkpoint', 'dino_checkpoint', 'dino_repository', 'geometry_fit'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_asset(asset, *, verify=False):
    if (set(asset) != {'path', 'sha256'} or not isinstance(asset['path'], str)
            or not isinstance(asset['sha256'], str) or len(asset['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in asset['sha256'])):
        raise ValueError('asset needs an explicit path and SHA256')
    if verify and sha(asset['path']) != asset['sha256']:
        raise ValueError('reference completion asset hash differs: ' + asset['path'])


def validate_profile(profile, scene, query, *, verify_assets=False):
    if set(profile) != KEYS or profile.get('schema') != 'radiogs.signed_reference_completion.v1':
        raise ValueError('undeclared or missing reference completion input')
    if profile['permissions'] != PERMISSIONS or set(profile['inputs']) != ASSETS:
        raise ValueError('signed reference input permissions differ')
    if (profile['policy'] not in ('feasible', 'quality') or profile['native_wh'] != [1008, 756]
            or profile['solver_steps'] != 120 or profile['threshold'] != .5
            or profile['points_per_sign'] != 16 or profile['footprint_backend'] != 'gsplat_cuda'
            or not 0 < profile['gpu_memory_fraction'] <= 1):
        raise ValueError('N14 migration requires the frozen algorithm recipe')
    if (query.get('intent') != 'reference_instance' or query.get('reference_mask_is_gt') is not False
            or query.get('modality') != 'prompt_2d' or query.get('prompt_raster_wh') != profile['native_wh']):
        raise ValueError('signed completion requires a non-GT reference query')
    for key in ('reference_rgb', 'reference_mask', 'positive_scribble'):
        if Path(query[key]).resolve() != Path(profile['inputs'][key]['path']).resolve():
            raise ValueError('reference completion differs from baseline query: ' + key)
    target = profile['target_frame']
    if (type(target) is not int or target == query['reference_frame'] or
            target in scene['source_frames'] or target in scene['tracking_frames']):
        raise ValueError('target view overlaps source-only initializer')
    expected_fit = Path(scene['geometry']['path']).parent / 'fit.json'
    if Path(profile['geometry_fit']['path']).resolve() != expected_fit.resolve():
        raise ValueError('geometry training provenance belongs to a different carrier')
    if not isinstance(profile['dino_repository'], str):
        raise ValueError('DINO implementation path required')
    for asset in [*profile['inputs'].values(), profile['image_checkpoint'], profile['dino_checkpoint'], profile['geometry_fit']]:
        validate_asset(asset, verify=verify_assets)
    return dict(PERMISSIONS)


def validate_geometry_training(profile, fit):
    if profile['target_frame'] in fit['source_frames'] or profile['target_frame'] in fit['validation_frames']:
        raise ValueError('target RGB was used to fit the initializer geometry')
