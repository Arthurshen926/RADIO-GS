"""Query input permissions, separate from teacher accuracy or match confidence."""
import hashlib
from pathlib import Path

_COMMON = {'query_id','benchmark_version','scene_id','modality'}
_PUBLIC = {
    'text': {'expression'},
    'image': {'crop_rgb_path','crop_rgb_sha256'},
    'point_2d': {'camera_to_world','camera_intrinsics','raster_size','positive_pixel_uv'},
    'point_3d': {'point_world_xyz'},
}


def validate_public_query(query):
    modality = query.get('modality')
    if modality not in _PUBLIC:
        raise ValueError('unsupported public query modality')
    required = _PUBLIC[modality] | {'query_id','scene_id','modality'}
    if required - query.keys():
        raise ValueError('public query lacks required fields')
    if query.keys() - (_COMMON | _PUBLIC[modality]):
        raise ValueError('public query contains undeclared inputs')
    return dict(modality=modality, source_rgb=False, target_rgb=False,
                query_rgb_crop=modality == 'image',
                query_camera=modality == 'point_2d',
                user_negative_annotation=False)


def authorized_crop_path(public_root, query):
    permissions = validate_public_query(query)
    if not permissions['query_rgb_crop']:
        raise ValueError('this query does not authorize RGB access')
    relative = Path(query['crop_rgb_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('query crop must stay within the public asset package')
    root = Path(public_root).resolve(); path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError('query crop escapes the public asset package')
    if hashlib.sha256(path.read_bytes()).hexdigest() != query['crop_rgb_sha256']:
        raise ValueError('query crop content differs from its declared hash')
    return path
