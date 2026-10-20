"""One explicit-map runtime for native matching and source-assisted set inference."""
from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time

REPOSITORY = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    scene: str
    queries: str
    output: str
    access: str = 'source-assisted'
    gpu: str = '0'
    native_python: str = '/root/miniconda3/envs/cybersim_agent/bin/python'
    sam_python: str = '/root/miniconda3/envs/sam3/bin/python'
    checkpoint: str = 'checkpoints/sam3_modelscope/sam3.pt'
    native_response: str | None = None
    source_evidence: str | None = None
    reference_tracks: str | None = None
    reference_evidence: str | None = None
    identity_admission: bool = True
    scope_transfer: bool = False
    instance_scope: bool = False
    member_chunk_size: int | None = None
    reference_completion: str | None = None


def _read(path):
    return json.loads(Path(_absolute(path)).read_text())


def _queries(path):
    value = _read(path)
    return value['queries'] if 'queries' in value else [value]


def _absolute(path):
    path = Path(path)
    return str(path if path.is_absolute() else REPOSITORY / path)


def build_plan(config):
    """Describe all providers before execution; no model or result cache is read."""
    scene = _read(config.scene)
    if config.reference_completion:
        from .reference_completion_contract import validate_profile
        queries = _queries(config.queries)
        if (config.access != 'source-assisted' or len(queries) != 1 or
                scene.get('runtime_provider') or scene.get('schema') == 'registered_native_observation_map_v1'):
            raise ValueError('signed reference completion requires one source-assisted reference query')
        validate_profile(_read(config.reference_completion),scene,queries[0])
    if scene.get('schema') == 'registered_native_observation_map_v1':
        if config.access != 'map-only':
            raise ValueError('Registered observation maps use the declared map-only query contract')
        output = Path(config.output).resolve() / 'registered'
        command = [config.native_python, '-m', 'radio_gs.querying.registered_observation_query',
                   '--map', str(Path(_absolute(config.scene)).parent),
                   '--public', _absolute(config.queries), '--output', str(output)]
        if config.scope_transfer:
            command.append('--scope-transfer')
        if config.instance_scope:
            command.append('--instance-scope')
        if config.member_chunk_size is not None:
            if type(config.member_chunk_size) is not int or config.member_chunk_size <= 0:
                raise ValueError('member chunk size must be positive')
            command.extend(['--member-chunk-size',str(config.member_chunk_size)])
        return dict(access=config.access, runtime_provider='registered_native_observation_map',
            scene=config.scene, queries=config.queries, commands=[command], groups=[],
            source_rgb_allowed=False, target_rgb_allowed=False, query_reference_crop_allowed=True,
            inherited_query_predictions=False, readout=str(output),
            response_operator='native region similarity on the same mesh observation map',
            selection_operator=('native instance identity and physical scope; independent boolean element sets'
                                if config.instance_scope else 'point anchor-region co-membership ranking'),
            capability=('text, image, point_2d, point_3d; native mesh ranks; independent sets when instance_scope is enabled; '
                        'ranking and membership are not calibrated probabilities'))
    queries = _queries(config.queries)
    output = Path(config.output).resolve()
    commands, groups = [], []
    normalized_scene = str(output / 'inputs' / 'scene.json')
    if scene.get('runtime_provider') in ('legacy_category_memory', 'native_category_map'):
        provider = scene['runtime_provider']
        if provider == 'native_category_map' and config.access != 'map-only' and not ((scene.get('native_category_head') or scene.get('native_category_prior')) and scene.get('source_evidence')):
            raise ValueError('The native category provider currently supports map-only inference; no online source provider is declared.')
        provider_output = output / 'category'
        command = [config.native_python, 'scripts/research/run_scannet_query.py',
                   '--scene', _absolute(config.scene), '--query', _absolute(config.queries),
                   '--output', str(provider_output), '--gpu', config.gpu,
                   '--source-evidence', 'declared' if config.access == 'source-assisted' else 'none']
        return dict(access=config.access, runtime_provider=provider,
                    scene=config.scene, queries=config.queries, commands=[command], groups=[],
                    query_modalities=[q['modality'] for q in queries],
                    source_rgb_allowed=config.access == 'source-assisted', target_rgb_allowed=False,
                    inherited_query_predictions=False, readout=str(provider_output / 'typed_readout.pt'),
                    provider_report=str(provider_output / 'result.json'),
                    response_operator='categorical logits on declared memory',
                    selection_operator=None,
                    capability='category logits; no independent selected-subset head')

    def stage(name, *arguments, sam=False):
        command = [config.sam_python if sam else config.native_python,
                   '-m', 'radio_gs.pipeline.stage', name, *map(str, arguments)]
        commands.append(command)

    if config.access == 'map-only':
        commands.append([config.native_python, '-m', 'radio_gs.pipeline.map_query', '--scene', normalized_scene, '--queries', _absolute(config.queries),
                         '--output', str(output / 'native')])
        return dict(access=config.access, scene=config.scene, queries=config.queries,
                    query_modalities=[q['modality'] for q in queries], commands=commands,
                    groups=[], native_output=str(output / 'native'), readout=str(output / 'native' / 'response.pt'),
                    response_operator='native cosine; only text has stored probability calibration',
                    selection_operator=None, source_rgb_allowed=False,
                    target_rgb_allowed=False, inherited_query_predictions=False)

    text_queries = [query for query in queries if query['modality'] == 'text']
    if text_queries:
        if any(query['intent'] not in ('category', 'text_union') for query in text_queries):
            raise ValueError('Text source evidence supports category unions; reference identity must be explicitly anchored.')
        text_root = output / 'text'
        text_input = str(output / 'inputs' / 'text.queries.json')
        response = _absolute(config.native_response) if config.native_response else str(text_root / 'native' / 'response.pt')
        evidence = _absolute(config.source_evidence) if config.source_evidence else str(text_root / 'detections')
        if not config.native_response:
            stage('native-text', '--scene', normalized_scene, '--queries', text_input,
                  '--output', text_root / 'native')
        if not config.source_evidence:
            stage('detect-text', '--scene', normalized_scene, '--queries', text_input,
                  '--checkpoint', _absolute(config.checkpoint), '--output', evidence, sam=True)
        inputs = str(text_root / 'inputs.json')
        if config.identity_admission:
            stage('record-identity', '--inputs', inputs, '--output', text_root / 'native_identity',
                  '--heldout-stride', '0', '--record-only')
            stage('admit-identity', '--inputs', inputs, '--identity-record',
                  text_root / 'native_identity' / 'identity.json', '--output', text_root / 'identity_evidence')
            inputs = str(text_root / 'identity_evidence' / 'inputs.json')
        stage('fit-selection', '--inputs', inputs, '--key', scene['scene'], '--output-base',
              text_root / 'sets', '--mode', 'admission', '--scope-initialization', 'source',
              '--pixel-loss', 'balanced', '--support-prior', 'footprint', '--raster-scale', '2', '--steps', '240')
        stage('fit-response', '--inputs', inputs, '--readout', text_root / 'sets' / scene['scene'] / 'typed_readout.pt',
              '--output', text_root / 'response', '--policies', 'native_protected', '--solver', 'backprojection')
        groups.append(dict(kind='text_union', queries=text_queries, response=response, evidence=evidence,
                           readout=str(text_root / 'response' / 'native_protected' / 'typed_readout.pt')))

    reference_queries = [query for query in queries if query['modality'] != 'text']
    if (config.reference_tracks or config.reference_evidence) and len(reference_queries) != 1:
        raise ValueError('One explicit track file identifies one reference query; run other reference queries separately.')
    for index, query in enumerate(reference_queries):
        if query['modality'] not in ('image', 'prompt_2d', 'prompt_3d') or query['intent'] != 'reference_instance':
            raise ValueError('Source-assisted visual/spatial queries currently require reference_instance intent.')
        branch = output / 'reference' / str(index)
        query_input = str(output / 'inputs' / f'reference_{index}.query.json')
        resolved_query = query_input
        if 'reference_rgb' not in query:
            resolved_query = str(branch / 'resolved' / 'query.json')
            stage('resolve-reference', '--scene', normalized_scene, '--query', query_input, '--output', branch / 'resolved')
        tracks = _absolute(config.reference_tracks) if config.reference_tracks else str(branch / 'tracking' / 'tracks.pt')
        if not config.reference_tracks and not config.reference_evidence:
            stage('track-reference', '--scene', normalized_scene, '--query', resolved_query,
                  '--output', branch / 'tracking', sam=True)
        reference_input = ('--evidence', _absolute(config.reference_evidence)) if config.reference_evidence else ('--tracks', tracks)
        stage('fit-reference', '--scene', normalized_scene, '--query', resolved_query,
              *reference_input, '--output', branch)
        stage('fit-selection', '--inputs', branch / 'subset_inputs.json', '--key', scene['scene'],
              '--output-base', branch / 'sets', '--mode', 'admission', '--scope-initialization', 'source',
              '--pixel-loss', 'balanced', '--support-prior', 'footprint', '--steps', '240',
              '--boundary-radius', str(scene.get('boundary_radius', 2)))
        stage('compose-reference', '--response-field', branch / 'field.pt', '--selection-field',
              branch / 'sets' / scene['scene'] / 'typed_readout.pt', '--output', branch / 'typed_readout.pt')
        groups.append(dict(kind='reference_instance', queries=[query], tracks=tracks if not config.reference_evidence else None,
                           evidence=_absolute(config.reference_evidence) if config.reference_evidence else None,
                           readout=str(branch / 'typed_readout.pt')))
    assembled = output / ('base_readout.pt' if config.reference_completion else 'typed_readout.pt')
    commands.append([config.native_python, '-m', 'radio_gs.pipeline.assemble',
                     '--plan', str(output / 'execution.json'), '--output', str(assembled)])
    if config.reference_completion:
        commands.append([config.native_python,'-m','radio_gs.pipeline.reference_completion',
            '--scene',normalized_scene,'--query',str(output/'inputs/reference_0.query.json'),
            '--profile',_absolute(config.reference_completion),'--initial',str(assembled),
            '--output',str(output/'reference_completion'),'--final-output',str(output/'typed_readout.pt'),
            '--sam-python',config.sam_python])
    return dict(access=config.access, scene=config.scene, queries=config.queries,
                query_modalities=[q['modality'] for q in queries], commands=commands, groups=groups,
                source_rgb_allowed=True, target_rgb_allowed=bool(config.reference_completion), inherited_query_predictions=False,
                reference_completion_profile=_absolute(config.reference_completion) if config.reference_completion else None,
                reference_completion_permissions=_read(config.reference_completion)['permissions'] if config.reference_completion else None,
                source_evidence_replayed=bool(config.source_evidence),
                native_query_response_replayed=bool(config.native_response),
                reference_tracks_replayed=bool(config.reference_tracks),
                reference_evidence_replayed=bool(config.reference_evidence),
                identity_admission='conditioned native density' if config.identity_admission else 'explicit provider evidence',
                response_operator='observed-support-normalized full-scene rendering',
                selection_operator='membership > 0.5, actual selected-subset alpha',
                readout=str(output / 'typed_readout.pt'))


def _prepare_inputs(config, plan):
    output = Path(config.output).resolve()
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    scene = _read(config.scene)
    if scene.get('schema') == 'registered_native_observation_map_v1':
        (inputs / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
        return
    queries = _queries(config.queries)
    if config.native_response:
        import torch
        native = torch.load(_absolute(config.native_response), map_location='cpu', weights_only=False)
        requested = [query['text'] for query in queries if query['modality'] == 'text']
        if native['texts'] != requested:
            raise ValueError('Explicit native response must encode these exact runtime texts in this order.')
        del native
    if scene.get('runtime_provider') in ('legacy_category_memory', 'native_category_map'):
        (inputs / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
        return
    # Populate explicit source paths once from the declared authority. No frame
    # subsampling, target-camera lookup, or query-name-specific branch exists.
    if config.access == 'source-assisted':
        if 'source_rgb' not in scene:
            authority = _read(scene['source_rgb_authority'])
            images = {int(row['image_id'].split('_')[-1]): row['path'] for row in authority['images']}
            scene['source_rgb'] = {str(frame): images[frame] for frame in scene['source_frames']}
        if 'source_rgb_authority' not in scene:
            authority_path = inputs / 'source_images.json'
            authority_path.write_text(json.dumps(dict(images=[
                dict(image_id=f"frame_{frame}", path=scene['source_rgb'][str(frame)])
                for frame in scene['source_frames']
            ]), indent=2) + '\n')
            scene['source_rgb_authority'] = str(authority_path)
        if 'geometry' not in scene and any(query['modality'] != 'text' for query in queries):
            import torch
            native = torch.load(_absolute(scene['native_field']), map_location='cpu', weights_only=False)
            scene['geometry'] = native['geometry']
            del native
        scene.setdefault('tracking_frames', list(scene['source_frames']))
        scene.setdefault('source_fit_hw', [192, 256])
        scene.setdefault('boundary_radius', 2)
        scene.setdefault('checkpoint', _absolute(config.checkpoint))
    (inputs / 'scene.json').write_text(json.dumps(scene, indent=2) + '\n')
    text_queries = [dict(query, intent='category') for query in queries if query['modality'] == 'text']
    if text_queries and config.access == 'source-assisted':
        (inputs / 'text.queries.json').write_text(json.dumps(dict(queries=text_queries), indent=2) + '\n')
        text_group = next(group for group in plan['groups'] if group['kind'] == 'text_union')
        text_root = output / 'text'
        text_root.mkdir(parents=True, exist_ok=True)
        (text_root / 'inputs.json').write_text(json.dumps(dict(
            scene=str(inputs / 'scene.json'), response=text_group['response'], detections=text_group['evidence'],
            tracks=[None] * len(text_queries), base_raster=[96, 128],
            evidence_contract='literal category union; absent candidates unknown; no inherited selection',
        ), indent=2) + '\n')
    for index, query in enumerate(query for query in queries if query['modality'] != 'text'):
        (inputs / f'reference_{index}.query.json').write_text(json.dumps(query, indent=2) + '\n')


def run(config):
    """Execute one recipe; evaluation stays outside prediction generation."""
    start = time.monotonic()
    plan = build_plan(config)
    _prepare_inputs(config, plan)
    output = Path(config.output).resolve()
    (output / 'execution.json').write_text(json.dumps(plan, indent=2) + '\n')
    env = dict(os.environ, PYTHONPATH=str(REPOSITORY), CUDA_VISIBLE_DEVICES=config.gpu, OMP_NUM_THREADS='4')
    for command in plan['commands']:
        subprocess.run(command, cwd=REPOSITORY, env=env, check=True)
    result = dict(plan, seconds=time.monotonic() - start, status='complete',
                  evaluation_performed=False, geometry_updated=False, native_features_updated=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
