"""Seal all three fixed support-domain controls, including any failed scene."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUNS=ROOT/'output/query_field_20261007'


def read(path):return json.loads(Path(path).read_text())


def receipt(path):
    path=Path(path);digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return dict(path=str(path.relative_to(ROOT)),sha256=digest.hexdigest(),bytes=path.stat().st_size)


def run():
    result=dict(schema='radio_gs.supported_anchor_evidence.v1',date='2026-10-07',parent_commit='bf6eaf9',
                additional_training_scenes=['teatime'],benchmark_replacement=False,independent_final_test=False,
                compact_student=False,scenes={},artifacts={},code={},configs={})
    for scene in ['figurines','ramen','teatime']:
        original=RUNS/f'{scene}_18views_v1';candidate=RUNS/f'{scene}_supported_anchor_v1'
        before=read(original/'development_trained.json');after=read(candidate/'development_trained.json')
        first=[r for r in before['rows'] if r['mode'].startswith('point_')]
        second=[r for r in after['rows'] if r['mode'].startswith('point_')]
        if len(first)!=len(second):raise ValueError('point evaluation roster changed')
        for a,b in zip(first,second):
            for key in ['frame','region','mode','scale_condition','status','selected_gaussians',
                        'projected_membership_iou','actual_subset_iou','partial_loss']:
                if a.get(key)!=b.get(key):raise ValueError('registered point evaluation differs')
        if scene=='teatime':
            earlier=RUNS/'supported_anchor_teatime_lerf_v1/teatime_original'
            later=RUNS/'supported_anchor_teatime_lerf_v1/teatime_supported'
        else:
            earlier=RUNS/f'lerf_two_scene_probe_v1/{scene}_18'
            later=RUNS/f'supported_anchor_lerf_v1/{scene}'
        a=read(earlier/'evaluation/evaluation.json')['metrics']
        b=read(later/'evaluation/evaluation.json')['metrics']
        for key in ['ap','iou']:
            if a['native_generic_relevance'][key]!=b['native_generic_relevance'][key]:
                raise ValueError('native response scoring changed')
        old_manifest=read(original/'models/trained/manifest.json')
        manifest=read(candidate/'models/trained/manifest.json')
        if manifest['provenance']['parent_field_sha256']!=old_manifest['files']['field.pt']['sha256']:
            raise ValueError('candidate parent differs')
        if read(earlier/'prediction.json')['field_sha256']!=old_manifest['files']['field.pt']['sha256']:
            raise ValueError('reference prediction uses another parent')
        if read(later/'prediction.json')['field_sha256']!=manifest['files']['field.pt']['sha256']:
            raise ValueError('candidate prediction uses another field')
        smoke=read(candidate/'runtime_smoke.json')
        if not all(smoke['exact_replay'].values()) or smoke['forbidden_open_attempts']:
            raise ValueError('source-denied cold reload failed')
        row=dict(support_audit=read(RUNS/f'{scene}_support_audit_v1/report.json')['summary'],
            parent_field_sha256=old_manifest['files']['field.pt']['sha256'],
            candidate_field_sha256=manifest['files']['field.pt']['sha256'],
            eligible_anchors=manifest['eligible_anchors'],observed_output_elements=manifest['observed'],
            extra_tensor_bytes=manifest['semantic_increment_bytes']-old_manifest['semantic_increment_bytes'],
            total_file_bytes=manifest['files']['field.pt']['bytes'],point_rows_exact=len(first),
            private_before=before['summary'],private_after=after['summary'],public={},cold_reload=smoke['exact_replay'])
        for name,data in [('before',a),('after',b)]:
            row['public'][name]={control:dict(semantic_iou=value['iou'],ap=value['ap'],
                actual_subset_iou=value['selected']['miou'],cases=value['observations']) for control,value in data.items()}
        row['gate']=dict(text_subset_improves=b['shared_membership']['selected']['miou']>a['shared_membership']['selected']['miou'],
            region_image_nonregression=after['summary']['region_image/default']['mean_actual_subset_iou']>=before['summary']['region_image/default']['mean_actual_subset_iou'],
            registered_points_exact=True,native_response_exact=True)
        row['local_gate_passed']=all(row['gate'].values());result['scenes'][scene]=row
        files=[original/'observations/observations.json',original/'models/training.json',original/'models/trained/manifest.json',
               original/'development_trained.json',candidate/'support_package.json',candidate/'models/trained/manifest.json',
               candidate/'development_trained.json',candidate/'runtime_smoke.json',RUNS/f'{scene}_support_audit_v1/report.json']
        for folder in [earlier,later]:files.extend([folder/'prediction.json',folder/'evaluation/evaluation.json'])
        for path in files:result['artifacts'][str(path.relative_to(ROOT))]=receipt(path)
    result['all_three_local_gates_passed']=all(r['local_gate_passed'] for r in result['scenes'].values())
    for name in ['supported_anchor_tests.log','training_support_refactor_tests.log','supported_anchor_integrity.json',
                 'supported_anchor_third_integrity.json','supported_anchor_original_worktree_check.json']:
        path=RUNS/name;result['artifacts'][str(path.relative_to(ROOT))]=receipt(path)
    for path in sorted((ROOT/'radio_gs/query_field').glob('*.py')):
        result['code'][str(path.relative_to(ROOT))]=receipt(path)
    for path in [Path(__file__),ROOT/'tests/test_query_field_supported_anchor.py']:
        result['code'][str(path.relative_to(ROOT))]=receipt(path)
    for path in sorted((ROOT/'configs/query_field').glob('*.json')):
        if any(word in path.name for word in ['support','teatime']):
            result['configs'][str(path.relative_to(ROOT))]=receipt(path)
    target=ROOT/'docs/experiments/2026-10-07-supported-anchor-evidence.json'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:row['gate'] for key,row in result['scenes'].items()},indent=2))


if __name__=='__main__':run()
