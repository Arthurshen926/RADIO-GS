"""Verify completed experiment provenance without rewriting earlier authorities."""
import json
from pathlib import Path
import re

from radio_gs.utils.immutable_artifacts import sha256_file


BASE=Path('output/system_gap_next32')
DOC=Path('docs/experiments/2026-09-26-system-gap-and-method-transplants-next32.md')
SEALS=[
    'historical_method_comparison_next32', 'carrier_source_resolution_next32',
    'alpha_weighted_source_fusion_next32', 'nvos_positive_ensemble_next32',
    'pair_ranked_scope_next32', 'source_scope_gradient_audit_next32',
    'directional_scope_gradient_next32', 'small_region_restoration_next32',
    'small_region_restoration_diagnostic_next32', 'fine_pixel_ownership_next32',
    'lerf_readout_sensitivity_next32', 'core_evidence_review_next32',
]
TESTS={
    'tests/test_alpha_weighted_source_evidence_next32.py':3,
    'tests/test_positive_reference_ensemble.py':6,
    'tests/test_disjoint_query_competition.py':6,
    'tests/test_small_region_inventory.py':3,
    'tests/test_directional_scope_gradient_next32.py':7,
    'tests/test_directional_scope_gradient_loss_next32_independent.py':2,
    'tests/test_fine_pixel_ownership.py':6,
    'tests/test_eval_prerendered_lerf_features.py':11,
}


def main():
    BASE.mkdir(exist_ok=False)
    checks=[]; cache={}
    def check(path, digest, origin):
        if path not in cache:
            cache[path]=sha256_file(path)
        assert cache[path]==digest,(origin,path,'immutable binding mismatch')
        checks.append(dict(path=path,sha256=digest,origin=origin))
    def visit(node, origin):
        if isinstance(node, dict):
            if isinstance(node.get('path'),str) and isinstance(node.get('sha256'),str):
                check(node['path'],node['sha256'],origin)
            for key,value in node.items():
                if isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) and '/' in key:
                    check(key,value,origin)
                elif isinstance(value,(dict,list)):
                    visit(value,origin)
        elif isinstance(node,list):
            for item in node:visit(item,origin)
    child={}
    for name in SEALS:
        path=Path('paper/artifacts')/(name+'.json')
        visit(json.loads(path.read_text()),str(path))
        child[str(path)]=sha256_file(path)
    frozen_path=Path('output/frozen_cross_benchmark_method_20260918/source_manifest.json')
    frozen=json.loads(frozen_path.read_text())
    for path,digest in frozen.items():
        assert sha256_file(path)==digest,('frozen source changed',path)
    for pilot in ('lerf_figurines','scannet_scene0000_00'):
        for directory in ('small_region_restoration_next32','fine_pixel_ownership_next32'):
            report=json.loads((Path('output')/directory/pilot/'report.json').read_text())
            if directory=='fine_pixel_ownership_next32':assert not report['source_admitted']
        assert not json.loads((Path('output/pair_ranked_scope_next32')/pilot/'summary.json').read_text())['source_admitted']
        assert not json.loads((Path('output/directional_scope_gradient_next32')/pilot/'summary.json').read_text())['source_admitted']
    readout=json.loads(Path('output/lerf_readout_sensitivity_next32/summary.json').read_text())
    assert readout['complete'] and readout['observations']==208 and not readout['official_protocol_equivalent']
    tests=dict(unique_tests_passed=sum(TESTS.values()),
        evidence='Actual per-component executions during this turn and independent reviewers; final collect-only confirmed44 distinct tests. Collection is not represented as execution. Earlier unrelated/overlapping tests are excluded.',
        files={p:dict(passed=count,sha256=sha256_file(p)) for p,count in TESTS.items()})
    summary=dict(schema='system_gap_and_transplants.next32.v1',complete=True,
        meaning_of_complete='All experiments started in this round completed; not all project issues solved.',
        mainline_promoted=False,all_benchmark_improvement=False,sota_established=False,independent_test=False,
        uqis='paused per user instruction',child_seals=child,
        child_direct_binding_checks=len(checks),child_unique_files=len(cache),
        tests=tests,frozen_sources=dict(path=str(frozen_path),sha256=sha256_file(frozen_path),files=len(frozen),unchanged=True),
        retained_development_percent=dict(lerf2d_iou=49.74079986486096,lerf_selected_projection_iou=33.5676,
            nvos_positive_reference_iou=81.1607927,scannet_provided_geometry_iou_19_15_10=[54.518966,55.666722,69.816645]),
        synthesis_sha256=sha256_file(DOC))
    with (BASE/'binding_verification.json').open('x') as f:json.dump(checks,f,indent=2)
    with (BASE/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
    artifacts=[DOC,Path(__file__),BASE/'summary.json',BASE/'binding_verification.json']
    bindings={str(p):sha256_file(p) for p in artifacts}
    bindings.update(child)
    with Path('paper/artifacts/system_gap_and_transplants_next32.json').open('x') as f:
        json.dump(dict(schema='system_gap_and_transplants.next32.seal.v1',complete=True,mainline_promoted=False,
            independent_test=False,sota_established=False,bindings=bindings),f,indent=2)
    print(json.dumps(dict(complete=True,child_direct_binding_checks=len(checks),child_unique_files=len(cache),
        frozen_sources_unchanged=len(frozen),unique_focused_tests=sum(TESTS.values()),mainline_promoted=False)))


if __name__=='__main__':main()
