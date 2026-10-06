"""Compact source-only budget and native-mask roundtrip summary."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

OUT=Path('output/carrier_source_resolution_next32')


def main():
    cohort=json.loads(Path('output/native_residual_basis_next30_v2/registration.json').read_text())['cohort']
    history=json.loads(Path('output/historical_method_comparison_next32/final_audit.json').read_text())
    historical_bindings=0
    rows=[]
    for item in cohort:
        if item['benchmark'] not in ('lerf','scannet'):continue
        path=OUT/(item['key']+'.json');d=json.loads(path.read_text())
        assert d['implementation_sha256']==sha256_file('scripts/audits/audit_carrier_source_resolution_next32.py')
        assert d['same_instance_native_index_replay_exact'] and d['area_filter_replay_exact'] and d['parent_prefix_exact']
        if item['benchmark']=='scannet':
            oldpath=d['historical']['authority_path']
            assert sha256_file(oldpath)==history['source_hashes'][oldpath]==d['inputs'][oldpath]
            historical_bindings+=1
        pixel_count=d['current']['source_grid'][0]*d['current']['source_grid'][1]
        masks=d['rows'];low=sum(r['coarse_area']/pixel_count<.002 for r in masks);high=sum(r['coarse_area']/pixel_count>.6 for r in masks)
        rows.append(dict(key=item['key'],benchmark=item['benchmark'],scene=item['scene'],current=d['current'],historical=d['historical'],
            native_mask_shapes=sorted({tuple(v['native_shape']) for v in d['views']}),
            native_proposals=len(masks),low_area_filtered=low,large_area_filtered=high,
            low_area_filtered_fraction=low/len(masks),retained_summary=d['strata']['retained_native'],
            retained_thin_summary=d['strata']['retained_thin_proxy_le2'],
            retained_small_summary=d['strata']['retained_expected_area_lt123'],
            erased_native_masks=d['strata']['all_native']['erased_count'],
            distinct_retained_native_coarse_collisions=len(d['distinct_native_retained_coarse_collisions']),
            appended_regions_not_compared=d['current']['appended_regions_without_native_comparison'],
            report_sha256=sha256_file(path)))
    result=dict(schema='carrier_source_resolution_summary.next32',source_only=True,gpu_used=False,target_gt_used=False,
        same_source_instances=True,cohort_complete=len(rows)==12,rows=rows,
        historical_scannet_parent_bindings_verified=historical_bindings,
        observed_support_contract='next30 native-map observed support, preserving original native coordinates; not all optional teacher/modality support',
        paired_roundtrip='Native SAM Boolean mask -> actual96x128 nearest-exact mask -> original grid nearest-exact. No VFM recognition, learned head, geometry transfer or target GT.',
        thin_subset='Fixed diagnostic 2*area/perimeter <=2 feature-grid pixels; not a semantic part label.',
        area_bins='Native expected source-grid area below123 is approximately below1% image area; no outcome-based bins.',
        interpretation=['Current authority grids are not smaller than the historical exact/MPR grids inspected.',
            'Native SAM proposals below0.2% coarse area are discarded before independent crop encoding.',
            'High source support does not prove geometric position accuracy or semantic boundary separability.',
            'Roundtrip IoU is a deterministic source-mask loss, not a final query metric or absolute model ceiling.',
            'Historical and current geometry and observed-validity definitions differ.'])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
