"""Seal source track audits and a rejected scale-only repair candidate."""
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    out=Path('paper/artifacts/source_track_audits_seal_20260916.json');assert not out.exists()
    archive=Path('/mnt/pool/sqy/results/RADIO-GS/output/source_track_audit_sources_20260916');archive.mkdir(exist_ok=True)
    names=['scripts/audits/audit_source_track_mask_consistency.py','scripts/audits/audit_source_track_geometry_transport.py','scripts/audits/audit_source_scale_crossfit.py','scripts/audits/visualize_source_track_conflicts.py','scripts/audits/seal_source_track_audits.py','radio_gs/geometry/source_scale_calibration.py','radio_gs/field/moge_geometry_prior.py','radio_gs/v4/geometry/fuse_lerf_moge3.py','tests/test_source_scale_calibration.py']
    sources=[];artifacts=[];checks={}
    for name in names:
        p=Path(name);digest=sha256_file(p);dest=archive/(digest+'.py')
        if not dest.exists():dest.write_bytes(p.read_bytes())
        assert sha256_file(dest)==digest;sources.append(dict(path=name,sha256=digest,snapshot=str(dest)))
    for scene in ['figurines','ramen']:
        for suffix in ['source_track_mask_consistency','source_track_geometry_transport','source_scale_crossfit']:
            p=Path(f'paper/artifacts/{scene}_{suffix}_20260916.json');report=json.loads(p.read_text());assert report['source_only'] and not report['benchmark_data_used']
            for name,digest in report['input_bindings'].items():assert sha256_file(name)==digest
            artifacts.append(dict(path=str(p),sha256=sha256_file(p)))
        root=Path('/mnt/pool/sqy/results/RADIO-GS/output/preserved_source_inventory_20260916')/scene/'exact_render_query_threshold96';r=json.loads((root/'report.json').read_text());assert sha256_file(root/'field.pt')==r['field_sha256'];checks[scene]=dict(mainline_unchanged_sha256=r['field_sha256'],no_new_retrieval_evaluation=True,scale_only_not_promoted=True)
    image=Path('/mnt/pool/sqy/results/RADIO-GS/output/source_track_conflicts_20260916.png');receipt=json.loads(image.with_suffix('.json').read_text());assert sha256_file(image)==receipt['image_sha256']
    for name,digest in receipt['source_images'].items():assert sha256_file(name)==digest
    for p in [image,image.with_suffix('.json')]:artifacts.append(dict(path=str(p),sha256=sha256_file(p)))
    out.write_text(json.dumps(dict(schema='source_track_audits_seal.v1',sources=sources,artifacts=artifacts,checks=checks),indent=2)+'\n');print('sealed',len(sources),'sources',len(artifacts),'artifacts')


if __name__=='__main__':main()
