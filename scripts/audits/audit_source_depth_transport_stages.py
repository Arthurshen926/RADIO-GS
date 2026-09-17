"""Separate source-prior error from frozen-carrier transport on the same tracks."""
import argparse
import json
from pathlib import Path
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.v4.geometry.fuse_lerf_moge3 import _read_strict_source_points, _bilinear
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geometry-report', type=Path, required=True)
    p.add_argument('--distribution-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    if a.output.exists():
        raise FileExistsError(a.output)
    report = json.loads(a.geometry_report.read_text())
    audit = json.loads(a.distribution_audit.read_text())
    assert audit['geometry_report_sha256'] == sha256_file(a.geometry_report)
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    inputs = {r['role']: r for r in prior.report['geometry_receipt']['inputs']}
    points, _ = _read_strict_source_points(Path(inputs['colmap_sparse_points']['path']),
        {prior.cameras[f].image_id for f in prior.records}, maximum_reprojection_error=2., minimum_track_views=2)
    rows = []
    for frame in sorted({r['frame'] for r in audit['rows']}):
        camera = prior.cameras[frame]
        override = prior.calibration[frame].get('source_geometry_override')
        if override is None:
            raise ValueError('materialized source depth required for this control')
        assert sha256_file(override['path']) == override['sha256']
        data = torch.load(override['path'], weights_only=False)
        indices = {pid: i for i, pid in enumerate(camera.point_ids.tolist()) if pid in points}
        selected = [r for r in audit['rows'] if r['frame'] == frame]
        xy = torch.stack([camera.xy[indices[r['point_id']]] for r in selected])
        xyz = torch.stack([points[r['point_id']] for r in selected])
        reference = xyz @ camera.world_to_camera[2, :3] + camera.world_to_camera[2, 3]
        depth = _bilinear(data['depth'], xy, camera.width, camera.height, pixel_contract='image_edge_v2')
        valid = (_bilinear(data['valid'].float(), xy, camera.width, camera.height, pixel_contract='image_edge_v2') > .999) & (reference > 0)
        error = (depth-reference).abs()/reference.clamp_min(1e-8)
        for i, r in enumerate(selected):
            rows.append(dict(frame=frame, split=r['split'], point_id=r['point_id'],
                prior_covered=bool(valid[i]), carrier_covered=r['covered'],
                prior_absrel=float(error[i]) if valid[i] else None,
                carrier_absrel=r.get('absrel')))
    groups = {}
    for split in sorted({r['split'] for r in rows}):
        rr = [r for r in rows if r['split'] == split]
        common = [r for r in rr if r['prior_covered'] and r['carrier_covered']]
        group = dict(observations=len(rr), prior_covered=sum(r['prior_covered'] for r in rr),
            carrier_covered=sum(r['carrier_covered'] for r in rr), common_covered=len(common))
        for stage in ['prior', 'carrier']:
            x = torch.tensor([r[stage+'_absrel'] for r in common])
            group[stage] = dict(median=float(x.median()), p90=float(x.quantile(.9)), bad_15pct=int((x>.15).sum())) if len(x) else None
        group['prior_good_carrier_bad_15pct'] = sum(r['prior_absrel']<=.15 and r['carrier_absrel']>.15 for r in common)
        group['prior_bad_carrier_bad_15pct'] = sum(r['prior_absrel']>.15 and r['carrier_absrel']>.15 for r in common)
        groups[split] = group
    result = dict(schema='source_depth_transport_stages.v1', source_only=True, benchmark_data_used=False,
        geometry_report_sha256=sha256_file(a.geometry_report), distribution_audit_sha256=sha256_file(a.distribution_audit),
        groups=groups, rows=rows, implementation_sha256=sha256_file(__file__),
        limitations='Sparse source SfM tracks, held out by point ID. Prior uses bilinear depth at continuous image coordinates; carrier uses original floor-raster contribution distribution. Common support only for errors; all support counts retained. Not dense boundary truth.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(groups)


if __name__ == '__main__':
    main()
