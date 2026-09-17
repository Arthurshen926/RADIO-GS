"""Pinned fixed source descriptors for the isolated ADR-0008 M4 control."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from radio_gs.field.moge_geometry_prior import MogeGeometryPrior
from radio_gs.models.native_query_encoders import NativeThreeQueryEncoder
from radio_gs.scripts.build_native_dinov2_exact_mpr_teacher import _load_model, _extract_feature_map
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    geometry_report = base / 'geometry_only_repair_final_20260914' / a.scene / 'report.json'
    report = json.loads(geometry_report.read_text())
    prior = MogeGeometryPrior(report['source_report'], report['source_report_sha256'])
    identity_path = base / 'preserved_source_inventory_20260916' / a.scene / 'native_query_encoder.json'
    identity = json.loads(identity_path.read_text())
    NativeThreeQueryEncoder(identity, a.device)  # Verify pinned file identities.
    out = base / 'geometry_source_dino_20260917' / a.scene
    out.mkdir(parents=True, exist_ok=False)
    model = _load_model(SimpleNamespace(checkpoint=identity['dino_checkpoint'],
                        torchhub_repo=identity['dino_repo'], model_name='dinov2_vitb14'), torch.device(a.device))
    model.eval().requires_grad_(False)
    assert not set(report['mapping_frames']) & set(report['heldout_frames'])
    rows = []
    for frame in report['mapping_frames']:
        r = prior.records[frame]
        image = Path(r['source_image'])
        assert sha256_file(image) == r['source_image_sha256']
        with torch.inference_mode():
            value = _extract_feature_map(model, image, device=torch.device(a.device),
                                         grid_height=46, grid_width=62, patch_size=14)
        path = out / f'frame_{frame:05d}.pt'
        torch.save(value.half().cpu(), path)
        rows.append(dict(frame=frame, image_sha256=r['source_image_sha256'], path=str(path), sha256=sha256_file(path)))
        print(a.scene, frame, flush=True)
    (out / 'manifest.json').write_text(json.dumps(dict(schema='fixed_source_correspondence_features.v1',
        records=rows, model_checkpoint_sha256=sha256_file(identity['dino_checkpoint']),
        geometry_report_sha256=sha256_file(geometry_report), query_encoder_identity_sha256=sha256_file(identity_path),
        mapping_frames=report['mapping_frames'], heldout_frames=report['heldout_frames'],
        heldout_rgb_opened=False, benchmark_data_used=False,
        implementation_sources={p: sha256_file(p) for p in [__file__, 'radio_gs/scripts/build_native_dinov2_exact_mpr_teacher.py']},
    ), indent=2) + '\n')


if __name__ == '__main__':
    main()
