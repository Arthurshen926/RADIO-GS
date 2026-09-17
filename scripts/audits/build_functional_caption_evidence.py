"""Re-encode every caption crop/phrase when replacing the aligned teacher.

Source SAM masks, their geometric lifts, and phrase-based groups are reusable;
no descriptor, text prototype, threshold or learned extent head is reusable.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.models.open_clip_native import OpenClipNativeRuntime
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--source-root', type=Path)
    p.add_argument('--root', type=Path)
    p.add_argument('--source-groups', type=Path)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    old = a.source_root or base/'preserved_source_inventory_20260916'/a.scene
    root = a.root or base/'openclip_replacement_20260917'/a.scene
    out = root/'caption_region_evidence.pt'
    if out.exists():
        raise FileExistsError(out)
    for name in ['responsibility.json', 'source_frames.json']:
        assert sha256_file(root/name) == sha256_file(old/name)
    info = json.loads((root/'teachers.json').read_text())
    identity = json.loads((root/'native_query_encoder.json').read_text())
    assert identity['semantic_backend'] == 'open_clip'
    path = old/'caption_region_evidence.pt'
    assert sha256_file(path) == json.loads(path.with_suffix('.json').read_text())['sha256']
    evidence = torch.load(path, weights_only=False)
    grounding = old/'caption_grounded_regions.pt'
    assert sha256_file(grounding) == evidence['grounding_sha256']
    regions = torch.load(grounding, weights_only=False)
    assert evidence['train_frames'] == info['train_frames']
    assert evidence['source_only'] and not evidence['benchmark_data_used']
    source = json.loads((root/'source_frames.json').read_text())
    sources = {r['frame_idx']: r for r in source['frames']}
    runtime = OpenClipNativeRuntime(identity['open_clip'], a.device)
    descriptors = torch.empty(len(regions['records']), 512, dtype=torch.float16)
    coarse = torch.empty_like(descriptors)
    for frame in info['train_frames']:
        src = sources[frame]
        image_path = Path(source['image_dir'])/src['source_file']
        assert sha256_file(image_path) == src['source_sha256']
        image = Image.open(image_path).convert('RGB')
        rgb = np.asarray(image)
        ids = torch.where(regions['proposal_views'] == info['frames'].index(frame))[0]
        for native, destination in [(True, descriptors), (False, coarse)]:
            crops = []
            for i in ids.tolist():
                r = regions['records'][i]
                assert r['frame'] == frame and r['rgb_sha256'] == src['source_sha256']
                mask = (np.unpackbits(r['packed_native'].numpy(), axis=-1)[:, :r['native_shape'][1]].astype(bool)
                        if native else np.asarray(Image.fromarray(regions['masks'][i].numpy()).resize(image.size, Image.Resampling.NEAREST)).astype(bool))
                assert mask.shape == rgb.shape[:2]
                y, x = np.where(mask)
                if not len(y):
                    raise ValueError('empty source crop')
                masked = rgb.copy()
                masked[~mask] = 127
                box = (max(0, int(x.min())-3), max(0, int(y.min())-3), min(image.width, int(x.max())+4), min(image.height, int(y.max())+4))
                crops.append(Image.fromarray(masked).crop(box))
            for start in range(0, len(ids), 8):
                destination[ids[start:start+8]] = runtime.encode_images(crops[start:start+8]).cpu().half()
        print(a.scene, frame, len(ids), flush=True)
    updated = dict(evidence, native_descriptors=descriptors, coarse_descriptors=coarse,
        text_embeddings=runtime.encode_text(evidence['phrases']).cpu().half(),
        teacher_manifest_sha256=sha256_file(root/'teachers.json'),
        native_query_encoder_sha256=sha256_file(root/'native_query_encoder.json'),
        original_evidence_sha256=sha256_file(path),
        geometry_and_mask_lifts_reused=True, native_embeddings_rebuilt=True,
        implementation_sha256=sha256_file(__file__))
    torch.save(updated, out)
    out.with_suffix('.json').write_text(json.dumps(dict(sha256=sha256_file(out), regions=len(descriptors),
        implementation_sha256=sha256_file(__file__), native_embeddings_rebuilt=True), indent=2)+'\n')
    groups_path = a.source_groups or old/'source_membership_balanced32/report.json'
    groups = json.loads(groups_path.read_text())
    assert groups['evidence_sha256'] == sha256_file(path)
    (root/'functional_membership_groups.json').write_text(json.dumps(dict(
        groups=groups['groups'], evidence_sha256=sha256_file(out), original_group_report_sha256=sha256_file(groups_path),
        reuse_reason='Source phrase/mask groups only; all embedding-dependent heads rebuilt.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
