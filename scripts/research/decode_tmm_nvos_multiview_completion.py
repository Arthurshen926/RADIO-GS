"""N15: matched original/completed reference transport to every source view."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from radio_gs.querying.image_conditioned_response import response_prompt
from radio_gs.querying.signed_reference_affinity import patch_authority, signed_reference_probability
from radio_gs.querying.reference_admission import reference_choices
from radio_gs.querying.coverage_evidence import coverage_evidence
from radio_gs.querying.reference_completion import signed_scribble_points, enforce_reference_scribbles
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model
from scripts.research.decode_tmm_nvos_memory_response import LazyImages


def binary(path):
    x = np.asarray(Image.open(path))
    return np.any(x > 127, axis=2) if x.ndim == 3 else x > 127


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--request', required=True)
    root = Path(parser.parse_args().request).resolve()
    if root.parent != Path('/dev/shm') or root.stat().st_mode & 0o077:
        raise ValueError('private tmpfs required')
    r = json.loads((root / 'request.json').read_text()); torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(r['gpu_memory_fraction'])
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in r['source_paths']}
    prior = np.load(root / 'prior.npy', allow_pickle=False); h, w = prior.shape
    assert (h, w) == (756, 1008)
    pos, neg, seed = [binary(r[k]) for k in ['positive_scribble', 'negative_scribble', 'reference_mask']]
    assert pos.shape == neg.shape == seed.shape == prior.shape and pos.any() and neg.any()
    sys.path.insert(0, r['dino_repository']); from dinov2.hub.backbones import dinov2_vitb14
    dino = dinov2_vitb14(pretrained=False)
    dino.load_state_dict(torch.load(r['dino_checkpoint'], map_location='cpu', weights_only=True), strict=True)
    dino.eval().cuda()
    image = Image.open(r['reference_rgb']).convert('RGB').resize((w, h), Image.Resampling.BILINEAR)
    x = torch.from_numpy(np.array(image)).permute(2, 0, 1).cuda().float() / 255
    mean = torch.tensor([.485, .456, .406], device='cuda')[:, None, None]
    std = torch.tensor([.229, .224, .225], device='cuda')[:, None, None]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        feature = dino.forward_features(((x - mean) / std)[None])['x_norm_patchtokens'][0].float()
    authority = patch_authority(torch.from_numpy(pos), torch.from_numpy(neg), (h // 14, w // 14)).cuda()
    prob, affinity = signed_reference_probability(feature, feature, authority)
    prob = F.interpolate(prob.reshape(1, 1, h // 14, w // 14), size=(h, w), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    del dino, x, feature, authority; gc.collect(); torch.cuda.empty_cache()
    sam = _load_sam3_model(checkpoint_path=r['checkpoint'], device='cuda', confidence_threshold=0,
        dtype='bfloat16', resolution=1008, point_only=True, build_on_cpu=True)
    tracker = sam.model.inst_interactive_predictor.model
    size = tuple(tracker.sam_prompt_encoder.mask_input_size)
    points = signed_scribble_points(pos, neg, r['points_per_sign'])
    records, seeds = {}, {}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        state = sam.set_image(image)
        for arm in ['semantic_memory', 'constrained_memory']:
            prompt = response_prompt(prob, size)
            fallback = affinity['fallback'] is not None or prompt is None
            if fallback:
                output = seed.astype(np.float32); quality = None
            else:
                if arm == 'constrained_memory': prompt.update(points)
                logits, quality, _ = sam.model.predict_inst(state, **prompt, multimask_output=False, return_logits=True)
                assert logits.shape == (1, h, w)
                output = torch.from_numpy(logits[0].astype(np.float32)).sigmoid().numpy()
                quality = float(np.asarray(quality).reshape(-1)[0])
            before = output >= .5
            records[arm] = dict(predicted_quality=quality, affinity_fallback=fallback,
                before_constraint_positive_coverage=float(before[pos].mean()),
                before_constraint_negative_violation=float(before[neg].mean()),
                original_positive_pixels=int(pos.sum()), original_negative_pixels=int(neg.sum()))
            if arm == 'constrained_memory': output = enforce_reference_scribbles(output, pos, neg)
            seeds[arm] = output >= .5
        del state; gc.collect(); torch.cuda.empty_cache()
        tracker.backbone = sam.model.backbone; tracker.use_memory_selection = True
        for arm, mask in seeds.items():
            records[arm].update(positive_coverage=float(mask[pos].mean()), negative_violation=float(mask[neg].mean()))
        choices = reference_choices(records)
        selected = seeds[choices['quality']]
        source_records = {}
        # No target RGB is opened until every source response has been generated.
        for frame in r['source_frames']:
            source_records[str(frame)] = {}
            for branch, mask in [('legacy', seed), ('completed', selected)]:
                if frame == r['reference_frame']:
                    y = torch.from_numpy(mask.astype(np.float32))
                    logits = torch.where(y > .5, 32., -32.)
                    presence_logit = 32.
                else:
                    state = tracker.init_state(video_height=h, video_width=w, num_frames=2,
                        offload_video_to_cpu=True, offload_state_to_cpu=True)
                    state['images'] = LazyImages([r['reference_rgb'], r['source_rgb'][str(frame)]], (w,h), sam.transform)
                    tracker.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=torch.from_numpy(mask))
                    last = None
                    for index, ids, low, masks, presence in tracker.propagate_in_video(state,
                            start_frame_idx=0, max_frame_num_to_track=2, reverse=False, propagate_preflight=True):
                        slot = list(ids).index(1); last = int(index)
                        if last == 1:
                            logits = masks[slot].float().cpu().squeeze()
                            presence_logit = float(torch.as_tensor(presence[slot]).cpu())
                    assert last == 1 and logits.shape == (h,w)
                    del state, low, masks; gc.collect(); torch.cuda.empty_cache()
                y, c = coverage_evidence(logits, r['source_hw'])
                if presence_logit <= 0: c.zero_()
                value = torch.cat([y,c],1).numpy()
                path = root / f'{branch}_{frame}.npy'
                np.save(path, value, allow_pickle=False)
                source_records[str(frame)][branch] = dict(object_logit=presence_logit,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest(), bytes=path.stat().st_size)
            assert sum(p.stat().st_size for p in root.iterdir()) < r['maximum_source_evidence_bytes']
            if len(source_records) % 10 == 0:
                print('transported', len(source_records), '/', len(r['source_frames']), 'source pairs', flush=True)
        state = tracker.init_state(video_height=h, video_width=w, num_frames=2,
            offload_video_to_cpu=True, offload_state_to_cpu=True)
        state['images'] = LazyImages([r['reference_rgb'], r['target_rgb']], (w,h), sam.transform)
        tracker.add_new_mask(inference_state=state, frame_idx=0, obj_id=1, mask=torch.from_numpy(selected))
        scores = []; last = None
        for index, ids, low, masks, presence in tracker.propagate_in_video(state,
                start_frame_idx=0, max_frame_num_to_track=2, reverse=False, propagate_preflight=True):
            slot = list(ids).index(1); last = int(index)
            scores.append(float(torch.as_tensor(presence[slot]).cpu()))
            if last == 1:
                output = masks[slot].float().cpu().squeeze().sigmoid().numpy()
                np.save(root / 'quality_image.npy', output, allow_pickle=False)
        assert last == 1
        records[choices['quality']]['object_logits'] = scores
    (root / 'result.json').write_text(json.dumps(dict(decoder=records, choices=choices, reference_affinity=affinity,
        sources=source_records, source_evidence_before_target_rgb=True,
        selected_reference_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        point_coords=points['point_coords'].tolist(), point_labels=points['point_labels'].tolist(),
        source_hashes=hashes, target_gt_available=False, reference_gt_available=False,
        sam_eval=not sam.model.training, dino_eval=True, negative_scribble_used=True,
        target_dino_features_used=False, memory_selection=True, torch_version=torch.__version__,
        peak_gpu_bytes=torch.cuda.max_memory_allocated()), separators=(',', ':')) + '\n')


if __name__ == '__main__': main()
