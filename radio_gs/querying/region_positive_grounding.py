"""Source-assisted regional identity from positive detections only.

The native distribution fills unknown area. A detection changes a region in
proportion to its covered area; missing detections never supply negative labels.
This is categorical source-assisted readout, not query-independent map storage.
"""
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from radio_gs.scripts.build_sam3_automatic_mask_cache import unpack_masks
from radio_gs.querying.observation_reliability import reliable_observation_response
from radio_gs.querying.typed_evidence import visible_operator


def positive_signal(evidence, texts, device):
    h, w = evidence['raster_hw']
    signal = []
    for text in texts:
        row = evidence['instances'][evidence['queries'].index(text)]
        if len(row['scores']):
            masks = row['masks'].float().to(device)
            core = -F.max_pool2d(-(masks > .5).float()[:, None], 3, 1, 1)[:, 0] > .5
            value = (masks * core * row['scores'].to(device)[:, None, None]).amax(0)
        else:
            value = torch.zeros(h, w, device=device)
        signal.append(value)
    return torch.stack(signal, -1).reshape(h*w, -1)


@torch.inference_mode()
def pixel_positive_grounding(scene, texts, field, fallback):
    """Simple control: project the identical positive pixels without regions."""
    started = time.monotonic()
    device = fallback.device
    numerator = torch.zeros_like(fallback)
    coverage = torch.zeros_like(fallback[:, :1])
    mass = torch.zeros_like(coverage)
    observed = field['supported'].to(device)
    for view in scene['source_evidence']['views']:
        evidence = torch.load(view['identity'], map_location='cpu')
        signal = positive_signal(evidence, texts, device)
        _, evop, visible = visible_operator(torch.load(view['contributions'], map_location='cpu'), observed, len(signal), False)
        weight = visible.float()
        numerator += torch.sparse.mm(evop.T, signal * weight)
        coverage += torch.sparse.mm(evop.T, signal.amax(1, keepdim=True) * weight)
        mass += torch.sparse.mm(evop.T, weight)
    alpha = (coverage / mass.clamp_min(1e-12)).clamp(0, 1)
    distribution = numerator / numerator.sum(1, keepdim=True).clamp_min(1e-12)
    probability = (1-alpha)*fallback.sigmoid() + alpha*distribution
    logits = torch.logit(probability.clamp(torch.finfo(probability.dtype).tiny, 1-torch.finfo(probability.dtype).eps))
    return torch.where(observed[:, None], logits, fallback), dict(kind='positive_pixel_projection_control',
        source_assisted=True, source_frames=len(scene['source_evidence']['views']),
        mean_positive_coverage=float(alpha[observed].mean()), source_masks_read=True,
        semantic_labels_read=False, negative_labels_from_absence=False, additional_teacher_calls=0,
        seconds=time.monotonic()-started)


@torch.inference_mode()
def region_positive_grounding(scene, query, texts, field, fallback):
    started = time.monotonic()
    if scene['source_region_grounding'] == 'pixel_control':
        return pixel_positive_grounding(scene, texts, field, fallback)
    device = query.device
    response = torch.softmax(100 * (F.normalize(field['descriptors'].float().to(device), dim=-1) @ query.T), dim=-1)
    original = response.clone()
    views = {v['frame']: v for v in scene['source_evidence']['views']}
    coverage = []
    offset = 0
    for frame in scene['source_frames']:
        cache = torch.load(Path(scene['source_region_directory']) / f'frame_{frame:05d}.pt', map_location='cpu')
        count = len(cache['descriptors'])
        if frame not in views or count == 0:
            coverage.append(torch.zeros(count, device=device))
            offset += count
            continue
        evidence = torch.load(views[frame]['identity'], map_location='cpu')
        h, w = evidence['raster_hw']
        signal = positive_signal(evidence, texts, device)
        sam = torch.load(cache['sam_path'], map_location='cpu')
        masks = unpack_masks(sam['packed_masks'], sam['mask_shape'][1])[cache['native_proposal_indices'].numpy()].copy()
        membership = F.interpolate(torch.as_tensor(masks).float()[:, None], size=(h, w), mode='nearest-exact')[:, 0].flatten(1).to(device)
        area = membership.sum(1, keepdim=True).clamp_min(1)
        grounded = membership @ signal / area
        # Max across categories counts each source pixel once, even when
        # teacher detections disagree about its identity.
        alpha = (membership @ signal.amax(1, keepdim=True) / area).clamp(0, 1)
        distribution = grounded / grounded.sum(1, keepdim=True).clamp_min(1e-12)
        response[offset:offset+count] = (1-alpha)*response[offset:offset+count] + alpha*distribution
        coverage.append(alpha[:, 0])
        offset += count
    transfer = torch.sparse_csr_tensor(field['crow'].to(device), field['columns'].to(device), field['weights'].to(device), size=field['shape'])
    probability = torch.sparse.mm(transfer, response)
    probability, reliability = reliable_observation_response(transfer, response, probability, field, 'entropy_information')
    logits = torch.logit(probability.clamp(torch.finfo(probability.dtype).tiny, 1-torch.finfo(probability.dtype).eps))
    logits = torch.where(field['supported'].to(device)[:, None], logits, fallback)
    alpha = torch.cat(coverage)
    return logits, dict(kind='positive_area_grounded_region_identity', source_assisted=True,
        source_frames=len(views), regions=len(response), mean_positive_coverage=float(alpha.mean()),
        changed_top1_regions=int((response.argmax(1)!=original.argmax(1)).sum()),
        source_masks_read=True, semantic_labels_read=False, negative_labels_from_absence=False,
        additional_teacher_calls=0, reliability=reliability, seconds=time.monotonic()-started)
