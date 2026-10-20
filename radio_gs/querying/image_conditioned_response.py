"""Use an existing query response as a prompt, then lift image evidence to 3D.

No target labels or task names enter these operations. Image access is declared
by the caller: an image can be a source-trained RGB render or an observed RGB.
The dense response is a fallible prior, not a user-provided segmentation mask.
"""
import numpy as np
import torch
from torch.nn import functional as F

from .response_reconstruction import AdmittedResponseSystem


def response_prompt(response, mask_hw):
    """Half-open pixel XYXY extent and bounded log-odds in SAM's native grid."""
    value = torch.as_tensor(response, dtype=torch.float32)
    if (value.ndim != 2 or not bool(torch.isfinite(value).all()) or
            bool(((value < 0) | (value > 1)).any())):
        raise ValueError('finite 2D probabilities required')
    if len(mask_hw) != 2 or min(mask_hw) <= 0:
        raise ValueError('positive mask dimensions required')
    yx = torch.nonzero(value >= .5)
    if not len(yx):
        return None
    lo, hi = yx.amin(0), yx.amax(0) + 1
    box = np.array([lo[1], lo[0], hi[1], hi[0]], dtype=np.float32)
    # The native SAM transform warps RGB to a square. The prior uses the same
    # full-frame warp, with no letterbox, crop or width/height exchange.
    logits = torch.logit(value.clamp(1e-4, 1 - 1e-4))
    mask = F.interpolate(logits[None, None], size=tuple(mask_hw), mode='bilinear',
                         align_corners=False)[0].numpy()
    return dict(box=box, mask_input=mask)


def lift_image_response(initial, operator, target, confidence, steps=120):
    """Fit authorized image evidence; preserve every unsupported 3D element.

    This is a view-conditioned update, not a proof of cross-view improvement.
    The source-only field initializes the solve. There is no target GT, new
    geometry, learned feature, or per-scene threshold in this operation.
    """
    if (initial.ndim != 2 or initial.shape[1] != 1 or
            operator.shape[1] != len(initial) or
            target.shape != confidence.shape or target.shape != (operator.shape[0], 1)):
        raise ValueError('response, operator and image evidence shapes differ')
    for value in (initial, target, confidence):
        if not bool(torch.isfinite(value).all()) or bool(((value < 0) | (value > 1)).any()):
            raise ValueError('probability/confidence outside [0,1]')
    operator = operator.coalesce()
    if not bool(torch.isfinite(operator.values()).all()) or bool((operator.values() < 0).any()):
        raise ValueError('nonnegative finite projection required')
    row_mass = torch.sparse.sum(operator, dim=1).to_dense()
    if bool(((row_mass > 1e-7) & ((row_mass - 1).abs() > 1e-5)).any()):
        raise ValueError('projection rows must be normalized or empty')
    system = AdmittedResponseSystem(initial.clone(), [(operator, target, confidence, 1.)])
    system.terms = [(a.to_sparse_csr(), at.to_sparse_csr(), y, w)
                    for a, at, y, w in system.terms]
    value = system.rendered(steps=steps, residual_clip=.25)
    unsupported = system.mass <= 1e-8
    value[unsupported] = initial[unsupported]
    return value, dict(source_image_loss_before=system.objective(initial, .25),
                       source_image_loss_after=system.objective(value, .25),
                       updated_elements=int((value != initial).sum()),
                       unsupported_changed=int((value[unsupported] != initial[unsupported]).sum()))
