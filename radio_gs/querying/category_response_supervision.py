"""Complete native/category responses on a fixed registered training carrier.

Preserves original entropy weighting and detector-matched dual-native fusion.
Training uses registered mesh vertices, not the benchmark Gaussian renderer.
"""
import numpy as np
from scipy.special import softmax, expit


def native_menu(transfer, cosines, keep_regions):
    response = softmax(100*np.asarray(cosines[keep_regions], np.float32), axis=1)
    selected = transfer[:, keep_regions]
    mass = np.asarray(selected.sum(1)).ravel()
    base = np.asarray(selected @ response)/np.maximum(mass[:, None], 1e-12)
    entropy = -(response*np.log(np.maximum(response, np.finfo(np.float32).tiny))).sum(1)
    reliability = np.clip(1-entropy/np.log(response.shape[1]), 0, 1)
    weighted = np.asarray(selected @ reliability).ravel()
    probability = np.asarray(selected @ (response*reliability[:, None]))/np.maximum(weighted[:, None], 1e-12)
    probability[weighted <= 0] = base[weighted <= 0]
    finite = np.finfo(np.float32)
    return probability.clip(finite.tiny, 1-finite.eps).astype(np.float32), mass > 0


def confidence_ratio(records, keep_views, classes):
    square, number = np.zeros(classes), np.zeros(classes)
    for v in keep_views:
        for c, row in enumerate(records[v]['identity']):
            margin = np.asarray(row['margin'])
            square[c] += (margin**2).sum(); number[c] += len(margin)
    scale = np.maximum(np.sqrt(square/np.maximum(number, 1)), 1e-6)
    original, detector = np.zeros(classes), np.zeros(classes)
    for v in keep_views:
        for c, row in enumerate(records[v]['identity']):
            score = np.asarray(row['score']); count = row['known_pixels']
            if len(score):
                original[c] += count*np.max(score*expit(np.asarray(row['margin'])/scale[c]))
                detector[c] += count*np.max(score)
    return (original/np.maximum(detector, 1e-8)).astype(np.float32)


def complete_candidates(siglip, adapter, numerator, mass, visibility, ratio):
    """One fixed pool: dual-native+source incumbent, dual-native, source readout.

    Every candidate is an intact 19-column response. The incumbent reproduces
    compile_local_evidence's average_logits, including its original fallback.
    """
    dual = (siglip+adapter)*.5
    numerator, mass = numerator*ratio[None], mass*ratio[None]
    strength = .1*visibility
    value = ((numerator+strength*dual)/np.maximum(mass+strength, 1e-8)).clip(1e-6, 1-1e-6)
    fused = np.where(mass > 0, value, siglip)
    source = np.where(mass > 0, numerator/np.maximum(mass, 1e-8), siglip)
    return np.stack([fused, dual, source]).clip(0, 1).astype(np.float32)
