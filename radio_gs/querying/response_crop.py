"""Predict a local decoding window and restore it without erasing the prior.

Only a model response defines the window; no annotation or task identity enters.
Coordinates are integer half-open XYXY in the original image raster.
"""
import math
import numpy as np


def probability_image(value):
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError('finite 2D probabilities required')
    return value


def response_crop_box(response, context_fraction=.25):
    value = probability_image(response)
    if not math.isfinite(context_fraction) or context_fraction < 0:
        raise ValueError('nonnegative finite context fraction required')
    y, x = np.nonzero(value >= .5)
    if not len(x):
        return None
    x0, x1, y0, y1 = int(x.min()), int(x.max())+1, int(y.min()), int(y.max())+1
    dx, dy = math.ceil((x1-x0)*context_fraction), math.ceil((y1-y0)*context_fraction)
    return (max(0,x0-dx), max(0,y0-dy), min(value.shape[1],x1+dx), min(value.shape[0],y1+dy))


def restore_response_crop(prior, local, box):
    value, local = probability_image(prior), probability_image(local)
    x0, y0, x1, y1 = box
    if not (all(type(x) is int for x in box) and 0 <= x0 < x1 <= value.shape[1] and
            0 <= y0 < y1 <= value.shape[0] and local.shape == (y1-y0, x1-x0)):
        raise ValueError('half-open XYXY crop and raster disagree')
    result = value.copy()
    result[y0:y1,x0:x1] = local
    return result
