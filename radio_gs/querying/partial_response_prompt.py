"""Separate positive identity evidence from uncertain background in a response."""
import cv2
import numpy as np

from .image_conditioned_response import response_prompt


def partial_response_prompts(response, mask_hw, points=8):
    """Three matched prompts: signed, positive-only dense, and sparse positive.

    All share a box and identical inferred positive points. The points are not
    new human annotations. Their selection prefers interior, confident and
    spatially distributed foreground locations, without reading labels.
    """
    response = np.asarray(response, dtype=np.float32)
    base = response_prompt(response, mask_hw)
    if base is None:
        return None
    if type(points) is not int or points < 1:
        raise ValueError('positive integer point budget required')
    foreground = response >= .5
    # Padding supplies an exterior even for a mask touching all image borders.
    distance = cv2.distanceTransform(np.pad(foreground.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    yx = np.argwhere(foreground)
    quality = response[foreground] * distance[foreground]
    xy = yx[:, ::-1].astype(np.float32)
    selected = [int(quality.argmax())]
    nearest = np.linalg.norm(xy - xy[selected[0]], axis=1)
    for _ in range(min(points, len(xy))-1):
        score = quality * nearest
        index = int(score.argmax())
        if score[index] <= 0:
            break
        selected.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(xy-xy[index], axis=1))
    common = dict(box=base['box'], point_coords=xy[selected],
                  point_labels=np.ones(len(selected), dtype=np.int32))
    return dict(signed_core=dict(common, mask_input=base['mask_input']),
                positive_core=dict(common, mask_input=np.maximum(base['mask_input'], 0)),
                sparse_core=common)
