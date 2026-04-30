"""ScanNet semantic classes and class-name query fallback.

The legacy ``SCANNET20_CLASSES`` table below uses the contiguous ScanNet20
training ids used by the original RADIO-GS experiments.  OpenGaussian's
ScanNet point-cloud protocol evaluates directly on NYU40 raw label ids; those
ids and helper remapping functions live in the ``OPENGAUSSIAN_*`` constants.
"""

from __future__ import annotations

import numpy as np


SCANNET20_CLASSES = {
    0: "wall",
    1: "floor",
    2: "cabinet",
    3: "bed",
    4: "chair",
    5: "sofa",
    6: "table",
    7: "door",
    8: "window",
    9: "bookshelf",
    10: "picture",
    11: "counter",
    12: "desk",
    13: "curtain",
    14: "refrigerator",
    15: "shower curtain",
    16: "toilet",
    17: "sink",
    18: "bathtub",
    19: "otherfurniture",
}

GROUNDING_QUERIES = {
    "cabinet": 2,
    "bed": 3,
    "chair": 4,
    "sofa": 5,
    "table": 6,
    "door": 7,
    "window": 8,
    "bookshelf": 9,
    "picture": 10,
    "counter": 11,
    "desk": 12,
    "curtain": 13,
    "refrigerator": 14,
    "toilet": 16,
    "sink": 17,
    "bathtub": 18,
}

NYU40_ID_TO_NAME = {
    0: "unannotated",
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refrigerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "night stand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

OPENGAUSSIAN_NYU40_CLASS_SPLITS = {
    "19": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36],
    "15": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34],
    "10": [1, 2, 4, 5, 6, 7, 8, 9, 10, 33],
}

OPENGAUSSIAN_NYU40_CLASS_NAMES = {
    split: [NYU40_ID_TO_NAME[class_id] for class_id in class_ids]
    for split, class_ids in OPENGAUSSIAN_NYU40_CLASS_SPLITS.items()
}


def remap_nyu40_labels(
    labels: np.ndarray,
    split_ids: list[int] | tuple[int, ...],
    ignore_index: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Map raw NYU40 ids to contiguous class indices for one eval split.

    Args:
        labels: Arbitrary-shape array of raw NYU40 ids.
        split_ids: Raw NYU40 ids that participate in the split.
        ignore_index: Value assigned to labels outside the split, including 0.

    Returns:
        ``(remapped, valid_mask)`` where valid_mask is true for target classes.
    """
    labels = np.asarray(labels)
    remapped = np.full(labels.shape, ignore_index, dtype=np.int64)
    valid = np.zeros(labels.shape, dtype=bool)
    for out_id, raw_id in enumerate(split_ids):
        mask = labels == int(raw_id)
        remapped[mask] = out_id
        valid |= mask
    return remapped, valid


def compute_split_metrics(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    split_ids: list[int] | tuple[int, ...],
) -> dict:
    """Compute OpenGaussian-style mIoU/mAcc on raw NYU40 labels.

    Predictions are expected as raw NYU40 ids.  Ground-truth labels outside
    *split_ids* are ignored.  Matching OpenGaussian's script, mean IoU and
    mean class accuracy are averaged only over target classes that appear in
    the scene ground truth.
    """
    pred_labels = np.asarray(pred_labels).reshape(-1)
    gt_labels = np.asarray(gt_labels).reshape(-1)
    if pred_labels.shape != gt_labels.shape:
        raise ValueError(
            f"Prediction/GT shape mismatch: {pred_labels.shape} vs {gt_labels.shape}"
        )

    _, valid = remap_nyu40_labels(gt_labels, split_ids)
    pred_valid = pred_labels[valid]
    gt_valid = gt_labels[valid]

    per_class: dict[str, dict[str, float | int | None]] = {}
    ious: list[float] = []
    accs: list[float] = []
    for raw_id in split_ids:
        raw_id = int(raw_id)
        pred_c = pred_valid == raw_id
        gt_c = gt_valid == raw_id
        intersection = int(np.logical_and(pred_c, gt_c).sum())
        union = int(np.logical_or(pred_c, gt_c).sum())
        gt_count = int(gt_c.sum())
        pred_count = int(pred_c.sum())

        iou = float(intersection / union) if gt_count > 0 and union > 0 else None
        acc = float(intersection / gt_count) if gt_count > 0 else None
        if iou is not None:
            ious.append(iou)
        if acc is not None:
            accs.append(acc)
        per_class[str(raw_id)] = {
            "name": NYU40_ID_TO_NAME.get(raw_id, f"class_{raw_id}"),
            "iou": iou,
            "acc": acc,
            "intersection": intersection,
            "union": union,
            "gt_count": gt_count,
            "pred_count": pred_count,
        }

    return {
        "miou": float(np.mean(ious)) if ious else 0.0,
        "macc": float(np.mean(accs)) if accs else 0.0,
        "num_valid": int(valid.sum()),
        "class_ids": [int(class_id) for class_id in split_ids],
        "class_names": [NYU40_ID_TO_NAME.get(int(class_id), f"class_{class_id}") for class_id in split_ids],
        "per_class": per_class,
    }


SEG_COLORS = {}
_rng = np.random.RandomState(7)
for _cid in SCANNET20_CLASSES:
    SEG_COLORS[_cid] = tuple(_rng.randint(60, 255, 3).tolist())
