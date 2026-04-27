"""Generate composite paper figures for grounding comparison and feature flow overview."""

import cv2
import numpy as np
import os

ROOT = "output/radio_gs"
OUT_DIR = f"{ROOT}/paper_figures"
os.makedirs(OUT_DIR, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 3
LABEL_WIDTH = 220  # pixels for left label column
BG = (255, 255, 255)


def add_label_column(img, label, label_w=LABEL_WIDTH):
    """Prepend a white column with a rotated scene label on the left."""
    h, w = img.shape[:2]
    col = np.full((h, label_w, 3), 255, dtype=np.uint8)
    # Measure text and center it
    (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICKNESS)
    x = (label_w - th) // 2 + th  # after rotation x maps to vertical center
    y = (h + tw) // 2
    # Draw rotated text: write on small image then rotate
    txt_img = np.full((label_w, h, 3), 255, dtype=np.uint8)
    tx = (h - tw) // 2
    ty = (label_w + th) // 2
    cv2.putText(txt_img, label, (tx, ty), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)
    col = cv2.rotate(txt_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return np.hstack([col, img])


def resize_width(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / w
    return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Figure 1: Grounding comparison ──────────────────────────────────────────
grounding_scenes = [
    ("Ramen", f"{ROOT}/paper_figures/ramen_grounding_vis/visualisations/ramen/lerf_grounding_frame_00006"),
    ("Figurines", f"{ROOT}/paper_figures/figurines_grounding_vis/visualisations/figurines/lerf_grounding_frame_00105"),
    ("Teatime", f"{ROOT}/paper_figures/teatime_grounding_vis/visualisations/teatime/lerf_grounding_frame_00043"),
]

rows = []
for label, prefix in grounding_scenes:
    gt = cv2.imread(f"{prefix}_gt.png")
    rd = cv2.imread(f"{prefix}_rendered.png")
    assert gt is not None and rd is not None, f"Missing {prefix}"
    # Match heights then concat side-by-side
    if gt.shape[0] != rd.shape[0]:
        h = min(gt.shape[0], rd.shape[0])
        gt = cv2.resize(gt, (int(gt.shape[1] * h / gt.shape[0]), h))
        rd = cv2.resize(rd, (int(rd.shape[1] * h / rd.shape[0]), h))
    pair = np.hstack([gt, rd])
    rows.append((label, pair))

# Normalize all rows to same width
target_w = max(r.shape[1] for _, r in rows)
labelled = []
for label, img in rows:
    img = resize_width(img, target_w)
    img = add_label_column(img, label)
    labelled.append(img)

# Add column headers
header_w = labelled[0].shape[1]
header_h = 60
header = np.full((header_h, header_w, 3), 255, dtype=np.uint8)
content_w = header_w - LABEL_WIDTH
half = content_w // 2
for text, cx in [("Ground Truth", LABEL_WIDTH + half // 2), ("Rendered", LABEL_WIDTH + half + half // 2)]:
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)
    cv2.putText(header, text, (cx - tw // 2, header_h - 10), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

composite1 = np.vstack([header] + labelled)
path1 = f"{OUT_DIR}/fig_grounding_comparison.png"
cv2.imwrite(path1, composite1)
print(f"Saved {path1}  ({composite1.shape[1]}x{composite1.shape[0]})")

# ── Figure 2: Feature flow overview ─────────────────────────────────────────
flow_scenes = [
    ("Figurines", f"{ROOT}/lerf_figurines_v14_fdh_ws240_240ep/feature_flow/composite/grid.png"),
    ("Ramen", f"{ROOT}/lerf_ramen_v14_fdh_ws240_240ep/feature_flow/composite/grid.png"),
    ("Teatime", f"{ROOT}/lerf_teatime_v14_fdh_ws240_240ep/feature_flow/composite/grid.png"),
    ("Waldo Kitchen", f"{ROOT}/lerf_waldo_kitchen_v14_fdh_ws240_240ep/feature_flow/composite/grid.png"),
]

rows2 = []
for label, path in flow_scenes:
    img = cv2.imread(path)
    assert img is not None, f"Missing {path}"
    rows2.append((label, img))

target_w2 = max(r.shape[1] for _, r in rows2)
labelled2 = []
for label, img in rows2:
    img = resize_width(img, target_w2)
    img = add_label_column(img, label)
    labelled2.append(img)

# Thin separator between rows
sep_h = 4
sep = np.full((sep_h, labelled2[0].shape[1], 3), 255, dtype=np.uint8)
parts = []
for i, row in enumerate(labelled2):
    if i > 0:
        parts.append(sep)
    parts.append(row)

composite2 = np.vstack(parts)
path2 = f"{OUT_DIR}/fig_feature_flow_overview.png"
cv2.imwrite(path2, composite2)
print(f"Saved {path2}  ({composite2.shape[1]}x{composite2.shape[0]})")
