"""Compose paper-style LERF qualitative comparison grids.

The script uses frozen RADIO-GS overlay assets. It intentionally does not
fabricate external baseline outputs; those can be added later as extra columns
once protocol-matched visualizations exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


GAP = 18
HEADER_HEIGHT = 96
ROW_LABEL_HEIGHT = 46


@dataclass(frozen=True)
class SceneRow:
    scene: str
    query: str
    teacher_path: Path
    rendered_path: Path


@dataclass(frozen=True)
class QueryRow:
    scene: str
    query: str
    frame_id: int
    teacher_path: Path
    rendered_path: Path


def _load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _fit_cell(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - resized_height) // 2
    x = (width - resized_width) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.72,
    color: tuple[int, int, int] = (30, 30, 30),
    thickness: int = 2,
) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _header(width: int, columns: list[str]) -> np.ndarray:
    canvas = np.full((HEADER_HEIGHT, width, 3), 245, dtype=np.uint8)
    _put_text(canvas, "Qualitative comparison on LERF-OVS (frozen overlay assets)", (GAP, 34), 0.78)
    for idx, label in enumerate(columns):
        x = GAP + idx * (columns_width(columns, width) + GAP)
        _put_text(canvas, label, (x + 12, 78), 0.72, color=(20, 70, 130))
    return canvas


def columns_width(columns: list[str], total_width: int) -> int:
    return (total_width - GAP * (len(columns) + 1)) // len(columns)


def _row_label(width: int, scene: str, query: str) -> np.ndarray:
    canvas = np.full((ROW_LABEL_HEIGHT, width, 3), 250, dtype=np.uint8)
    _put_text(canvas, f"{scene}  |  representative queries: {query}", (GAP, 31), 0.68)
    return canvas


def _crop_query_column(image: np.ndarray, column: int, *, total_columns: int = 6) -> np.ndarray:
    if not (0 <= column < total_columns):
        raise ValueError(f"column must be in [0, {total_columns}), got {column}")
    # Per-query LERF visualisations have a small text header followed by one
    # row with columns: query, GT mask, heatmap, RGB, GT/RGB, heatmap/RGB.
    header_height = 28 if image.shape[0] > 28 else 0
    column_width = image.shape[1] // total_columns
    x0 = column * column_width
    x1 = (column + 1) * column_width if column + 1 < total_columns else image.shape[1]
    return image[header_height:, x0:x1]


def _label_cell(scene: str, query: str, frame_id: int, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)
    _put_text(canvas, scene, (10, 32), 0.68, thickness=2)
    _put_text(canvas, f'"{query}"', (10, 67), 0.52, color=(35, 35, 35), thickness=1)
    _put_text(canvas, f"frame {frame_id:05d}", (10, height - 18), 0.45, color=(95, 95, 95), thickness=1)
    return canvas


def _compact_header(width: int, label_width: int, cell_width: int) -> np.ndarray:
    columns = ["Query", "RGB", "GT", "Teacher RADIO", "RADIO-GS (Ours)"]
    column_widths = [label_width, cell_width, cell_width, cell_width, cell_width]
    canvas = np.full((HEADER_HEIGHT, width, 3), 245, dtype=np.uint8)
    _put_text(canvas, "Qualitative comparison on LERF-OVS", (GAP, 34), 0.78)
    x = GAP
    for label, col_width in zip(columns, column_widths):
        _put_text(canvas, label, (x + 8, 78), 0.62, color=(20, 70, 130))
        x += col_width + GAP
    return canvas


def make_compact_qualitative_grid(
    rows: list[QueryRow],
    output_path: Path,
    label_width: int = 270,
    cell_width: int = 300,
    cell_height: int = 220,
) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")

    total_width = label_width + 4 * cell_width + GAP * 6
    parts = [_compact_header(total_width, label_width, cell_width)]

    for row in rows:
        teacher = _load_image(row.teacher_path)
        rendered = _load_image(row.rendered_path)
        panels = [
            _label_cell(row.scene, row.query, row.frame_id, label_width, cell_height),
            _fit_cell(_crop_query_column(teacher, 3), cell_width, cell_height),
            _fit_cell(_crop_query_column(teacher, 4), cell_width, cell_height),
            _fit_cell(_crop_query_column(teacher, 5), cell_width, cell_height),
            _fit_cell(_crop_query_column(rendered, 5), cell_width, cell_height),
        ]
        row_canvas = np.full((cell_height, total_width, 3), 255, dtype=np.uint8)
        x = GAP
        for panel in panels:
            row_canvas[:, x : x + panel.shape[1]] = panel
            x += panel.shape[1] + GAP
        parts.append(row_canvas)
        parts.append(np.full((GAP, total_width, 3), 255, dtype=np.uint8))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure = np.concatenate(parts, axis=0)
    cv2.imwrite(str(output_path), figure)
    return output_path


def make_qualitative_grid(
    rows: list[SceneRow],
    output_path: Path,
    cell_width: int = 1040,
    cell_height: int = 720,
) -> Path:
    if not rows:
        raise ValueError("rows must not be empty")
    columns = ["Teacher RADIO", "RADIO-GS (Ours)"]
    total_width = len(columns) * cell_width + GAP * (len(columns) + 1)
    parts = [_header(total_width, columns)]

    for row in rows:
        parts.append(_row_label(total_width, row.scene, row.query))
        teacher = _fit_cell(_load_image(row.teacher_path), cell_width, cell_height)
        rendered = _fit_cell(_load_image(row.rendered_path), cell_width, cell_height)
        row_canvas = np.full((cell_height, total_width, 3), 255, dtype=np.uint8)
        row_canvas[:, GAP : GAP + cell_width] = teacher
        x = GAP * 2 + cell_width
        row_canvas[:, x : x + cell_width] = rendered
        parts.append(row_canvas)
        parts.append(np.full((GAP, total_width, 3), 255, dtype=np.uint8))

    figure = np.concatenate(parts, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), figure)
    return output_path


def default_rows() -> list[SceneRow]:
    return [
        SceneRow(
            scene="Figurines",
            query="green apple, pikachu, rubics cube",
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/"
                "visualisations/figurines/lerf_grounding_frame_00152_teacher.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/"
                "visualisations/figurines/lerf_grounding_frame_00152_rendered.png"
            ),
        ),
        SceneRow(
            scene="Ramen",
            query="bowl, chopsticks, egg, wavy noodles",
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/"
                "visualisations/ramen/lerf_grounding_frame_00024_teacher.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/"
                "visualisations/ramen/lerf_grounding_frame_00024_rendered.png"
            ),
        ),
        SceneRow(
            scene="Teatime",
            query="apple, coffee mug, sheep, tea",
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/"
                "visualisations/teatime/lerf_grounding_frame_00140_teacher.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/"
                "visualisations/teatime/lerf_grounding_frame_00140_rendered.png"
            ),
        ),
        SceneRow(
            scene="Waldo Kitchen",
            query="knife, sink, refrigerator",
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/"
                "visualisations/waldo_kitchen/lerf_grounding_frame_00089_teacher.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/"
                "visualisations/waldo_kitchen/lerf_grounding_frame_00089_rendered.png"
            ),
        ),
    ]


def default_query_rows() -> list[QueryRow]:
    return [
        QueryRow(
            scene="Figurines",
            query="green apple",
            frame_id=152,
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/"
                "visualisations/figurines/lerf_grounding_frame_00152_teacher_green_apple.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_figurines_overlay_20260502/"
                "visualisations/figurines/lerf_grounding_frame_00152_rendered_green_apple.png"
            ),
        ),
        QueryRow(
            scene="Ramen",
            query="wavy noodles",
            frame_id=24,
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/"
                "visualisations/ramen/lerf_grounding_frame_00024_teacher_wavy_noodles.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_ramen_overlay_20260502/"
                "visualisations/ramen/lerf_grounding_frame_00024_rendered_wavy_noodles.png"
            ),
        ),
        QueryRow(
            scene="Teatime",
            query="coffee mug",
            frame_id=140,
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/"
                "visualisations/teatime/lerf_grounding_frame_00140_teacher_coffee_mug.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_teatime_overlay_20260502/"
                "visualisations/teatime/lerf_grounding_frame_00140_rendered_coffee_mug.png"
            ),
        ),
        QueryRow(
            scene="Waldo Kitchen",
            query="knife",
            frame_id=89,
            teacher_path=Path(
                "output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/"
                "visualisations/waldo_kitchen/lerf_grounding_frame_00089_teacher_knife.png"
            ),
            rendered_path=Path(
                "output/radio_gs/freeze_eval/lerf_waldo_overlay_20260502/"
                "visualisations/waldo_kitchen/lerf_grounding_frame_00089_rendered_knife.png"
            ),
        ),
    ]


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Compose frozen LERF qualitative comparison grid")
    parser.add_argument(
        "--output",
        default="output/radio_gs/paper_figures/submission_freeze_lerf_qualitative_comparison.png",
    )
    parser.add_argument("--layout", choices=["compact", "full"], default="compact")
    parser.add_argument("--label_width", type=int, default=270)
    parser.add_argument("--cell_width", type=int, default=300)
    parser.add_argument("--cell_height", type=int, default=220)
    args = parser.parse_args(argv)
    if args.layout == "compact":
        output = make_compact_qualitative_grid(
            default_query_rows(),
            Path(args.output),
            label_width=args.label_width,
            cell_width=args.cell_width,
            cell_height=args.cell_height,
        )
    else:
        output = make_qualitative_grid(
            default_rows(),
            Path(args.output),
            cell_width=args.cell_width,
            cell_height=args.cell_height,
        )
    print(f"Wrote {output}")
    return output


if __name__ == "__main__":
    main()
