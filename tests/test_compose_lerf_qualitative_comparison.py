from pathlib import Path

import cv2
import numpy as np

from radio_gs.scripts import compose_lerf_qualitative_comparison as compose


def _write_image(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (80, 60)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_query_grid(path: Path, colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros((28, 6 * 40, 3), dtype=np.uint8)
    cells = [np.full((30, 40, 3), color, dtype=np.uint8) for color in colors]
    cv2.imwrite(str(path), np.concatenate([header, np.concatenate(cells, axis=1)], axis=0))


def test_make_qualitative_grid_writes_two_column_scene_comparison(tmp_path: Path) -> None:
    teacher = tmp_path / "scene_teacher.png"
    rendered = tmp_path / "scene_rendered.png"
    _write_image(teacher, (20, 40, 60))
    _write_image(rendered, (80, 100, 120), size=(60, 80))
    output = tmp_path / "grid.png"

    compose.make_qualitative_grid(
        [
            compose.SceneRow(
                scene="Figurines",
                query="small objects",
                teacher_path=teacher,
                rendered_path=rendered,
            )
        ],
        output,
        cell_width=160,
        cell_height=120,
    )

    result = cv2.imread(str(output))
    assert result is not None
    assert result.shape[1] == 2 * 160 + 3 * compose.GAP
    assert result.shape[0] > 120


def test_make_compact_qualitative_grid_crops_paper_columns(tmp_path: Path) -> None:
    teacher = tmp_path / "scene_teacher_green_apple.png"
    rendered = tmp_path / "scene_rendered_green_apple.png"
    _write_query_grid(
        teacher,
        [
            (0, 0, 0),
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
            (40, 40, 40),
            (50, 50, 50),
        ],
    )
    _write_query_grid(
        rendered,
        [
            (0, 0, 0),
            (10, 10, 10),
            (20, 20, 20),
            (30, 30, 30),
            (40, 40, 40),
            (90, 90, 90),
        ],
    )
    output = tmp_path / "compact.png"

    compose.make_compact_qualitative_grid(
        [
            compose.QueryRow(
                scene="Figurines",
                query="green apple",
                frame_id=152,
                teacher_path=teacher,
                rendered_path=rendered,
            )
        ],
        output,
        label_width=80,
        cell_width=60,
        cell_height=45,
    )

    result = cv2.imread(str(output))
    assert result is not None
    assert result.shape[1] == 80 + 4 * 60 + 6 * compose.GAP
    assert result.shape[0] > 45
