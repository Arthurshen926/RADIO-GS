from pathlib import Path

import torch

from radio_gs.scripts import build_storage_footprint_report as storage


def test_parse_ply_vertex_count_reads_ascii_header(tmp_path: Path) -> None:
    ply = tmp_path / "point_cloud.ply"
    ply.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 10",
                "property float x",
                "end_header",
            ]
        ),
        encoding="utf-8",
    )

    assert storage.parse_ply_vertex_count(ply) == 10


def test_build_row_reports_direct_and_compact_storage(tmp_path: Path) -> None:
    ply = tmp_path / "point_cloud.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 10\nend_header\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(
        {
            "model_state_dict": {
                "_latent": torch.zeros(10, 32, dtype=torch.float32),
                "hash_field.hash_tables.0": torch.zeros(4, 4, dtype=torch.float32),
            },
            "codec_state_dict": {
                "decoder.weight": torch.zeros(2, 2, dtype=torch.float32),
            },
            "refiner_state_dict": {
                "refiner.weight": torch.zeros(1, 1, dtype=torch.float32),
            },
        },
        checkpoint,
    )

    row = storage.build_storage_row(
        storage.SceneFootprintInput(
            scene="Toy",
            ply_path=ply,
            checkpoint_path=checkpoint,
        )
    )

    assert row.gaussians == 10
    assert row.direct_fp16_bytes == 10 * 1280 * 2
    assert row.model_bytes > 0
    assert row.total_compact_bytes > row.model_bytes
    assert row.saving_ratio > 1.0
