from argparse import Namespace
from pathlib import Path

import pytest

from reproductions.ludvig import run_ludvig_sam as runner


def _config_args(
    *, retain_full_carrier: bool, materialize_only: bool = False
) -> Namespace:
    return Namespace(
        benchmark="spin",
        sam_checkpoint=Path("/tmp/sam.pth"),
        retain_full_carrier=retain_full_carrier,
        materialize_only=materialize_only,
    )


def test_full_carrier_query_interface_disables_visibility_pruning(
    tmp_path: Path,
) -> None:
    config = runner._config(
        _config_args(retain_full_carrier=True),
        tmp_path,
        {"mask_pattern": "/tmp/masks/{}"},
    )

    assert "prune_gaussians" not in config
    assert config["feature"]["multimask_output"] is True


def test_released_reproduction_keeps_original_visibility_pruning(
    tmp_path: Path,
) -> None:
    config = runner._config(
        _config_args(retain_full_carrier=False),
        tmp_path,
        {"mask_pattern": "/tmp/masks/{}"},
    )

    assert config["prune_gaussians"] == 0.5


def test_materialize_only_keeps_target_evaluator_out_of_gpu_stage(
    tmp_path: Path,
) -> None:
    config = runner._config(
        _config_args(retain_full_carrier=True, materialize_only=True),
        tmp_path,
        {"mask_pattern": "/tmp/masks/{}"},
    )

    assert "evaluation" not in config
    assert "prune_gaussians" not in config


@pytest.mark.parametrize("physical_gpu", [0, 1])
def test_runtime_can_pin_either_physical_gpu(
    tmp_path: Path, physical_gpu: int
) -> None:
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"driver")
    environment, selected_driver = runner._runtime_environment(
        Namespace(
            driver_library_dir=tmp_path,
            physical_gpu=physical_gpu,
            pythonpath=None,
        )
    )

    assert environment["CUDA_VISIBLE_DEVICES"] == str(physical_gpu)
    assert selected_driver == driver
    assert environment["LD_LIBRARY_PATH"].split(":", 1)[0] == str(tmp_path)


def test_runtime_rejects_an_unregistered_gpu(tmp_path: Path) -> None:
    (tmp_path / "libcuda.so.1").write_bytes(b"driver")

    with pytest.raises(runner.ProtocolError, match="physical-gpu"):
        runner._runtime_environment(
            Namespace(driver_library_dir=tmp_path, physical_gpu=2, pythonpath=None)
        )
