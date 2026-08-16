from pathlib import Path

import numpy as np

from radio_gs.scripts.predict_nvos_method_v1_field_box_sam3 import (
    choose_candidate_by_signed_points,
)

from radio_gs.scripts.run_nvos_method_v1_full8_readout import (
    DATASET_MANIFEST,
    ReadoutScene,
    _validate_prediction_manifest,
    frozen_scene_order,
    render_command,
    score_command,
    signed_prompt_command,
    transient_sam_command,
)


def _scenes() -> tuple[ReadoutScene, ...]:
    return tuple(
        ReadoutScene(
            scene_id=scene_id,
            camera_map=Path(f"/{scene_id}/camera.json"),
            config=Path(f"/{scene_id}/method.yaml"),
            geometry=Path(f"/{scene_id}/geometry.pth"),
            final_field=Path(f"/{scene_id}/final.pth"),
            final_field_sha256=str(index) * 64,
            gate=Path(f"/{scene_id}/gate.json"),
        )
        for index, scene_id in enumerate(frozen_scene_order(), start=1)
    )


def test_commands_preserve_full8_order_and_pre_gt_boundary() -> None:
    output = Path("/tmp/nvos-full8")
    scenes = _scenes()
    order = frozen_scene_order()

    signed = signed_prompt_command(scenes, output)
    transient = transient_sam_command(scenes, output)
    assert [
        signed[index + 1] for index, value in enumerate(signed) if value == "--scene-id"
    ] == list(order)
    assert [
        transient[index + 1]
        for index, value in enumerate(transient)
        if value == "--scene-id"
    ] == list(order)
    assert "--require-render-authority" in signed
    assert "target" not in " ".join(signed).lower()
    assert "score_nvos_method_v1_full8.py" not in " ".join(signed)
    assert "score_nvos_method_v1_full8.py" not in " ".join(transient)


def test_render_is_hash_bound_to_factorized_field() -> None:
    scene = _scenes()[0]
    command = render_command(scene, Path("/tmp/nvos-full8"))

    assert command[command.index("--manifest") + 1] == str(DATASET_MANIFEST)
    assert command[command.index("--canonical-field-checkpoint") + 1] == str(
        scene.final_field
    )
    assert (
        command[command.index("--canonical-field-checkpoint-schema") + 1]
        == "factorized-v2"
    )
    assert (
        command[command.index("--expected-canonical-field-checkpoint-sha256") + 1]
        == scene.final_field_sha256
    )


def test_only_final_command_opens_the_frozen_scorer() -> None:
    command = score_command(Path("/tmp/nvos-full8"))

    assert command[0] == "radio_gs/scripts/score_nvos_method_v1_full8.py"
    assert command[command.index("--prediction-manifest") + 1].endswith(
        "transient_sam/prediction_manifest.json"
    )


def test_signed_prompt_resume_reads_evaluation_boundary_from_safety(
    tmp_path: Path,
) -> None:
    import json

    path = tmp_path / "prediction_manifest.json"
    path.write_text(
        json.dumps(
            {
                "kind": "promptable_nvs_continuous_score_predictions",
                "input": {"selected_scene_ids": ["fern", "flower"]},
                "safety": {
                    "evaluation_performed": False,
                    "evaluation_ground_truth_opened": False,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _validate_prediction_manifest(
        path,
        kind="promptable_nvs_continuous_score_predictions",
        scene_order=("fern", "flower"),
    )


def test_signed_points_override_larger_coarse_overlap_candidate() -> None:
    margin = np.full((20, 20), -0.1, dtype=np.float32)
    margin[2:7, 2:7] = 1.0
    margin[2:7, 12:17] = 0.05
    coarse = margin >= 0.0
    requested_left = np.zeros_like(coarse)
    requested_left[1:8, 1:8] = True
    wrong_large = np.zeros_like(coarse)
    wrong_large[1:9, 10:19] = True

    selected, report = choose_candidate_by_signed_points(
        margin,
        coarse,
        np.stack((wrong_large, requested_left)),
        scores=np.asarray([0.9, 0.2], dtype=np.float32),
    )

    assert report["accepted"] is True
    assert report["selected_index"] == 1
    assert np.array_equal(selected, requested_left)
