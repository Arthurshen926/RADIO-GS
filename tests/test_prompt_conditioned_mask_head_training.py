import pytest
import torch
from PIL import Image

from radio_gs.models.foundation_cache import load_foundation_cache
from radio_gs.scripts.train_prompt_conditioned_sam3_mask_head import (
    _target_for_loss,
    build_coarse_prompt_from_target,
    categories_for_training_frame,
    load_coarse_prompt_mask,
    resolve_feature_dir_for_scene,
    select_prompt_mask_targets,
)


def test_select_prompt_mask_targets_uses_query_indices_and_best_score():
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": "frame_00041",
            "heads": {
                "sam3": {
                    "mask_logits": torch.stack(
                        [
                            torch.zeros(4, 4),
                            torch.ones(4, 4) * 0.25,
                            torch.ones(4, 4) * 0.75,
                        ]
                    ),
                    "queries": ["cup", "plate"],
                    "scores": torch.tensor([0.1, 0.9, 0.8]),
                    "mask_query_indices": torch.tensor([0, 0, 1]),
                    "mask_query_ranks": torch.tensor([0, 1, 0]),
                }
            },
        }
    )
    text_embeddings = {
        "cup": torch.ones(6),
        "plate": torch.ones(6) * 2.0,
    }

    selected = select_prompt_mask_targets(
        cache.heads["sam3"],
        categories=["cup", "plate"],
        text_embeddings=text_embeddings,
        target_size=(2, 2),
    )

    assert selected.categories == ["cup", "plate"]
    assert selected.prompts.shape == (2, 6)
    assert selected.targets.shape == (2, 2, 2)
    assert torch.allclose(selected.targets[0], torch.full((2, 2), 0.25))
    assert torch.allclose(selected.targets[1], torch.full((2, 2), 0.75))


def test_select_prompt_mask_targets_requires_query_indices():
    cache = load_foundation_cache(
        {
            "version": 1,
            "frame_id": "frame_00041",
            "heads": {
                "sam3": {
                    "mask_logits": torch.ones(1, 4, 4),
                    "queries": ["cup"],
                    "scores": torch.tensor([0.9]),
                }
            },
        }
    )

    with pytest.raises(ValueError, match="mask_query_indices"):
        select_prompt_mask_targets(
            cache.heads["sam3"],
            categories=["cup"],
            text_embeddings={"cup": torch.ones(6)},
            target_size=(2, 2),
        )


def test_build_coarse_prompt_from_target_dilates_masks():
    target = torch.zeros(1, 5, 5)
    target[:, 2, 2] = 1.0

    coarse = build_coarse_prompt_from_target(target, dilate=1)

    assert coarse.shape == target.shape
    assert int(coarse.sum().item()) == 9


def test_target_for_loss_binary_uses_probability_threshold():
    targets = torch.tensor([[[0.1, 0.6], [0.5, 0.9]]])

    binary = _target_for_loss(targets, "binary", threshold=0.5)

    assert torch.equal(binary, torch.tensor([[[0.0, 1.0], [0.0, 1.0]]]))


def test_resolve_feature_dir_for_scene_accepts_root_or_scene_dir(tmp_path):
    root = tmp_path / "features"
    scene_dir = root / "figurines"
    (scene_dir / "backbone").mkdir(parents=True)

    assert resolve_feature_dir_for_scene(root, "figurines") == scene_dir
    assert resolve_feature_dir_for_scene(scene_dir, "figurines") == scene_dir


def test_load_coarse_prompt_mask_uses_direct3d_saved_mask_names(tmp_path):
    mask = Image.new("L", (4, 4), 0)
    mask.putpixel((1, 2), 255)
    mask.save(tmp_path / "frame_00041_green apple.png")

    loaded = load_coarse_prompt_mask(tmp_path, frame_id=41, category="green apple", target_size=(2, 2))

    assert loaded.shape == (2, 2)
    assert loaded.max().item() == 1.0


def test_categories_for_training_frame_uses_scene_queries_for_unlabelled_frames():
    frame_annotations = {41: [{"category": "cup"}, {"category": "plate"}]}

    assert categories_for_training_frame(frame_annotations, 41, ["cup", "plate", "spoon"]) == [
        "cup",
        "plate",
    ]
    assert categories_for_training_frame(frame_annotations, 7, ["cup", "plate", "spoon"]) == [
        "cup",
        "plate",
        "spoon",
    ]
