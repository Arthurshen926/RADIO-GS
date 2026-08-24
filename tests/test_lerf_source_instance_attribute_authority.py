import torch

from radio_gs.scripts.build_lerf_source_instance_attribute_authority import compile_authority


def test_attribute_authority_selects_one_persistent_phrase_per_track() -> None:
    episodes = {
        "episode_query_proposal": torch.tensor([0, 1, 3]),
        "episode_target_proposal": torch.tensor([1, 2, 4]),
        "episode_target_view": torch.tensor([1, 2, 1]),
        "episode_object_id": torch.tensor([7, 7, 8]),
        "negative_proposal_offsets": torch.tensor([0, 1, 2, 2]),
        "negative_proposals": torch.tensor([3, 4]),
    }
    teacher = {"descriptors": torch.tensor([
        [1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9],
    ])}
    bank = {"embeddings": torch.eye(2), "queries": ["red object", "blue object"], "split": "fit"}
    result = compile_authority(episodes, teacher, bank, 2, 0.0, 0.5)
    assert result["schema_version"] == 2
    assert result["selected_text_index"][0] == result["selected_text_index"][1] == 0
    assert result["selected_text_index"][2] == 1
    assert result["metadata"]["track_consistent_description"] is True
