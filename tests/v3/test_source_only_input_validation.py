import pytest
import torch

from radio_gs.v3.training.instance_upper_bound import validate_source_only_inputs


def _membership() -> dict:
    return {
        "metadata": {
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "evaluation_rgb_opened": False,
        }
    }


def _native_authority() -> dict:
    return {
        "schema": "radio_gs.sugm_v3.native_language_authority.v3",
        "edge_relation": torch.tensor([-1, 0, 1], dtype=torch.int8),
        "metadata": {
            "source_only": True,
            "historical_field_opened": False,
            "benchmark_images_opened": False,
            "benchmark_masks_opened": False,
            "benchmark_metrics_opened": False,
            "dev_and_audit_text_scores_used_for_label_selection": False,
        },
    }


def test_native_language_v3_is_accepted_as_source_only_relation() -> None:
    validate_source_only_inputs(_membership(), _native_authority())


def test_native_language_v3_rejects_dev_selected_labels() -> None:
    relation = _native_authority()
    relation["metadata"]["dev_and_audit_text_scores_used_for_label_selection"] = True
    with pytest.raises(ValueError, match="not source-only"):
        validate_source_only_inputs(_membership(), relation)
