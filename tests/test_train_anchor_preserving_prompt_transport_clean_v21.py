from __future__ import annotations

from pathlib import Path

import torch

from radio_gs.scripts import train_anchor_preserving_prompt_transport_clean_v21 as v21


def test_cvar_weights_are_mean_one_and_emphasize_exact_worst_quartile() -> None:
    weights = v21.cvar_prompt_weights(torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(weights[:6], torch.full((6,), 0.5))
    torch.testing.assert_close(weights[6:], torch.full((2,), 2.5))


def _result(deltas: list[float], *, macro_ap: float) -> dict:
    macro = {
        mode: {
            "analytic": {"average_precision": 0.5, "iou_at_0_5": 0.3, "bce": 0.4},
            "candidate": {
                "average_precision": macro_ap,
                "iou_at_0_5": 0.35,
                "bce": 0.3,
            },
        }
        for mode in ("all", "full_mask", "scribble")
    }
    return {
        "macro": macro,
        "records": [{"delta": {"average_precision": value}} for value in deltas],
    }


def test_selection_prioritizes_lower_tail_over_higher_macro() -> None:
    unsafe = _result([-0.02, 0.08, 0.08, 0.08], macro_ap=0.60)
    safe = _result([-0.001, 0.01, 0.01, 0.01], macro_ap=0.53)
    assert v21.risk_sensitive_selection_key(safe, 2) > v21.risk_sensitive_selection_key(
        unsafe, 1
    )


def test_checkpoint_schema_and_risk_lineage_are_distinct(tmp_path: Path) -> None:
    payload = v21.checkpoint_payload(
        state_dict={},
        best_epoch=3,
        result_path=tmp_path / "result.json",
        result_sha256="result",
        lineage={"trainer_sha256": "trainer"},
    )
    assert payload["schema"].endswith("checkpoint.v2_1")
    assert payload["risk_sensitive_training"] == {
        "objective": "0.5_prompt_mean_plus_0.5_worst_quartile_cvar",
        "cvar_fraction": 0.25,
        "uniform_mixture": 0.5,
    }
