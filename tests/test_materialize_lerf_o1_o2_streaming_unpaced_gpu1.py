from __future__ import annotations

from copy import deepcopy

import pytest

from radio_gs.scripts import materialize_lerf_o1_o2_streaming_unpaced_gpu1 as gpu1


def test_gpu1_contract_binds_both_numerical_entrypoints() -> None:
    contract = gpu1.method_contract()
    assert contract["streaming_core_implementation"]["sha256"] == (
        "f779d025e0754dec583c4565542995c6133e4217d60bcf725090336c81370058"
    )
    assert contract["unpaced_streaming_entrypoint"] == gpu1.UNPACED_IMPLEMENTATION
    assert contract["projection_pacing_seconds_per_batch"] == 0
    assert contract["device_namespace_affects_method_numerics"] is False


def test_gpu1_execution_distinguishes_host_and_program_ordinals() -> None:
    assert gpu1.expected_execution() == {
        "physical_gpu": 1,
        "cuda_visible_devices": "1",
        "program_device": "cuda:0",
        "projection_batch_candidates": [128, 64],
        "pacing_seconds_per_projection_batch": 0,
        "thermal_poll_seconds": 300,
        "soft_pause_temperature_c": 0,
        "maximum_temperature_c": 88,
    }


def test_authority_translation_changes_only_device_namespace() -> None:
    authority = {"execution": gpu1.expected_execution(), "sentinel": {"x": 7}}
    translated = gpu1._translate_authority_for_core_validation(authority)
    assert translated["sentinel"] == authority["sentinel"]
    assert translated["execution"] == gpu1._core_logical_execution()
    assert translated["execution"]["program_device"] == "cuda:0"
    assert authority["execution"]["physical_gpu"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("physical_gpu", 0), ("cuda_visible_devices", "0"), ("program_device", "cuda:1")],
)
def test_authority_translation_rejects_device_mismatch(field: str, value: object) -> None:
    authority = {"execution": deepcopy(gpu1.expected_execution())}
    authority["execution"][field] = value
    with pytest.raises(ValueError, match="GPU1 O1/O2 execution authority differs"):
        gpu1._translate_authority_for_core_validation(authority)


def test_materialize_rejects_wrong_or_missing_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES=1"):
        gpu1.materialize(object())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="requires CUDA_VISIBLE_DEVICES=1"):
        gpu1.materialize(object())


def test_materialize_delegates_with_exact_visible_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = object()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(gpu1, "_CORE_MATERIALIZE", lambda args: {"args": args})
    assert gpu1.materialize(sentinel) == {"args": sentinel}
