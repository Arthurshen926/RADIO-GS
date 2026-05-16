from pathlib import Path

import torch


def test_load_trusted_checkpoint_disables_weights_only(monkeypatch, tmp_path):
    from radio_gs.utils.checkpoint_io import load_trusted_checkpoint

    calls = []

    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return {"ok": True}

    monkeypatch.setattr(torch, "load", fake_load)

    path = tmp_path / "trusted.pth"
    payload = load_trusted_checkpoint(path, map_location="cpu")

    assert payload == {"ok": True}
    assert calls == [(Path(path), {"map_location": "cpu", "weights_only": False})]


def test_load_trusted_checkpoint_supports_older_torch_without_weights_only(
    monkeypatch, tmp_path
):
    from radio_gs.utils.checkpoint_io import load_trusted_checkpoint

    calls = []

    def fake_load(path, **kwargs):
        calls.append(kwargs.copy())
        if "weights_only" in kwargs:
            raise TypeError("unexpected keyword argument 'weights_only'")
        return {"legacy": True}

    monkeypatch.setattr(torch, "load", fake_load)

    payload = load_trusted_checkpoint(tmp_path / "legacy.pth", map_location="cpu")

    assert payload == {"legacy": True}
    assert calls == [
        {"map_location": "cpu", "weights_only": False},
        {"map_location": "cpu"},
    ]
