from pathlib import Path
import hashlib

import pytest
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


def test_load_trusted_checkpoint_uses_restricted_sha_bound_path(tmp_path):
    from radio_gs.utils.checkpoint_io import load_trusted_checkpoint

    path = tmp_path / "formal.pth"
    torch.save(
        {"state_dict": {"weight": torch.ones(2)}, "output": Path("run")},
        path,
    )
    expected = hashlib.sha256(path.read_bytes()).hexdigest()

    payload = load_trusted_checkpoint(
        path,
        expected_sha256=expected,
        map_location="cpu",
    )

    assert payload["output"] == Path("run")
    assert torch.equal(payload["state_dict"]["weight"], torch.ones(2))


def test_load_trusted_checkpoint_rejects_wrong_formal_digest(tmp_path):
    from radio_gs.utils.checkpoint_io import load_trusted_checkpoint

    path = tmp_path / "formal.pth"
    torch.save({"state_dict": {}}, path)

    with pytest.raises(ValueError, match="SHA-256 differs"):
        load_trusted_checkpoint(
            path,
            expected_sha256="0" * 64,
            map_location="cpu",
        )
