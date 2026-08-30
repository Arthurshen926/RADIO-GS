from __future__ import annotations

import torch

from radio_gs.v4.geometry.moge3_inference import OfficialMoge3Runner


class _FakeMoge3:
    def to(self, device):
        self.device = torch.device(device)
        return self

    def eval(self):
        return self

    def infer(self, image, **kwargs):
        height, width = image.shape[-2:]
        points = torch.zeros(height, width, 3, device=image.device)
        points[..., 2] = 2
        normal = torch.zeros_like(points)
        normal[..., 2] = 1
        mask = torch.ones(height, width, dtype=torch.bool, device=image.device)
        points[0, 0, 2] = -1
        return {"points": points, "normal": normal, "mask": mask}


def test_official_moge3_adapter_is_revision_pinned_and_validates_geometry(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"audited-v3-checkpoint")
    model_id = "Ruicheng/moge-3-vitl"
    revision = OfficialMoge3Runner.RELEASES[model_id]
    runner = OfficialMoge3Runner(
        _FakeMoge3(),
        checkpoint,
        model_id=model_id,
        revision=revision,
        device="cpu",
        use_fp16=False,
    )

    prediction = runner.predict(torch.rand(3, 8, 12))

    assert prediction.point_map.shape == (8, 12, 3)
    assert prediction.normals.shape == (8, 12, 3)
    assert not prediction.validity[0, 0]
    assert torch.equal(prediction.confidence, prediction.validity.float())


def test_official_moge3_adapter_rejects_unapproved_revision(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    try:
        OfficialMoge3Runner(
            _FakeMoge3(),
            checkpoint,
            model_id="Ruicheng/moge-3-vitl",
            revision="main",
            device="cpu",
        )
    except ValueError as exc:
        assert "allowlist" in str(exc)
    else:
        raise AssertionError("unapproved MoGe-3 revision was accepted")
