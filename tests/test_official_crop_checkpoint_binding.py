from types import SimpleNamespace
import torch
from torch import nn
from radio_gs.interfaces.frozen_radio_views import OfficialCropSummaryRuntime


def test_crop_backbone_and_head_share_checkpoint_and_select_declared_teacher(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"test checkpoint identity")
    seen = {}

    def hub_load(*args, **kwargs):
        seen["backbone"] = kwargs["version"]
        assert kwargs["return_checkpoint"]
        return nn.Linear(1, 1), {"args": SimpleNamespace(cls_token_per_teacher=True, teachers=[
            {"name": "dino", "use_summary": True},
            {"name": "siglip2-g", "use_summary": True, "token_slot": 1},
        ])}

    def head_load(path):
        seen["head"] = str(path)
        return nn.Linear(1, 1)

    monkeypatch.setattr(torch.hub, "load", hub_load)
    monkeypatch.setattr("radio_gs.interfaces.frozen_radio_views.SigLIP2SummaryHead.from_radio_checkpoint", head_load)
    runtime = OfficialCropSummaryRuntime.load(checkpoint_path=checkpoint, device="cpu")
    assert seen["backbone"] == seen["head"] == str(checkpoint.resolve())
    assert runtime.summary_slot_index == 1
