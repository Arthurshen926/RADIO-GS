from types import MethodType, SimpleNamespace

import pytest
import torch

from radio_gs.scripts.train_feature_field import RadioGSTrainer


class _CapturingGroundingLoss:
    def __init__(self) -> None:
        self.seen_features = None

    def __call__(self, projected_features, text_embeddings, semantic_labels, query_class_ids):
        self.seen_features = projected_features
        return {
            "loss": projected_features.sum() * 0.0,
            "accuracy": torch.tensor(1.0),
            "valid_ratio": torch.tensor(1.0),
        }


def _make_trainer_for_grounding_query() -> RadioGSTrainer:
    trainer = object.__new__(RadioGSTrainer)
    trainer.device = torch.device("cpu")
    trainer.cfg = SimpleNamespace(grounding_query_loss_downsample=1)
    trainer.grounding_query_loss_weight = 1.0
    trainer.grounding_text_embeddings = torch.randn(2, 1536)
    trainer.grounding_query_class_ids = [1, 2]
    trainer.siglip_summary_head = object()
    trainer.grounding_query_loss_fn = _CapturingGroundingLoss()
    return trainer


def test_grounding_query_loss_uses_summary_head_projection() -> None:
    trainer = _make_trainer_for_grounding_query()
    decoded = torch.randn(1, 1280, 2, 3)
    projected = torch.randn(1, 1536, 2, 3)

    def project_summary(self, features):
        assert features is decoded
        return projected

    def project_siglip(self, features):
        raise AssertionError("grounding query loss must use SigLIP2SummaryHead")

    trainer._project_summary_head_features = MethodType(project_summary, trainer)
    trainer._project_siglip_features = MethodType(project_siglip, trainer)

    stats = trainer._compute_grounding_query_loss(
        {"semantics": torch.ones(1, 2, 3, dtype=torch.long)},
        decoded,
    )

    assert stats["loss"].shape == ()
    assert trainer.grounding_query_loss_fn.seen_features is projected


def test_grounding_query_loss_requires_summary_head() -> None:
    trainer = _make_trainer_for_grounding_query()
    trainer.siglip_summary_head = None

    with pytest.raises(RuntimeError, match="SigLIP2SummaryHead"):
        trainer._compute_grounding_query_loss(
            {"semantics": torch.ones(1, 2, 3, dtype=torch.long)},
            torch.randn(1, 1280, 2, 3),
        )
