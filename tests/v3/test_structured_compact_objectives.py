import torch

from radio_gs.v3.memory.structured_memory import StructuredSharedPrivateMemory
from radio_gs.v3.training.instance_upper_bound import relation_contrastive_loss
from radio_gs.v3.training.run_structured_source_mapping import (
    compact_relation_contrastive_loss,
)


def test_compact_relation_loss_and_gradient_match_full_table_evaluation():
    torch.manual_seed(9)
    initial = torch.randn(11, 512)
    full_model = StructuredSharedPrivateMemory(initial)
    compact_model = StructuredSharedPrivateMemory(initial)
    compact_model.load_state_dict(full_model.state_dict())
    supports = (
        (torch.tensor([0, 2, 4]), torch.tensor([0.2, 0.3, 0.5])),
        (torch.tensor([1, 3]), torch.tensor([0.6, 0.4])),
        (torch.tensor([5, 7, 9]), torch.tensor([0.1, 0.7, 0.2])),
    )
    left = torch.tensor([0, 0])
    right = torch.tensor([1, 2])
    labels = torch.tensor([1, 0], dtype=torch.int8)
    full_loss = relation_contrastive_loss(
        full_model(0.5), supports, left, right, labels, temperature=0.1
    )
    compact_loss = compact_relation_contrastive_loss(
        compact_model, supports, left, right, labels, temperature=0.1
    )
    full_loss.backward()
    compact_loss.backward()
    torch.testing.assert_close(compact_loss, full_loss)
    torch.testing.assert_close(
        compact_model.memory.grad, full_model.memory.grad, atol=1e-6, rtol=1e-5
    )

