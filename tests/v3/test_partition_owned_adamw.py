import torch

from radio_gs.v3.training.partition_owned_adamw import PartitionOwnedAdamW


def test_partition_optimizer_cannot_decay_or_move_unowned_columns():
    table = torch.nn.Parameter(torch.randn(5, 8))
    before = table.detach().clone()
    optimizer = PartitionOwnedAdamW(
        table, slice(3, 6), lr=1e-2, weight_decay=0.1
    )
    table.grad = torch.ones_like(table)
    optimizer.step()
    assert torch.equal(table[:, :3], before[:, :3])
    assert torch.equal(table[:, 6:], before[:, 6:])
    assert not torch.equal(table[:, 3:6], before[:, 3:6])


def test_chunked_owned_update_matches_torch_adamw_on_active_block():
    torch.manual_seed(3)
    table = torch.nn.Parameter(torch.randn(4, 7))
    reference = torch.nn.Parameter(table.detach()[:, 2:6].clone())
    owned = PartitionOwnedAdamW(
        table, slice(2, 6), lr=3e-3, weight_decay=2e-2, chunk_elements=3
    )
    standard = torch.optim.AdamW([reference], lr=3e-3, weight_decay=2e-2)
    for _ in range(4):
        gradient = torch.randn_like(table)
        table.grad = gradient.clone()
        reference.grad = gradient[:, 2:6].clone()
        owned.step()
        standard.step()
        torch.testing.assert_close(table[:, 2:6], reference, atol=1e-7, rtol=1e-6)
