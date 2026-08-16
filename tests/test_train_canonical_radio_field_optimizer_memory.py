from __future__ import annotations

import torch

import radio_gs.scripts.train_canonical_radio_field as trainer


def test_chunked_offloaded_adamw_matches_single_tensor_update() -> None:
    initial = torch.linspace(-0.8, 0.9, 15, dtype=torch.float64).reshape(5, 3)
    reference_parameter = torch.nn.Parameter(initial.clone())
    chunked_parameter = torch.nn.Parameter(initial.clone())
    kwargs = {
        "lr": 0.002,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 1e-5,
        "foreach": False,
    }
    reference = torch.optim.AdamW([reference_parameter], **kwargs)
    chunked = torch.optim.AdamW([chunked_parameter], **kwargs)

    for scale in (0.3, -0.2, 0.7):
        gradient = torch.linspace(
            -1.0, 1.0, 15, dtype=torch.float64
        ).reshape(5, 3)
        gradient = gradient * scale
        reference_parameter.grad = gradient.clone()
        chunked_parameter.grad = gradient.clone()
        reference.step()
        trainer._offloaded_adamw_step(chunked, chunk_elements=4)

    assert torch.equal(chunked_parameter, reference_parameter)
    for name in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(
            chunked.state[chunked_parameter][name],
            reference.state[reference_parameter][name],
        )
        assert chunked.state[chunked_parameter][name].device.type == "cpu"


def test_offloaded_optimizer_reuses_gradient_buffer() -> None:
    parameter = torch.nn.Parameter(torch.ones(8))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.arange(8, dtype=torch.float32)
    pointer = parameter.grad.data_ptr()

    trainer._zero_optimizer_gradients(optimizer, preserve_buffers=True)

    assert parameter.grad is not None
    assert parameter.grad.data_ptr() == pointer
    assert torch.count_nonzero(parameter.grad) == 0

    trainer._zero_optimizer_gradients(optimizer, preserve_buffers=False)
    assert parameter.grad is None


def test_chunked_offloaded_adamw_rejects_invalid_chunk_size() -> None:
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)

    try:
        trainer._offloaded_adamw_step(optimizer, chunk_elements=0)
    except ValueError as error:
        assert str(error) == "optimizer-state chunk size must be positive"
    else:
        raise AssertionError("zero-sized optimizer chunk was accepted")
