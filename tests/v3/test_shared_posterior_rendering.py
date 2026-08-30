import torch
from torch import nn

from radio_gs.v3.memory.structured_memory import LowRankPrivateBranchMemory
from radio_gs.v3.query.identity_adapter import AffineTextAlignment, DirectTextProjection
from radio_gs.v3.query.interface import StructuredGaussianQueryInterface
from radio_gs.v3.query.packet import QueryPacket
from radio_gs.v3.query.membership import (
    membership_from_prototype,
    pool_prototype,
    relative_membership_from_prototypes,
)
from radio_gs.v3.training.rendered_mask import render_membership, rendered_mask_loss


def test_2d_is_only_a_render_of_the_same_3d_posterior():
    instance = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    prototype = pool_prototype(instance, torch.tensor([1.0, 1.0, 0.0]))
    posterior = membership_from_prototype(instance, prototype, temperature=0.2)
    image = render_membership(
        posterior,
        gaussian_ids=torch.tensor([0, 1, 2]),
        pixel_ids=torch.tensor([0, 0, 1]),
        contribution_weights=torch.tensor([0.6, 0.4, 1.0]),
        num_pixels=2,
    )
    torch.testing.assert_close(image[0], 0.6 * posterior[0] + 0.4 * posterior[1])
    loss = rendered_mask_loss(image, torch.tensor([1.0, 0.0]))
    loss.total.backward if False else None
    assert float(loss.total) > 0


def test_unknown_pixels_are_excluded_from_mask_loss():
    prediction = torch.tensor([0.9, 0.99, 0.1], requires_grad=True)
    loss = rendered_mask_loss(prediction, torch.tensor([1.0, 0.0, 0.0]), known=torch.tensor([1, 0, 1], dtype=torch.bool))
    loss.total.backward()
    assert prediction.grad[1] == 0


def test_positive_membership_margin_reduces_unrelated_coverage():
    instance = torch.tensor([[1.0, 0.0], [0.2, 0.98], [-1.0, 0.0]])
    prototype = torch.tensor([1.0, 0.0])
    raw = membership_from_prototype(instance, prototype, temperature=0.15)
    calibrated = membership_from_prototype(
        instance, prototype, temperature=0.15, margin=0.2
    )
    assert bool((calibrated < raw).all())
    assert float(raw[1]) > 0.5
    assert float(calibrated[1]) < 0.5


def test_relative_membership_uses_hardest_competing_prototype():
    instance = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]])
    posterior = relative_membership_from_prototypes(
        instance,
        torch.tensor([1.0, 0.0]),
        torch.tensor([[0.0, 1.0], [-1.0, 0.0]]),
        temperature=0.15,
    )
    assert float(posterior[0]) > 0.99
    assert float(posterior[1]) < 0.01
    torch.testing.assert_close(posterior[2], torch.tensor(0.5))


def test_affine_text_alignment_is_finite_and_normalized():
    alignment = AffineTextAlignment(torch.eye(128), torch.ones(128) * 0.1)
    output = alignment(torch.randn(3, 128))
    torch.testing.assert_close(output.norm(dim=1), torch.ones(3))


def test_direct_text_projection_uses_raw_token_coordinates():
    basis = torch.zeros(1536, 128)
    basis[:128] = torch.eye(128)
    projection = DirectTextProjection(basis)
    token = torch.zeros(1536)
    token[7] = 2.0
    output = projection.project_raw(token, torch.zeros(1536))
    assert output.shape == (1, 128)
    torch.testing.assert_close(output[0, 7], torch.tensor(1.0))


def test_canonical_interface_renders_the_exact_gaussian_posterior():
    torch.manual_seed(3)
    model = LowRankPrivateBranchMemory(torch.randn(5, 512))
    interface = StructuredGaussianQueryInterface(
        model, torch.zeros(5, 5), nn.Linear(16, 1)
    )
    posterior = interface.gaussian_posterior(
        torch.tensor([0, 1]), torch.tensor([0.7, 0.3]), scale=0.4
    )
    rendered = interface.render_posterior(
        posterior,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([0.6, 0.4, 0.2, 0.8]),
        num_pixels=2,
    )
    torch.testing.assert_close(
        rendered,
        torch.stack((0.6 * posterior[0] + 0.4 * posterior[1], 0.2 * posterior[2] + 0.8 * posterior[3])),
    )


def test_text_and_image_tokens_compile_to_the_same_identity_anchors():
    torch.manual_seed(5)
    model = LowRankPrivateBranchMemory(torch.randn(13, 512))
    interface = StructuredGaussianQueryInterface(
        model,
        torch.zeros(13, 5),
        nn.Linear(16, 1),
        siglip_mean=torch.randn(1536),
        siglip_basis=torch.randn(1536, 128),
    )
    token = torch.randn(1536)
    text = QueryPacket("text", token=token)
    image = QueryPacket("image", token=token.clone())

    text_rows, text_weights, text_identity = interface.compile_identity_anchors(text, topk=4)
    image_rows, image_weights, image_identity = interface.compile_identity_anchors(image, topk=4)
    text_posterior, _ = interface.posterior_from_packet(text, scale=0.3, topk=4)
    image_posterior, _ = interface.posterior_from_packet(image, scale=0.3, topk=4)

    assert torch.equal(text_rows, image_rows)
    torch.testing.assert_close(text_weights, image_weights)
    torch.testing.assert_close(text_identity, image_identity)
    torch.testing.assert_close(text_posterior, image_posterior)


def test_text_identity_uses_hardest_canonical_negative_when_sealed():
    memory = torch.zeros(3, 512)
    memory[0, 320] = 1.0
    memory[1, 321] = 1.0
    memory[2, 320:322] = 1.0
    basis = torch.zeros(1536, 128)
    basis[:128] = torch.eye(128)
    positive = torch.zeros(1536)
    positive[0] = 1.0
    negatives = torch.zeros(1, 1536)
    negatives[0, 1] = 1.0
    interface = StructuredGaussianQueryInterface(
        LowRankPrivateBranchMemory(memory),
        torch.zeros(3, 5),
        nn.Linear(16, 1),
        siglip_mean=torch.zeros(1536),
        siglip_basis=basis,
        text_negative_tokens=negatives,
        text_logit_scale=10.0,
    )
    _rows, _weights, identity = interface.compile_identity_anchors(
        QueryPacket("text", token=positive), topk=2
    )
    assert float(identity[0]) > 0.99
    assert float(identity[1]) < 0.01
    assert abs(float(identity[2]) - 0.5) < 1e-5


def test_positive_text_anchors_keep_null_as_separate_evidence():
    memory = torch.zeros(3, 512)
    memory[0, 320] = 1.0
    memory[1, 321] = 1.0
    memory[2, 320:322] = 1.0
    basis = torch.zeros(1536, 128)
    basis[:128] = torch.eye(128)
    positive = torch.zeros(1536)
    positive[0] = 1.0
    negatives = torch.zeros(1, 1536)
    negatives[0, 1] = 1.0
    interface = StructuredGaussianQueryInterface(
        LowRankPrivateBranchMemory(memory),
        torch.zeros(3, 5),
        nn.Linear(16, 1),
        siglip_mean=torch.zeros(1536),
        siglip_basis=basis,
        text_negative_tokens=negatives,
    )
    packet = QueryPacket("text", token=positive)
    rows, _weights, score = interface.compile_identity_anchors(
        packet, topk=2, text_anchor_policy="positive"
    )
    raw, null, unknown = interface.semantic_text_evidence(packet)

    assert torch.equal(rows, torch.tensor([0, 2]))
    torch.testing.assert_close(score, raw)
    torch.testing.assert_close(null, torch.tensor([0.0, 1.0, 2 ** -0.5]))
    assert not bool(unknown.any())

    replay_score, replay_rows, replay_weights = interface.replay_identity_from_packet(
        packet, topk=2
    )
    assert torch.equal(replay_score, raw)
    assert torch.equal(replay_rows, rows)
    _rows, positive_weights, _score = interface.compile_identity_anchors(
        packet, topk=2, text_anchor_policy="positive"
    )
    assert torch.equal(replay_weights, positive_weights)


def test_prompt_packet_uses_only_finite_gaussian_seed_rows():
    model = LowRankPrivateBranchMemory(torch.randn(6, 512))
    interface = StructuredGaussianQueryInterface(
        model, torch.zeros(6, 5), nn.Linear(16, 1)
    )
    seed = torch.tensor([float("nan"), 0.2, 0.9, 0.4, float("nan"), 0.8])
    rows, weights, _ = interface.compile_identity_anchors(
        QueryPacket("prompt", seed_probability=seed), topk=2
    )

    assert torch.equal(rows, torch.tensor([2, 5]))
    assert float(weights.sum()) == 1.0


def test_disabled_signed_boundary_is_exact_noop():
    model = LowRankPrivateBranchMemory(torch.zeros(3, 512))
    head = nn.Linear(16, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    interface = StructuredGaussianQueryInterface(model, torch.zeros(3, 5), head)
    base = torch.tensor([0.2, 0.5, 0.8])
    refined, magnitude = interface.refine_instance_with_boundary(base)

    assert torch.equal(refined, base)
    assert not bool(magnitude.any())


def test_signed_boundary_uses_instance_membership_for_inside_outside_direction():
    memory = torch.zeros(2, 512)
    memory[:, 496] = 1.0
    model = LowRankPrivateBranchMemory(memory)
    head = nn.Linear(16, 1)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    head.weight.data[0, 0] = 1.0
    interface = StructuredGaussianQueryInterface(model, torch.zeros(2, 5), head)
    base = torch.tensor([0.2, 0.8])
    refined, magnitude = interface.refine_instance_with_boundary(base)

    assert bool((magnitude > 0).all())
    assert refined[0] < base[0]
    assert refined[1] > base[1]
