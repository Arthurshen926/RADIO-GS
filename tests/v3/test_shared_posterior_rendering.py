import torch

from radio_gs.v3.query.membership import membership_from_prototype, pool_prototype
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
