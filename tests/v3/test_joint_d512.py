import torch

from radio_gs.v3.training.joint_d512 import JointD512Arm


def make_arm() -> JointD512Arm:
    return JointD512Arm(
        torch.randn(5, 512),
        radio_basis=torch.randn(7, 512),
        radio_mean=torch.randn(7),
        radio_scale=torch.rand(7) + 0.5,
        output_dim=4,
    )


def test_joint_arm_starts_at_canonical_field_and_serializes_one_d512():
    arm = make_arm()
    assert torch.equal(arm.latent, arm.base_latent)
    assert arm.state_dict().keys() == {
        "latent",
        "projection.weight",
        "scale_adapter.weight",
        "scale_adapter.bias",
    }
    assert arm.deployment_latent().shape == (5, 512)


def test_joint_arm_visual_and_instance_losses_reach_same_latent():
    arm = make_arm()
    rows = torch.tensor([0, 2])
    loss = arm.radio_anchor_loss(rows) + arm(0.3, rows).square().mean()
    loss.backward()
    assert arm.latent.grad is not None
    assert torch.isfinite(arm.latent.grad).all()
