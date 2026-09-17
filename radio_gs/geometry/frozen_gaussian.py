"""Frozen surface-shaped Gaussian geometry; no semantic parameters or optimizer."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F


class FrozenGaussianGeometry(nn.Module):
    feature_confidence_contract = "post_visibility_normalized_v1"

    def __init__(self, centres, normals, *, tangent_sigma, normal_sigma, opacity=0.8):
        super().__init__()
        xyz = torch.as_tensor(centres).detach().float()
        n = torch.as_tensor(normals, device=xyz.device).detach().float()
        if xyz.ndim != 2 or xyz.shape[1] != 3 or n.shape != xyz.shape or not xyz.shape[0]:
            raise ValueError('surface geometry must be nonempty [N,3]')
        if not bool(torch.isfinite(xyz).all() & torch.isfinite(n).all()) or bool((n.norm(dim=1)<1e-6).any()):
            raise ValueError('surface geometry/normals invalid')
        if not 0 < normal_sigma <= tangent_sigma or not 0 < opacity < 1:
            raise ValueError('invalid surface thickness/opacity')
        n = F.normalize(n,dim=1)
        # Quaternion rotates the Gaussian's thin local z axis to the normal.
        q = torch.stack([1+n[:,2],-n[:,1],n[:,0],torch.zeros_like(n[:,0])],dim=1)
        antipodal = n[:,2] < -0.999999
        q[antipodal] = torch.tensor([0.,1.,0.,0.],device=xyz.device)
        q = F.normalize(q,dim=1)
        scales = torch.tensor([tangent_sigma,tangent_sigma,normal_sigma],device=xyz.device).expand_as(xyz).clone()
        for key,value in dict(xyz=xyz,normals=n,quaternions=q,scales=scales,
                              opacities=torch.full((xyz.shape[0],1),opacity,device=xyz.device)).items():
            self.register_buffer(key,value)

    def get_xyz(self): return self.xyz
    def get_rotation(self): return self.quaternions
    def get_scaling(self): return self.scales
    def get_opacity(self): return self.opacities

    def fingerprint(self):
        """Bind every quantity affecting visibility, not just centre locations."""
        import hashlib
        tensors = {name: hashlib.sha256(value.detach().float().cpu().contiguous().numpy().astype('<f4').tobytes()).hexdigest()
                   for name,value in self.state_dict().items()}
        return {'schema':'radio_gs.frozen_gaussian_geometry_binding.v1',
                'num_gaussians':int(self.xyz.shape[0]),'tensors':tensors}

    @classmethod
    def load(cls, path, *, expected_sha256, device='cpu'):
        from radio_gs.utils.immutable_artifacts import sha256_file
        if sha256_file(path) != expected_sha256:
            raise ValueError('frozen geometry file identity differs')
        payload = torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('schema') != 'radio_gs.frozen_surface_gaussian_geometry.v1' or payload.get('geometry_trainable_during_semantics') is not False:
            raise ValueError('not a frozen geometry bundle')
        state=payload['state_dict']
        model=cls(state['xyz'],state['normals'],tangent_sigma=1.,normal_sigma=.1)
        model.load_state_dict(state,strict=True)
        if not all(bool(torch.isfinite(v).all()) for v in model.buffers()):
            raise ValueError('nonfinite frozen geometry')
        if bool((model.scales<=0).any()) or bool(((model.opacities<=0)|(model.opacities>=1)).any()):
            raise ValueError('invalid frozen geometry scales/opacity')
        if not torch.allclose(model.quaternions.norm(dim=1),torch.ones(model.xyz.shape[0]),atol=1e-5):
            raise ValueError('invalid frozen geometry rotation')
        return model.to(device)

    def assert_binding(self, expected):
        if self.fingerprint() != expected:
            raise ValueError('geometry changed: rebuild all pixel/feature assignments before semantic use')
