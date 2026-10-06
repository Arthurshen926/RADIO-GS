"""Depth-only visibility for a declared registered reconstruction mesh.

This is a calibration reference, not semantic GT or independent sensor truth.
OpenCV pinhole coordinates use pixel centres at (column + .5, row + .5).
"""
import torch


@torch.no_grad()
def rasterize_registered_mesh_depth(camera_vertices, faces, intrinsic, height, width,
                                    *, near=.01, far=100., context=None):
    import nvdiffrast.torch as dr
    vertices = torch.as_tensor(camera_vertices)
    if not vertices.is_cuda or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('CUDA camera-space vertices of shape [N,3] required')
    if not 0 < near < far or min(height, width) <= 0:
        raise ValueError('Invalid raster dimensions or clipping planes')
    vertices = vertices.float()
    if not torch.isfinite(vertices).all():
        raise ValueError('Non-finite mesh vertices')
    k = torch.as_tensor(intrinsic, device=vertices.device, dtype=vertices.dtype)
    triangles = torch.as_tensor(faces, device=vertices.device, dtype=torch.int32).contiguous()
    if triangles.ndim != 2 or triangles.shape[1] != 3 or triangles.numel() == 0:
        raise ValueError('Non-empty triangular mesh required')
    if int(triangles.min()) < 0 or int(triangles.max()) >= len(vertices):
        raise ValueError('Mesh index outside vertex table')
    x, y, z = vertices.unbind(-1)
    clip = torch.stack((2*k[0,0]/width*x + (2*k[0,2]/width-1)*z,
                        -2*k[1,1]/height*y + (1-2*k[1,2]/height)*z,
                        (far+near)/(far-near)*z - 2*far*near/(far-near), z), -1)
    ctx = context or dr.RasterizeCudaContext(device=vertices.device)
    raster, _ = dr.rasterize(ctx, clip[None].contiguous(), triangles,
                             resolution=(height, width), grad_db=False)
    # OpenGL row zero is the bottom of the image; OpenCV row zero is the top.
    raster = raster[0].flip(0)
    depth = 2*far*near / (far+near-raster[...,2]*(far-near))
    return torch.where(raster[...,3] > 0, depth, torch.full_like(depth, torch.inf))
