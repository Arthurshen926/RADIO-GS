"""Independent installed-renderer oracle for cancellation-prone thin conics."""
import shutil
import pytest
import torch
from radio_gs.rendering.contribution_compositor import gaussian_footprint_alphas


@pytest.mark.skipif(not torch.cuda.is_available() or shutil.which('nvcc') is None,reason='CUDA/NVCC required')
def test_thin_conic_matches_installed_gsplat_on_nondefault_stream():
    from gsplat.cuda._wrapper import rasterize_to_pixels
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        means=torch.tensor([[[500.8,500.6]]],device='cuda')
        conic=torch.tensor([[[1.,-1.,1.000001]]],device='cuda')
        opacity=torch.tensor([[.9]],device='cuda')
        colors=torch.ones(1,1,1,device='cuda')
        _,expected=rasterize_to_pixels(means,conic,colors,opacity,1,1,16,
            torch.zeros(1,1,1,device='cuda',dtype=torch.int32),torch.zeros(1,device='cuda',dtype=torch.int32))
        ids=torch.zeros(1,device='cuda',dtype=torch.int64)
        exact=gaussian_footprint_alphas(ids,ids,means,conic,opacity,width=1,backend='gsplat_cuda')
        eager=gaussian_footprint_alphas(ids,ids,means,conic,opacity,width=1)
    stream.synchronize()
    assert 0<float(expected)<.999
    torch.testing.assert_close(exact,expected.flatten(),atol=2e-7,rtol=0)
    # This fixture must exercise the cancellation that escaped ordinary cases.
    assert float((eager-expected.flatten()).abs().max())>1e-4
