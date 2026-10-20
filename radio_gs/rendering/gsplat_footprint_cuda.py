"""Accepted-hit alphas with the CUDA expression used by installed gsplat.

This optional backend preserves CUDA multiplication order, fused operations and
__expf. Separate eager Torch operations can lose precision for thin conics.
The small compiled helper lives only in a private temporary directory, without model artifacts.
"""
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile
import torch

SOURCE = r'''
#include <cuda_runtime.h>
#include <stdint.h>
__global__ void footprint(const int64_t* gids, const int64_t* pids,
    const float* means, const float* conics, const float* opacities,
    int64_t n, int width, float* output) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t g = gids[i], p = pids[i];
    float dx = means[2*g] - ((float)(p % width) + 0.5f);
    float dy = means[2*g+1] - ((float)(p / width) + 0.5f);
    float sigma = 0.5f * (conics[3*g] * dx * dx + conics[3*g+2] * dy * dy)
                  + conics[3*g+1] * dx * dy;
    output[i] = min(0.999f, opacities[g] * __expf(-sigma));
}
extern "C" int footprint_launch(const int64_t* gids, const int64_t* pids,
    const float* means, const float* conics, const float* opacities,
    int64_t n, int width, float* output, cudaStream_t stream) {
    footprint<<<(n+255)/256,256,0,stream>>>(gids,pids,means,conics,opacities,n,width,output);
    return (int)cudaGetLastError();
}
'''


@lru_cache(maxsize=1)
def _kernel():
    with tempfile.TemporaryDirectory(prefix='radiogs_footprint_cuda_') as name:
        root=Path(name);(root/'kernel.cu').write_text(SOURCE)
        subprocess.run(['nvcc','-O3','--use_fast_math','--shared','-Xcompiler=-fPIC',
            '--cudart=static',str(root/'kernel.cu'),'-o',str(root/'kernel.so')],check=True,capture_output=True)
        library=ctypes.CDLL(str(root/'kernel.so'))
    fn=library.footprint_launch
    fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int64,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
    fn.restype=ctypes.c_int
    return library,fn


def accepted_hit_alphas(gids,pids,means,conics,opacity,width):
    tensors=[gids,pids,means,conics,opacity]
    if not all(t.is_cuda and t.device==gids.device and t.is_contiguous() for t in tensors):
        raise ValueError('contiguous CUDA tensors on one device required')
    if gids.dtype!=torch.int64 or pids.dtype!=torch.int64 or any(t.dtype!=torch.float32 for t in tensors[2:]):
        raise ValueError('int64 hit indices and float32 projected geometry required')
    output=torch.empty(gids.numel(),device=gids.device,dtype=torch.float32)
    if not gids.numel():return output
    with torch.cuda.device(gids.device):
        _,fn=_kernel()
        error=fn(*(t.data_ptr() for t in tensors),gids.numel(),width,output.data_ptr(),torch.cuda.current_stream().cuda_stream)
        if error:raise RuntimeError('CUDA footprint launch failed: '+str(error))
    return output
