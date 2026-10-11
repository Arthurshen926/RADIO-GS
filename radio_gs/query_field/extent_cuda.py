"""Packed optical arithmetic matching gsplat's float32 forward operations."""
from functools import lru_cache


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load_inline
    return load_inline(name='radio_extent_optics_v1', cpp_sources='''
        torch::Tensor footprint(torch::Tensor, torch::Tensor, torch::Tensor);
        torch::Tensor optical(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
        ''', cuda_sources=r'''
        #include <torch/extension.h>
        #include <ATen/cuda/CUDAContext.h>
        #include <c10/cuda/CUDAGuard.h>
        __global__ void footprint_kernel(const float* m, const float* c, const float* xy,
                                         float* result, int64_t G, int64_t size) {
            int64_t k = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if (k >= size) return;
            int64_t g=k%G, r=k/G;
            float dx=m[2*g]-xy[2*r], dy=m[2*g+1]-xy[2*r+1];
            float sigma=0.5f*(c[3*g]*dx*dx+c[3*g+2]*dy*dy)+c[3*g+1]*dx*dy;
            result[k] = sigma < 0.f ? 0.f : __expf(-sigma);
        }
        torch::Tensor footprint(torch::Tensor means, torch::Tensor conics, torch::Tensor xy) {
            c10::cuda::CUDAGuard guard(means.device());
            auto result=torch::empty({xy.size(0),means.size(0)},means.options());
            int64_t size=result.numel();
            if(size) footprint_kernel<<<(size+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
                means.data_ptr<float>(),conics.data_ptr<float>(),xy.data_ptr<float>(),result.data_ptr<float>(),means.size(0),size);
            return result;
        }
        __global__ void optical_kernel(const int64_t* ptr,const int64_t* idx,const float* visibility,
            const float* opacity,const float* member,float* result,int64_t rows) {
            int64_t r=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
            if(r>=rows)return;
            float T=1.f;
            for(int64_t k=ptr[r];k<ptr[r+1];++k) {
                int64_t g=idx[k];
                float a=fminf(.999f,(opacity[g]*member[g])*visibility[k]);
                if(a<1.f/255.f)continue;
                float next=T*(1.f-a);
                if(next<=1e-4)break;
                T=next;
            }
            result[r]=1.f-T;
        }
        torch::Tensor optical(torch::Tensor ptr,torch::Tensor idx,torch::Tensor vis,torch::Tensor op,torch::Tensor member) {
            c10::cuda::CUDAGuard guard(member.device());
            auto result=torch::empty({ptr.size(0)-1},member.options());
            int64_t rows=result.numel();
            if(rows) optical_kernel<<<(rows+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
                ptr.data_ptr<int64_t>(),idx.data_ptr<int64_t>(),vis.data_ptr<float>(),op.data_ptr<float>(),
                member.data_ptr<float>(),result.data_ptr<float>(),rows);
            return result;
        }
        ''', functions=['footprint', 'optical'], extra_cuda_cflags=['-O3'], verbose=False)
