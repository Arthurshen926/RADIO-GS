"""Paired same-weight/same-image test of the SAM3 complex-buffer corruption."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from radio_gs.scripts.build_sam3_foundation_cache import _load_sam3_model, set_requested_cuda_device, sam3_autocast_context
from radio_gs.utils.immutable_artifacts import sha256_file

p = argparse.ArgumentParser()
p.add_argument('--image', type=Path, required=True)
p.add_argument('--device', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
assert not a.output.exists()
a.output.mkdir(parents=True)
set_requested_cuda_device(a.device)
checkpoint = Path('checkpoints/sam3_modelscope/sam3.pt')
processor = _load_sam3_model(checkpoint_path=str(checkpoint), device=a.device,
    confidence_threshold=0, dtype='bfloat16', resolution=1008, point_only=True)
model = processor.model
buffers = []
for name, module in model.named_modules():
    for key, tensor in module.named_buffers(recurse=False):
        if tensor.is_complex():
            buffers.append((name, module, key, tensor.detach().clone()))
assert buffers and all(torch.isfinite(t).all() for _, _, _, t in buffers)
image = Image.open(a.image).convert('RGB')
points = np.array([[x*image.width, y*image.height] for y in [.2,.4,.6,.8] for x in [.2,.4,.6,.8]], dtype=np.float32)
results = {}
with torch.inference_mode(), sam3_autocast_context(a.device, torch.bfloat16):
    for mode in ['fixed', 'legacy', 'fixed_repeat']:
        for _, module, key, tensor in buffers:
            # Exactly the old .to(bfloat16) effect on registered complex buffers;
            # all real weights, image pixels and prompts remain identical.
            module._buffers[key] = tensor.real.to(torch.bfloat16) if mode=='legacy' else tensor.clone()
        state = processor.set_image(image)
        masks, qualities = [], []
        for point in points:
            mask, quality, _ = model.predict_inst(state, point_coords=point[None], point_labels=np.ones(1,dtype=np.int32), multimask_output=True)
            masks.append(np.asarray(mask).astype(bool)); qualities.append(np.asarray(quality))
        results[mode] = dict(masks=torch.from_numpy(np.stack(masks)), quality=torch.from_numpy(np.stack(qualities)))
        torch.save(results[mode], a.output/(mode+'.pt'))
        print(mode, flush=True)
assert torch.equal(results['fixed']['masks'], results['fixed_repeat']['masks'])
f, old = results['fixed']['masks'], results['legacy']['masks']
intersection=(f&old).flatten(2).sum(-1); union=(f|old).flatten(2).sum(-1)
report=dict(schema='sam3_complex_buffer_control.v1',source_only=True,benchmark_data_used=False,
    image_sha256=sha256_file(a.image),checkpoint_sha256=sha256_file(checkpoint),
    complex_buffers=[dict(name=name+'.'+key,dtype=str(t.dtype),imaginary_nonzero=int((t.imag!=0).sum())) for name,_,key,t in buffers],
    prompt_count=len(points),fixed_repeat_masks_exact=True,
    same_prompt_candidate_iou_mean=float((intersection/union.clamp_min(1)).float().mean()),
    changed_mask_pixel_fraction=float((f!=old).float().mean()),
    quality_mean_abs_difference=float((results['fixed']['quality']-results['legacy']['quality']).abs().mean()),
    semantic_quality_ground_truth_available=False,
    outputs={str(p):sha256_file(p) for p in a.output.glob('*.pt')},
    implementation_sources={f:sha256_file(f) for f in [__file__,'radio_gs/scripts/build_sam3_foundation_cache.py']})
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k not in ('complex_buffers','outputs','implementation_sources')})
