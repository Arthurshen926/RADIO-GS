"""Region readout after natural-image encoding with the frozen SigLIP head.

Self-attention sees the complete natural crop. Fractional mask coverage limits
only the final learned-probe pooling. This preserves parameters and dimensions,
not a guarantee that the changed readout preserves semantic accuracy.
"""
import torch
from torch.nn import functional as F


class ContextRegionEncoder:
    def __init__(self, native_runtime):
        self.native = native_runtime

    @torch.inference_mode()
    def encode(self, crops, masks):
        model = self.native.model
        dtype = next(model.parameters()).dtype
        pixels = (2*crops.to(self.native.device).float()-1).to(dtype)
        output = model(pixel_values=pixels, return_dict=True)
        tokens = output.last_hidden_state
        patch = model.config.patch_size
        coverage = F.adaptive_avg_pool2d(masks[:, None].float().to(tokens.device),
            (crops.shape[-2]//patch, crops.shape[-1]//patch)).flatten(1)
        # A zero-area token contributes no key/value mass. Fractional boundary
        # tokens enter proportionally, without a size filter or hard threshold.
        bias = torch.where(coverage > 0, coverage.log(), -torch.inf).to(dtype)
        head = model.vision_model.head
        probe = head.probe.expand(len(tokens), -1, -1)
        pooled = head.attention(probe, tokens, tokens, key_padding_mask=bias)[0]
        pooled = pooled + head.mlp(head.layernorm(pooled))
        return dict(natural_global=F.normalize(output.pooler_output.float(), dim=-1),
                    context_region=F.normalize(pooled[:, 0].float(), dim=-1))
