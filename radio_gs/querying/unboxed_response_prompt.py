"""Ablate inherited extent while keeping the same inferred identity witnesses."""
from .partial_response_prompt import partial_response_prompts


def unboxed_response_prompts(response, mask_hw, points=8):
    variants = partial_response_prompts(response, mask_hw, points)
    if variants is None:
        return None
    return dict(boxed_sparse=variants['sparse_core'],
                unboxed_sparse={k:v for k,v in variants['sparse_core'].items() if k!='box'},
                unboxed_signed={k:v for k,v in variants['signed_core'].items() if k!='box'})
