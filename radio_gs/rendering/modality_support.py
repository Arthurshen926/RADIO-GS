"""Normalize scalar query renders using the matching modality's evidence mass."""
import torch


def supported_modality_response(image, alpha, *, query_index, modality_index, num_queries, num_modalities, shared_support):
    """Image layout: all query responses, then one or M support channels."""
    expected=num_queries*num_modalities+(1 if shared_support else num_modalities)
    if image.shape[:-1]!=alpha.shape or image.shape[-1]!=expected or not 0<=query_index<num_queries or not 0<=modality_index<num_modalities:
        raise ValueError('query/support raster axes differ')
    support_index=-1 if shared_support else num_queries*num_modalities+modality_index
    mass=image[...,support_index]
    valid=(mass/alpha.clamp_min(1e-8)>.5)&(alpha>1e-8)
    response=image[...,modality_index*num_queries+query_index]/mass.clamp_min(1e-8)
    return torch.where(valid,response,torch.zeros_like(response))
