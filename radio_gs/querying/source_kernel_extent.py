"""Continuous source-text extent evidence over a frozen native semantic field."""
import torch
from torch.nn import functional as F
from radio_gs.models.native_scope_dictionary import NativeScopeDictionary, attach_scope_text_readout


def attach_kernel_native(model, payload):
    if payload.get('kernel_baseline') == 'pipeline':
        from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
        attach_scope_hypergraph(model, payload)
    else:
        dictionary = NativeScopeDictionary(payload)
        dictionary.modalities = ['text']
        attach_scope_text_readout(model, dictionary)
    calibration = payload.get('kernel_native_calibration')
    if calibration is not None:
        original = model.native_query_thresholds['text'].forward
        weight = calibration['weight'].to(model.latent.device)
        bias = calibration['bias'].to(model.latent.device)
        def threshold(q):
            delta = F.normalize(q, dim=-1) @ weight + bias
            return original(q) + delta.clamp(-calibration['max_delta'], calibration['max_delta'])
        model.native_query_thresholds['text'].forward = threshold
    category = payload.get('kernel_category')
    if category is not None:
        from radio_gs.querying.bounded_native_scope import refine_native_scope_logits
        from radio_gs.querying.category_competition import correct_category_logits
        original_logits = model.logits
        def category_logits(query, modality, rows=None):
            if modality!='text':return original_logits(query,modality,rows)
            native=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(query)-model.native_query_thresholds['text'](query))
            value=refine_native_scope_logits(native,original_logits(query,modality),max_correction=1.) if category['use_scope'] else native
            if category['competition'] is not None:value=correct_category_logits(value,query,category['competition'])
            return value if rows is None else value[rows]
        model.logits=category_logits
    return model


def native_probability(model, query):
    cosine = model.native_cosine(query, modality='text')
    scale = model.readout_log_scale['text'].clamp(-4, 6).exp()
    return torch.sigmoid(scale * (cosine-model.native_query_thresholds['text'](query)))


def attach_source_kernel_extent(model, payload):
    attach_kernel_native(model, payload)
    state = payload['source_kernel_extent']
    device = model.latent.device
    anchors = F.normalize(state['anchors'].float().to(device), dim=-1)
    inverse = state['inverse'].float().to(device)
    residual = state.get('residual')
    if residual is not None:residual=residual.to(device)
    if 'scope_coefficients' in state:
        graph=payload['scope_hypergraph']
        membership=torch.sparse_coo_tensor(graph['indices'].to(device),graph['weights'].to(device),graph['shape']).coalesce()
        spatial_coefficients=state['scope_coefficients'].float().to(device)
    original = model.logits
    def logits(query, modality, rows=None):
        if modality != 'text':
            return original(query, modality, rows)
        q = F.normalize(query.float(), dim=-1)
        kernel = torch.exp(state['concentration'] * (anchors @ q.T - 1))
        coefficients = inverse @ kernel
        if payload.get('kernel_baseline')=='pipeline':
            base_logits=original(q,'text')
        else:
            base_logits=model.readout_log_scale['text'].clamp(-4,6).exp()*(model.native_cosine(q)-model.native_query_thresholds['text'](q))
        native=base_logits.sigmoid()
        if state.get('interpolation')=='conditional_evidence':
            from radio_gs.querying.partial_extent_supervision import source_extent_delta
            delta=source_extent_delta(state,kernel,device)
        else:
            delta = (torch.sparse.mm(membership,spatial_coefficients@coefficients) if 'scope_coefficients' in state else torch.cat([part.float() @ coefficients for part in residual.split(8192)]))
        value = torch.logit((native+delta).clamp(1e-6, 1-1e-6))
        if state.get('readout','probability')=='odds':
            # Preserve relative categorical evidence outside the probability
            # clipping interval; a zero correction leaves every logit unchanged.
            value=base_logits+(value-torch.logit(native.clamp(1e-6,1-1e-6)))
        if 'kernel_category_competition' in payload:
            from radio_gs.querying.category_competition import correct_category_logits
            value=correct_category_logits(value,q,payload['kernel_category_competition'])
        return value if rows is None else value[rows]
    model.logits = logits
    return model
