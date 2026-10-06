"""Native and legacy category readers with explicit deployment assets.

The current native reader preserves encoder coordinates and fuses declared
source observations. The legacy field/head path remains a frozen comparator.
Neither reader retrieves historical final query predictions.
"""
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from radio_gs.models.element_retrieval_field import ElementRetrievalField
from radio_gs.models.native_query_encoders import load_siglip_text
from radio_gs.querying.scope_hypergraph import attach_scope_hypergraph
from radio_gs.querying.source_accepted_category import source_accepted_category_logits
from radio_gs.querying.category_competition import correct_category_logits
from radio_gs.querying.typed_evidence import visible_operator, view_evidence, solve_set, SetType


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


@torch.inference_mode()
def query_memory(scene, queries, device='cuda'):
    texts = [q['text'] for q in queries]
    encoder, tokenizer = load_siglip_text(scene['text_encoder'], device)
    tokens = tokenizer(texts, padding='max_length', max_length=64, truncation=True,
                       return_tensors='pt').to(device)
    query = F.normalize(F.normalize(encoder(**tokens).pooler_output.float(), dim=-1).half().float(), dim=-1)
    del encoder
    memory = load(scene['memory'])
    base = load(scene['base_field'])
    model = ElementRetrievalField(**base['architecture']).to(device)
    model.load_state_dict(base['state_dict']); model.eval()
    attach_scope_hypergraph(model, memory)
    calibration = load(scene['calibration'])
    original = model.native_query_thresholds['text'].forward
    def threshold(q):
        delta = F.normalize(q, dim=-1) @ calibration['weight'].to(device) + calibration['bias'].to(device)
        return original(q) + delta.clamp(-calibration['max_delta'], calibration['max_delta'])
    model.native_query_thresholds['text'].forward = threshold
    native = model.readout_log_scale['text'].clamp(-4, 6).exp() * (
        model.native_cosine(query) - model.native_query_thresholds['text'](query))
    scope = json.loads(Path(scene['source_scope_receipt']).read_text())
    admission = json.loads(Path(scene['source_admission_receipt']).read_text())
    values, decision = source_accepted_category_logits(native, lambda:model.logits(query, 'text'),
        scope_receipt=scope, admission_receipt=admission, max_correction=1.)
    values = correct_category_logits(values, query, memory['category_competition'])
    return values, query, model.observed, decision


@torch.inference_mode()
def declared_source_views(observed, scene, texts, identity_gates=None):
    """Recompute transient source evidence; no old element score is loaded."""
    spec = scene['source_evidence']; h, w = spec['raster_hw']
    records = [load(r['identity']) for r in spec['views']]
    query_ids = [records[0]['queries'].index(t) for t in texts]
    square = torch.zeros(len(texts)); count = torch.zeros_like(square)
    for data in records:
        for j, qi in enumerate(query_ids):
            row = data['instances'][qi]
            square[j] += row['foreground_margin'].square().sum(); count[j] += len(row['scores'])
    scale = (square / count.clamp_min(1)).sqrt().clamp_min(1e-6)
    examples = []
    detector_confidence = []
    for view, data in zip(spec['views'], records):
        rows = [data['instances'][qi] for qi in query_ids]
        op, evop, visible = visible_operator(load(view['contributions']), observed, h*w, False)
        probability = torch.stack([r['masks'].float().amax(0) if len(r['scores']) else
            torch.zeros(h,w) for r in rows], dim=-1).reshape(h*w,-1).to(observed.device)
        detected = torch.tensor([len(r['scores'])>0 for r in rows], device=observed.device)
        confidence = torch.tensor([float((r['scores']*(r['foreground_margin']/scale[j]).sigmoid()).max())
            if len(r['scores']) else 0. for j,r in enumerate(rows)], device=observed.device)
        example = view_evidence(op,evop,visible,probability,detected,view['frame'],h,w,confidence,boundary_radius=data.get('boundary_radius',1))
        if scene.get('source_identity_assignment') == 'spatial_proposal':
            # A positive pixel inherits identity reliability from the proposal
            # that supplied its mask probability, not an unrelated instance.
            spatial = example.confidence.expand(h*w, -1).clone()
            for j, row in enumerate(rows):
                if len(row['scores']):
                    owner = row['masks'].float().flatten(1).argmax(0)
                    reliability = row['scores'] * (row['foreground_margin']/scale[j]).sigmoid()
                    local = reliability[owner].to(observed.device)
                    spatial[:, j] = torch.where(example.positive[:, j], local, spatial[:, j])
            example.confidence = spatial
        examples.append(example)
        if scene.get('source_identity_control') == 'detector_matched_mass':
            detector_confidence.append(torch.tensor([float(r['scores'].max()) if len(r['scores']) else 0.
                                                     for r in rows], device=observed.device)[None])
    if detector_confidence:
        # Causal control: remove native identity-margin variation while keeping
        # the total known-pixel confidence mass identical for every query.
        original_mass = sum(((e.positive | e.negative).float() * e.confidence).sum(0) for e in examples)
        detector_mass = sum(((e.positive | e.negative).float() * c).sum(0)
                            for e, c in zip(examples, detector_confidence))
        scale = original_mass / detector_mass.clamp_min(1e-8)
        for example, confidence in zip(examples, detector_confidence):
            example.confidence = confidence * scale[None]
    if identity_gates is not None:
        for example in examples:
            entry = identity_gates[example.frame]
            if isinstance(entry, dict):
                ratio = torch.tensor([entry['ratio'][t] for t in texts], device=observed.device)[None]
                if scene['source_category_ambiguity'] == 'union_global':
                    example.confidence = example.confidence * ratio
                    continue
                gate = torch.where(entry['conflict'].to(observed.device)[:, None], ratio, torch.ones_like(ratio))
            else:
                gate = entry.to(observed.device)[:, None]
            example.confidence = example.confidence * torch.where(example.positive, gate, torch.ones_like(gate))
    return examples


@torch.inference_mode()
def refine_from_declared_sources(prior_logits, observed, scene, texts, identity_gates=None, observation_field=None, query_encodings=None):
    """Apply the declared source reader to native category logits."""
    if scene.get('source_region_mode'):
        from radio_gs.querying.region_observation_vote import fuse_region_observations
        return fuse_region_observations(prior_logits,observed,scene,texts)
    if scene.get('local_element_checkpoint') or scene.get('local_element_control'):
        from radio_gs.querying.local_element_fusion import fuse_local_elements
        return fuse_local_elements(prior_logits,observed,scene,texts)
    if scene.get("source_observation_model"):
        from radio_gs.querying.source_observation_likelihood import fuse_observation_likelihood
        return fuse_observation_likelihood(prior_logits, observed, scene, texts, observation_field, query_encodings)
    if scene.get('source_category_fusion') in ('union_native_categories', 'union_native_hard_categories', 'union_emitting_categories', 'union_verified_categories'):
        from radio_gs.querying.union_category_observation import fuse_union_category_observations
        return fuse_union_category_observations(prior_logits, observed, scene, texts)
    if scene.get('source_category_ambiguity') and identity_gates is None:
        from radio_gs.querying.source_category_ambiguity import source_category_ambiguity
        gates, report = source_category_ambiguity(scene, texts, prior_logits.device)
        logits, stats = refine_from_declared_sources(prior_logits, observed, scene, texts, identity_gates=gates)
        stats['source_category_ambiguity'] = report
        return logits, stats
    if scene.get('source_category_fusion') == 'categorical_observation_vote':
        from radio_gs.querying.category_observation import fuse_categorical_observations
        return fuse_categorical_observations(prior_logits, observed, scene, texts)
    batch_size = scene.get('source_query_batch_size', len(texts))
    if len(texts) > batch_size:
        # The prior is formed over the COMPLETE vocabulary before this call.
        # Scope inversion is independent per column: chunking it keeps every
        # source view and class without changing category competition.
        values, reports = [], []
        for start in range(0, len(texts), batch_size):
            value, report = refine_from_declared_sources(
                prior_logits[:, start:start + batch_size], observed, scene,
                texts[start:start + batch_size], identity_gates=identity_gates)
            values.append(value)
            reports.append(report)
        stats = dict(reports[0])
        for key in ('conflicting_elements', 'positive_entries',
                    'admitted_negative_entries', 'available_negative_entries',
                    'active_elements'):
            if key in stats:
                stats[key] = sum(report[key] for report in reports)
        stats['query_batch_size'] = batch_size
        return torch.cat(values, dim=1), stats
    if scene.get("source_streaming"):
        from radio_gs.querying.streamed_category_vote import streamed_category_vote
        return streamed_category_vote(prior_logits, observed, scene, texts)
    examples = declared_source_views(observed, scene, texts, identity_gates=identity_gates)
    prior = prior_logits.sigmoid()
    if scene.get('source_category_fusion') == 'visibility_weighted_vote':
        # Category partitions need reliable competing scores, not independent
        # high-frequency mask inversion. Direct visible evidence aggregation
        # retains the native prior wherever there is no supported observation.
        numerator, mass, visibility = [torch.zeros_like(prior) for _ in range(3)]
        negative_unknown = scene.get('source_negative_policy') == 'unknown'
        for example in examples:
            known = example.positive if negative_unknown else (example.positive | example.negative)
            weight = known.float() * example.confidence
            numerator += torch.sparse.mm(example.adjoint, example.target * weight)
            mass += torch.sparse.mm(example.adjoint, weight)
            visibility += torch.sparse.mm(example.adjoint, example.visible.float())
        value = (numerator + .1 * visibility * prior) / (mass + .1 * visibility).clamp_min(1e-8)
        logits = torch.where(mass > 0, torch.logit(value.clamp(1e-6, 1-1e-6)), prior_logits)
        return logits, dict(method='visibility_weighted_category_vote', views=len(examples),
            identity_weighting=scene.get('source_identity_control', 'native_margin'),
            steps=0, passes=1, prior_strength=.1, active_elements=int((mass > 0).sum()),
            positive_entries=sum(int(e.positive.sum()) for e in examples),
            available_negative_entries=sum(int(e.negative.sum()) for e in examples),
            admitted_negative_entries=0. if negative_unknown else sum(float((e.negative * e.confidence).sum()) for e in examples),
            outside_detected_scope='unknown' if negative_unknown else 'negative',
            positive_confidence_budget_unchanged=True)
    value, stats = solve_set(prior,examples,SetType.TEXT_UNION,corroborate=False,consistency=True)
    logits = torch.where(value != prior, torch.logit(value.clamp(1e-6,1-1e-6)), prior_logits)
    return logits, stats
