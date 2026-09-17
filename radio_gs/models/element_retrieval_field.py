"""One element latent and replaceable query adapters; no required VFM."""
from collections.abc import Mapping
import torch
from torch import nn
from torch.nn import functional as F

class NativePrefixDecoder(nn.Linear):
    """Decode the semantic coordinates of a single partitioned element vector."""
    def forward(self, latent):
        return F.linear(latent[..., :self.in_features],self.weight,self.bias)

class ElementRetrievalField(nn.Module):
    def __init__(self, num_elements: int, latent_dim: int, query_dims: Mapping[str,int], shared_query_groups=None, calibrated_readout=False, native_siglip_readout=False, standardize_native_text=False, native_semantic_dim=None, structural_membership=None, scale_affinity=None, native_query_thresholds=None, source_membership_code=None, grounded_transfer=None, bounded_extent=None, native_semantic_residual=None, image_support_extension=False, extent_benefit_admission=False):
        super().__init__()
        if standardize_native_text and not native_siglip_readout:
            raise ValueError('text score standardization requires native readout')
        self.standardize_native_text=standardize_native_text
        self.bounded_extent = bounded_extent
        if bounded_extent is not None:
            import math
            if (not native_siglip_readout or not grounded_transfer or not grounded_transfer.get('num_proposals') or grounded_transfer.get('hidden_dim',0)
                or standardize_native_text or structural_membership is not None or source_membership_code is not None
                or (scale_affinity and scale_affinity.get('native_seed_queries',False))
                or not bounded_extent or not set(bounded_extent)<= {'text','image'}
                or any(not math.isfinite(v) or not 0<=v<=2 for v in bounded_extent.values())):
                raise ValueError('bounded semantic extent requires native readout, linear proposal extents and strengths in [0,2]')
        if native_semantic_dim is not None and (not native_siglip_readout or not 0<native_semantic_dim<latent_dim):
            raise ValueError('native semantic partition requires a nonempty structural suffix')
        self.native_semantic_dim=native_semantic_dim
        if num_elements < 1 or latent_dim < 1 or not query_dims:
            raise ValueError("nonempty element/query dimensions required")
        if not set(query_dims) <= {"text","image","prompt"} or any(v<1 for v in query_dims.values()):
            raise ValueError("invalid query modalities or dimensions")
        self.latent=nn.Parameter(torch.randn(num_elements,latent_dim)*.01)
        self.query_adapters=nn.ModuleDict({m:nn.Linear(d,latent_dim,bias=False) for m,d in query_dims.items()})
        self.readout_log_scale=nn.ParameterDict({m:nn.Parameter(torch.tensor(10.).log()) for m in query_dims} if calibrated_readout else {})
        self.readout_bias=nn.ParameterDict({m:nn.Parameter(torch.tensor(-5.)) for m in query_dims} if calibrated_readout else {})
        used = set()
        for group in shared_query_groups or []:
            if len(group) < 2 or len(set(group)) != len(group) or not set(group) <= set(query_dims) or used.intersection(group):
                raise ValueError("invalid shared query adapter group")
            if len({query_dims[m] for m in group}) != 1:
                raise ValueError("shared query encoders must have equal dimensions")
            # Sharing is explicit: equal dimensions alone do not imply aligned spaces.
            for modality in group[1:]:
                self.query_adapters[modality].weight = self.query_adapters[group[0]].weight
            used.update(group)
        self.native_decoder = None
        if native_siglip_readout:
            if not {'text','image'} <= set(query_dims) or query_dims['text']!=query_dims['image'] or self.query_adapters['text'].weight is not self.query_adapters['image'].weight:
                raise ValueError('native readout requires explicitly shared native text/image encoders')
            self.native_decoder=(NativePrefixDecoder(native_semantic_dim,query_dims['text'])
                                 if native_semantic_dim is not None else nn.Linear(latent_dim,query_dims['text']))
            self.query_adapters['text'].requires_grad_(False)
            self.query_adapters['image'].requires_grad_(False)
        self.native_semantic_residual = native_semantic_residual
        self.native_residual_decoder = None
        if native_semantic_residual is not None:
            import math
            bound = native_semantic_residual.get('relative_bound')
            if (self.native_decoder is None or not {'relative_bound'} <= set(native_semantic_residual) <= {'relative_bound','modalities'}
                or not isinstance(bound, (int, float)) or not math.isfinite(bound) or not 0 <= bound <= .5
                or not native_semantic_residual.get('modalities',['text','image'])
                or not set(native_semantic_residual.get('modalities',['text','image'])) <= {'text','image'}
                or source_membership_code is not None):
                raise ValueError('native residual requires native readout and relative_bound in [0,.5]')
            self.native_residual_decoder = nn.Linear(latent_dim, query_dims['text'])
            nn.init.zeros_(self.native_residual_decoder.weight)
            nn.init.zeros_(self.native_residual_decoder.bias)
        self.register_buffer("observed",torch.zeros(num_elements,dtype=torch.bool))
        if image_support_extension and (not native_siglip_readout or structural_membership is not None
            or source_membership_code is not None or (bounded_extent and 'image' in bounded_extent)
            or (scale_affinity and scale_affinity.get('native_seed_queries',False))):
            raise ValueError('image support extension requires a plain native image readout')
        self.register_buffer('image_observed',torch.zeros(num_elements,dtype=torch.bool) if image_support_extension else None)
        self.native_query_thresholds = nn.ModuleDict()
        if native_query_thresholds:
            if not native_siglip_readout or standardize_native_text or structural_membership is not None or (scale_affinity and scale_affinity.get('native_seed_queries', False)):
                raise ValueError('query thresholds require plain native cosine readout')
            from radio_gs.models.native_query_threshold import NativeQueryThreshold
            for modality, config in native_query_thresholds.items():
                if modality not in ('text','image') or modality not in query_dims:
                    raise ValueError('native text/image threshold modality required')
                self.native_query_thresholds[modality] = NativeQueryThreshold(query_dims[modality], **config)
        self.affinity = None
        if scale_affinity is not None:
            if native_semantic_dim is None:raise ValueError('scale affinity requires a structural partition')
            from radio_gs.models.scale_affinity_readout import ScaleAffinityReadout
            self.affinity=ScaleAffinityReadout(latent_dim-native_semantic_dim, **scale_affinity)
        self.membership = None
        if structural_membership is not None:
            if native_semantic_dim is None:
                raise ValueError('structural membership requires native partition')
            from radio_gs.models.structural_membership_readout import StructuralMembershipReadout
            self.membership = StructuralMembershipReadout(latent_dim-native_semantic_dim, **structural_membership)
        self.source_membership = None
        if source_membership_code is not None:
            if native_semantic_dim is None or native_query_thresholds or structural_membership is not None or standardize_native_text:
                raise ValueError('source membership code requires a plain native/structural partition')
            if 'prompt' in query_dims and self.affinity is None:
                raise ValueError('source membership prompt requires registered geometry')
            from radio_gs.models.source_membership_code import SourceMembershipCode
            self.source_membership=SourceMembershipCode(latent_dim-native_semantic_dim,**source_membership_code)
        self.extent_benefit_admission = None
        if extent_benefit_admission:
            if not grounded_transfer or bounded_extent is not None:
                raise ValueError('Extent benefit admission requires ordinary grounded readout')
            from radio_gs.models.extent_benefit_gate import ExtentBenefitGate, ExtentUtilityTree, SemanticExtentBenefitGate
            if extent_benefit_admission is True:
                self.extent_benefit_admission = ExtentBenefitGate()
            elif extent_benefit_admission == {'kind':'utility_tree'}:
                self.extent_benefit_admission = ExtentUtilityTree()
            elif extent_benefit_admission == {'kind':'semantic_linear'}:
                if not (grounded_transfer.get('routing') or {}).get('scope_gate',False):
                    raise ValueError('Semantic admission requires existing source scope prototypes')
                self.extent_benefit_admission = SemanticExtentBenefitGate()
            else:
                raise ValueError('Unknown extent benefit admission configuration')
        self.grounded_transfer=None
        if grounded_transfer is not None:
            if native_semantic_dim is None or source_membership_code is not None or structural_membership is not None or standardize_native_text or (scale_affinity and scale_affinity.get('native_seed_queries',False)):
                raise ValueError('grounded transfer requires an unchanged plain native/structural field')
            from radio_gs.models.grounded_semantic_transfer import GroundedSemanticTransfer
            self.grounded_transfer=GroundedSemanticTransfer(latent_dim,native_semantic_dim,query_dims['text'],**grounded_transfer)

    def native_cosine(self, query, rows=None, *, modality="text"):
        z = self.latent if rows is None else self.latent[rows]
        if self.native_residual_decoder is not None and modality in self.native_semantic_residual.get('modalities',['text','image']):
            from radio_gs.models.bounded_native_residual import bounded_native_cosine
            corrected = bounded_native_cosine(z, query, self.native_decoder.weight, self.native_decoder.bias,
                                             self.native_residual_decoder.weight, self.native_residual_decoder.bias,
                                             self.native_semantic_residual["relative_bound"])
            if modality == 'image' and self.image_observed is not None:
                # Extension rows have measured native evidence, but no fitted
                # structural/residual supervision. Do not extrapolate that head.
                from radio_gs.models.native_decoder_readout import affine_decoder_cosine
                original = affine_decoder_cosine(z[:,:self.native_decoder.in_features],query,self.native_decoder.weight,self.native_decoder.bias)
                trained = self.observed if rows is None else self.observed[rows]
                corrected = torch.where(trained[:,None],corrected,original)
            return corrected
        from radio_gs.models.native_decoder_readout import affine_decoder_cosine
        return affine_decoder_cosine(z[:,:self.native_decoder.in_features],query,
                                     self.native_decoder.weight,self.native_decoder.bias)


    def logits(self, query, modality, rows=None):
        if modality not in self.query_adapters:
            raise ValueError("query modality has no configured encoder adapter")
        if query.ndim!=2 or query.shape[1]!=self.query_adapters[modality].in_features or not torch.isfinite(query).all():
            raise ValueError("query must be finite [Q,encoder_dim]")
        if self.source_membership is not None:
            return self.source_membership(query,modality,self.latent[:,self.native_semantic_dim:],self.observed,self.affinity,rows)
        if self.bounded_extent is not None and modality in self.bounded_extent:
            from radio_gs.models.bounded_extent_readout import read_bounded_extent
            return read_bounded_extent(self,query,modality,self.bounded_extent[modality],rows)
        if self.grounded_transfer is not None and modality=='text':
            full_cosine=self.native_cosine(query)
            cosine=full_cosine
            if rows is not None:cosine=cosine[rows]
            if modality in self.native_query_thresholds:
                scale=self.readout_log_scale[modality].clamp(-4.,6.).exp() if modality in self.readout_log_scale else 10.
                native_threshold=self.native_query_thresholds[modality](query)
                original=scale*(cosine-native_threshold[None])
            else:
                original=self.cosine_logits(cosine,modality)
                native_threshold=(-self.readout_bias[modality]/self.readout_log_scale[modality].clamp(-4.,6.).exp()
                                  if modality in self.readout_log_scale else .5)
            selection=self.grounded_transfer.identity_selection
            support=(full_cosine>native_threshold if selection is not None and selection.get('metric')=='support_iou' else None)
            preserve=selection is not None and selection.get('completion_mode') in ('native_threshold','native_within_extent')
            global_readout=preserve or self.extent_benefit_admission is not None
            proposed,accepted=self.grounded_transfer(query,self.latent,None if global_readout else rows,native_threshold=native_threshold,
                                                     native_support=support,observed=self.observed)
            if preserve:
                from radio_gs.models.grounded_semantic_transfer import rank_preserving_extent_threshold
                _,concept_accepted,_=self.grounded_transfer.route(query)
                refine=(accepted if selection.get('calibrate_accepted_concepts',False)
                        else accepted&~concept_accepted)
                if refine.any():
                    native_scale=self.readout_log_scale[modality].clamp(-4.,6.).exp() if modality in self.readout_log_scale else 10.
                    if selection.get('completion_mode')=='native_within_extent':
                        probability=torch.sigmoid(native_scale*(full_cosine-native_threshold))[:,refine]
                        probability=.5*((proposed[:,refine]>0).to(probability)+probability)
                        proposed[:,refine]=torch.logit(probability.clamp(1e-6,1-1e-6))
                    else:
                        cutoff=rank_preserving_extent_threshold(full_cosine[:,refine],proposed[:,refine]>0,self.observed)
                        proposed[:,refine]=native_scale*(full_cosine[:,refine]-cutoff)
            if self.extent_benefit_admission is not None:
                _,concept_accepted,_=self.grounded_transfer.route(query)
                accepted=accepted & self.extent_benefit_admission(
                    full_cosine,native_threshold,proposed,self.observed,concept_accepted,
                    query=query,head=self.grounded_transfer)
            if global_readout and rows is not None:proposed=proposed[rows]
            return torch.where(accepted[None],proposed,original)
        if self.affinity is not None and self.affinity.native_seed_queries and modality in ('text','image'):
            return self.affinity.native_logits(self.native_cosine(query,modality=modality),self.latent[:,self.native_semantic_dim:],self.observed,rows)
        if self.affinity is not None and modality=='prompt':
            return self.affinity(query,self.latent if self.affinity.full_prompt_features else self.latent[:,self.native_semantic_dim:],self.observed,rows)
        if self.membership is not None and modality in ('text','image'):
            cosine = self.native_cosine(query,modality=modality)
            return torch.stack([self.membership(cosine[:,i], self.latent[:,self.native_semantic_dim:],
                                 self.observed, modality, rows) for i in range(len(query))], dim=1)
        z=self.latent if rows is None else self.latent[rows]
        if self.native_decoder is not None and modality in ('text','image'):
            from radio_gs.models.native_decoder_readout import affine_decoder_cosine
            z=z[:,:self.native_decoder.in_features]
            if modality in self.native_query_thresholds:
                cosine=self.native_cosine(query,rows,modality=modality)
                scale=self.readout_log_scale[modality].clamp(-4.,6.).exp() if modality in self.readout_log_scale else 10.
                return scale*(cosine-self.native_query_thresholds[modality](query)[None])
            if modality=='text' and self.standardize_native_text:
                cosine=self.native_cosine(query)
                standardized=standardize_observed_scores(cosine,self.observed)
                return self.cosine_logits(standardized if rows is None else standardized[rows],modality)
            return self.cosine_logits(self.native_cosine(query,rows,modality=modality),modality)
        q=F.normalize(self.query_adapters[modality](query),dim=-1)
        return self.cosine_logits(F.normalize(z,dim=-1)@q.T,modality)

    def cosine_logits(self, cosine, modality, *, detach_calibration=False):
        if modality not in self.query_adapters:raise ValueError('unknown readout modality')
        if modality not in self.readout_log_scale:return 10*cosine-5
        scale=self.readout_log_scale[modality].clamp(-4.,6.).exp()
        bias=self.readout_bias[modality]
        if detach_calibration:scale,bias=scale.detach(),bias.detach()
        return scale*cosine+bias

    def observed_for(self, modality):
        if modality not in self.query_adapters:raise ValueError('unknown support modality')
        return self.image_observed if modality=='image' and self.image_observed is not None else self.observed

    def retrieve(self, query, modality, rows=None):
        support = self.observed_for(modality)
        known=support if rows is None else support[rows]
        scores=torch.sigmoid(self.logits(query,modality,rows))
        return {"scores":torch.where(known[:,None],scores,torch.zeros_like(scores)),
                "valid":known}


def standardize_observed_scores(scores, observed):
    """Query-wise moments on the complete map, never on target pixels or labels.

    Element-count weighting is explicit; this is not an object probability.
    Degenerate constant scores map to zero and require calibrated abstention.
    """
    if scores.ndim!=2 or observed.shape!=(len(scores),) or observed.dtype!=torch.bool or not observed.any():
        raise ValueError('observed map required for score standardization')
    values=scores[observed]
    return (scores-values.mean(0))/values.std(0,unbiased=False).clamp_min(1e-6)


def load_element_retrieval_field(path, *, expected_sha256, geometry, expected_query_encoder, device="cpu"):
    """Cold-load a candidate only with matching geometry and encoder identity."""
    from radio_gs.utils.immutable_artifacts import sha256_file
    if not expected_sha256 or sha256_file(path)!=expected_sha256:
        raise ValueError("retrieval field file identity differs")
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema")!="radio_gs.element_retrieval_field.candidate.v1":
        raise ValueError("not an element retrieval field")
    geometry.assert_binding(payload.get("geometry_binding"))
    if payload.get("query_encoder")!=expected_query_encoder:
        raise ValueError("query encoder identity differs; equal dimensions are insufficient")
    field=ElementRetrievalField(**payload["architecture"])
    if len(field.latent)!=len(geometry.get_xyz()):
        raise ValueError("retrieval field and geometry row counts differ")
    field.load_state_dict(payload["state_dict"],strict=True)
    if field.image_observed is not None and (field.observed & ~field.image_observed).any():
        raise ValueError('image support extension cannot remove original support')
    if not all(bool(torch.isfinite(v).all()) for v in field.parameters()):
        raise ValueError("nonfinite retrieval field")
    field.to(device).eval()
    if field.membership is not None:
        field.membership.bind_geometry(geometry.get_xyz().to(device), field.observed)
    if field.affinity is not None:
        field.affinity.bind_geometry(geometry.get_xyz().to(device),payload['query_context']['prompt_world_normalization'])
    return field,payload


def retrieve_contrastive_text(field, query, negative_queries):
    """Optional transient text readout against explicit generic alternatives.

    The same element field and frozen adapter encode both sides. No per-query
    feature map is stored; output remains a single element posterior shared by
    2D/3D. This is a different readout contract from absolute membership.
    """
    if negative_queries.ndim!=2 or len(negative_queries)==0:
        raise ValueError('at least one encoded generic alternative is required')
    positive=field.logits(query,'text')
    negative=field.logits(negative_queries,'text').amax(dim=1,keepdim=True)
    scores=torch.sigmoid(positive-negative)
    return {'scores':torch.where(field.observed[:,None],scores,torch.zeros_like(scores)),
            'valid':field.observed}
