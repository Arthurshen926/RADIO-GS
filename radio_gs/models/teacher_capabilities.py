"""Functional teacher requirements, separate from checkpoint brand names."""
from dataclasses import dataclass


@dataclass(frozen=True)
class TeacherCapabilities:
    provider: str
    capabilities: frozenset
    space_id: str = ''
    dimension: int = 0


def validate_teacher_plan(teachers, query_modalities):
    allowed = {'text_image_alignment', 'visual_correspondence', 'boundary', 'geometry'}
    if not set(query_modalities) <= {'text', 'image', 'prompt'}:
        raise ValueError('unknown query modality')
    providers = {}
    for teacher in teachers:
        if not teacher.provider or teacher.provider in providers or not teacher.capabilities <= allowed:
            raise ValueError('invalid or duplicate teacher declaration')
        if 'text_image_alignment' in teacher.capabilities and (not teacher.space_id or teacher.dimension <= 0):
            raise ValueError('aligned semantic teacher needs an explicit vector space')
        providers[teacher.provider] = teacher
    capabilities = set().union(*(t.capabilities for t in teachers))
    required = {'geometry'}
    if 'text' in query_modalities:
        required.add('text_image_alignment')
    if 'image' in query_modalities and not capabilities & {'text_image_alignment', 'visual_correspondence'}:
        raise ValueError('image retrieval needs a visual capability')
    if 'prompt' in query_modalities:
        required.add('boundary')
    if not required <= capabilities:
        raise ValueError('missing functional capabilities: '+str(sorted(required-capabilities)))
    return {cap: sorted(t.provider for t in teachers if cap in t.capabilities) for cap in sorted(capabilities)}


def validate_semantic_space(teacher, query):
    if ('text_image_alignment' not in teacher.capabilities or 'text_image_alignment' not in query.capabilities
            or not teacher.space_id or not query.space_id or teacher.dimension <= 0 or query.dimension <= 0
            or teacher.space_id != query.space_id or teacher.dimension != query.dimension):
        raise ValueError('teacher and query vectors must use the same aligned encoder space')
