import pytest
from radio_gs.models.teacher_capabilities import TeacherCapabilities as T, validate_teacher_plan, validate_semantic_space


def test_functional_replacement_does_not_require_radio_or_siglip():
    clip=T('clip',frozenset({'text_image_alignment'}),'checkpoint-and-preprocessing-hash',512)
    plan=validate_teacher_plan([clip,T('mask_model',frozenset({'boundary'})),T('depth_model',frozenset({'geometry'}))],['text','image','prompt'])
    assert plan['text_image_alignment']==['clip']
    validate_semantic_space(clip,clip)


def test_visual_only_teacher_cannot_replace_text_alignment():
    with pytest.raises(ValueError):
        validate_teacher_plan([T('dino',frozenset({'visual_correspondence'})),T('depth',frozenset({'geometry'}))],['text'])


def test_equal_dimensions_do_not_make_spaces_compatible():
    with pytest.raises(ValueError):
        validate_semantic_space(T('a',frozenset({'text_image_alignment'}),'a',512),T('b',frozenset({'text_image_alignment'}),'b',512))
