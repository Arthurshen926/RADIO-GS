from radio_gs.scripts import eval_lerf_grounding as eval_lerf


def test_prompt_sam3_defaults_use_stable_support_gate():
    assert eval_lerf.DEFAULT_SAM3_PROMPT_MASK_HEAD_LOGIT_THRESHOLD == 0.0
    assert eval_lerf.DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_INITIAL_IOU == 0.50
    assert eval_lerf.DEFAULT_SAM3_PROMPT_MASK_HEAD_MIN_REFINED_AREA_RATIO == 0.70
    assert eval_lerf.DEFAULT_SAM3_PROMPT_MASK_HEAD_MAX_REFINED_AREA_RATIO == 1.30
    assert eval_lerf.DEFAULT_SAM3_PROMPT_MASK_HEAD_SUPPORT_DILATE == 12
