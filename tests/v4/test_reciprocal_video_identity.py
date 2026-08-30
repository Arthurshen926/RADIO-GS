from radio_gs.v4.contracts.build_reciprocal_video_identity import _reciprocal_edges


def _pair(edges):
    return {"edges": edges}


def _edge(source, target, iou):
    return {
        "source_proposal_index": source,
        "target_proposal_index": target,
        "tracked_to_target_root_iou": iou,
    }


def test_reciprocal_edges_require_exact_return_and_both_thresholds():
    forward = _pair([_edge(1, 4, 0.9), _edge(2, 5, 0.9), _edge(3, 6, 0.9)])
    reverse = _pair([_edge(4, 1, 0.8), _edge(5, 7, 0.9), _edge(6, 3, 0.6)])

    result = _reciprocal_edges(forward, reverse, 0.7)

    assert [(edge["source_proposal_index"], edge["target_proposal_index"]) for edge in result] == [(1, 4)]
    assert result[0]["tracked_to_target_root_iou"] == 0.8


def test_reciprocal_edges_resolve_duplicate_target_before_agreement():
    forward = _pair([_edge(1, 4, 0.7), _edge(2, 4, 0.8)])
    reverse = _pair([_edge(4, 2, 0.9)])

    assert _reciprocal_edges(forward, reverse, 0.7)[0]["source_proposal_index"] == 2
