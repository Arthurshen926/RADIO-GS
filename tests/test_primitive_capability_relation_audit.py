from argparse import Namespace

import torch

from radio_gs.scripts.audit_primitive_capability_relation import audit


def test_primitive_relation_audit_separates_rendering_from_field(tmp_path):
    xyz = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    teacher = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    prediction = teacher.clone()
    capability = tmp_path / "capability.pt"
    target = tmp_path / "target.pt"
    graph = tmp_path / "graph.pt"
    output = tmp_path / "audit.json"
    torch.save(
        {"xyz": xyz, "valid": torch.ones(3, dtype=torch.bool), "boundary_sam3": prediction},
        capability,
    )
    torch.save(
        {"xyz": xyz, "valid": torch.ones(3, dtype=torch.bool), "features": teacher},
        target,
    )
    torch.save(
        {
            "global_rows": torch.arange(3),
            "edge_index": torch.tensor([[0, 1, 0], [1, 2, 2]]),
        },
        graph,
    )
    report = audit(
        Namespace(
            capability_cache=str(capability),
            target_mpr_cache=str(target),
            support_graph=str(graph),
            capability_key="boundary_sam3",
            chunk_size=2,
            output=str(output),
        )
    )
    assert report["rendering_used"] is False
    assert report["row_cosine"]["mean"] > 0.999
    assert report["local_relation"]["boundary_margin_retention"] > 0.999
    assert output.is_file()
