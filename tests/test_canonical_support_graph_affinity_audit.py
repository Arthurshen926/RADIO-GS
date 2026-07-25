import torch

from radio_gs.scripts.audit_canonical_support_graph_affinity import audit


def _save_graph(path, *, mode, edge_index, appearance, boundary):
    torch.save(
        {
            "schema_version": 1,
            "global_rows": torch.tensor([0, 2, 3]),
            "num_global_rows": 4,
            "edge_index": edge_index,
            "edge_channels": {
                "appearance": torch.tensor(appearance),
                "boundary": torch.tensor(boundary),
            },
            "metadata": {"capability_affinity": {"mode": mode}},
        },
        path,
    )


def test_audit_compares_only_aligned_label_free_capability_graphs(tmp_path):
    edges = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])
    hashed = tmp_path / "hashed.pt"
    native = tmp_path / "native.pt"
    _save_graph(
        hashed,
        mode="signed_hash",
        edge_index=edges,
        appearance=[0.10, 0.90, 0.20, 0.80, 0.30, 0.70],
        boundary=[0.60, 0.40, 0.55, 0.45, 0.50, 0.50],
    )
    _save_graph(
        native,
        mode="exact_official_capability",
        edge_index=edges,
        appearance=[0.20, 0.80, 0.25, 0.75, 0.35, 0.65],
        boundary=[0.65, 0.35, 0.60, 0.40, 0.55, 0.45],
    )

    report = audit(hashed, native)

    assert report["mode"] == "label_free_canonical_capability_affinity_audit"
    assert report["topology"] == {
        "identical": True,
        "capability_valid_nodes": 3,
        "global_primitive_rows": 4,
        "directed_edges": 6,
    }
    appearance = report["capability_affinity"]["channels"]["appearance"]
    assert appearance["mean_absolute_delta"] > 0
    assert appearance["sampled_outgoing_top1_agreement"] == 1.0
    assert report["labels_opened"] is False
    assert report["queries_opened"] is False


def test_audit_rejects_a_topology_change(tmp_path):
    hashed = tmp_path / "hashed.pt"
    native = tmp_path / "native.pt"
    _save_graph(
        hashed,
        mode="signed_hash",
        edge_index=torch.tensor([[0, 0], [1, 2]]),
        appearance=[0.1, 0.2],
        boundary=[0.3, 0.4],
    )
    _save_graph(
        native,
        mode="exact_official_capability",
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        appearance=[0.1, 0.2],
        boundary=[0.3, 0.4],
    )

    try:
        audit(hashed, native)
    except ValueError as error:
        assert "topology" in str(error)
    else:  # pragma: no cover - makes the intended fail-closed behavior explicit
        raise AssertionError("topology change should fail")
