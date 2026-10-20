"""Self-contained scene exports; provenance never grants runtime file access."""
import hashlib
import json
from pathlib import Path

import torch
from .field import CapabilityField


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deployment_bytes(manifest):
    """Count the complete model payload, including every referenced shared blob."""
    if manifest["schema"] == "radio_gs.query_preserving_field.v2":
        return manifest["unique_blob_bytes"] + len(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) + 1
    return sum(item["bytes"] for item in manifest["files"].values())


def field_identity(manifest):
    if manifest["schema"] == "radio_gs.query_preserving_field.v2":
        return manifest["state_identity"]
    return manifest["files"]["field.pt"]["sha256"]


def save_field(field, directory, provenance):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    state = {key: value.detach().cpu() for key, value in field.state_dict().items()}
    torch.save(state, root / "field.pt")
    manifest = dict(schema="radio_gs.query_preserving_field.v1", spec=field.export_spec(),
                    files={"field.pt": dict(sha256=sha256(root / "field.pt"),
                                            bytes=(root / "field.pt").stat().st_size)},
                    elements=len(field.xyz), observed=int(field.observed.sum()),
                    inferred_anchor_domain="all_observed" if field.anchor_eligible is None else "training_eligible",
                    eligible_anchors=None if field.anchor_eligible is None else int(field.anchor_eligible.sum()),
                    semantic_native_dim=field.semantic.native_dim,
                    local_native_dim=field.local.native_dim,
                    semantic_code_dim=field.semantic.codes.shape[1],
                    local_code_dim=field.local.codes.shape[1], group_dim=field.group.shape[1],
                    provenance=provenance, external_scene_dependencies=[],
                    query_encoder_cost_excluded=True,
                    geometry_bytes=sum(state[k].numel()*state[k].element_size() for k in
                                       ["xyz", "normals", "quaternions", "scales", "opacities"]),
                    tensor_bytes=sum(v.numel()*v.element_size() for v in state.values()))
    manifest["semantic_increment_bytes"] = manifest["tensor_bytes"] - manifest["geometry_bytes"]
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_field(directory, device="cpu"):
    root = Path(directory).resolve(strict=True)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema"] != "radio_gs.query_preserving_field.v1":
        raise ValueError("unsupported field schema")
    if manifest.get("external_scene_dependencies") != [] or set(manifest["files"]) != {"field.pt"}:
        raise ValueError("a deployed scene must be self-contained")
    path = root / "field.pt"
    if path.is_symlink() or path.resolve().parent != root:
        raise ValueError("deployment payload must be local to the scene package")
    if sha256(path) != manifest["files"]["field.pt"]["sha256"]:
        raise ValueError("deployment field digest differs")
    state = torch.load(path, map_location="cpu", weights_only=True)
    return field_from_state(state, manifest, device), manifest


def load_field_versioned(directory, device="cpu", *, store_root=None):
    """Explicit version-aware entry; shared v2 never guesses its blob store."""
    path = Path(directory) / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 16*1024*1024:
        raise ValueError("invalid field manifest path or size")
    schema = json.loads(path.read_text()).get("schema")
    if schema == "radio_gs.query_preserving_field.v1":
        return load_field(directory, device)
    if schema == "radio_gs.query_preserving_field.v2":
        from .shared_storage import load_field_v2
        return load_field_v2(directory, device, store_root=store_root)
    raise ValueError("unsupported field schema")


def field_from_state(state, manifest, device="cpu", *, preserve_dtypes=False):
    """Construct exactly the same typed field from a validated state mapping."""
    if manifest["spec"].get("architecture") == "unified_compact_dictionary_v3":
        from .unified_dictionary import from_state
        return from_state(state, manifest["spec"], device)
    if manifest["spec"].get("architecture") == "unified_compact_modes_v2":
        from .unified_modes import from_state
        return from_state(state, manifest["spec"], device)
    if manifest["spec"].get("architecture") == "unified_compact_v1":
        from .unified_compact import from_state
        return from_state(state, manifest["spec"], device)
    geometry = {k: state[k] for k in ["xyz", "normals", "quaternions", "scales", "opacities"]}
    codecs = {name: dict(weight=state.get(name+".weight"), bias=state.get(name+".bias"))
              for name in ["semantic", "local"]}
    field = CapabilityField(geometry, state["semantic.codes"], state["local.codes"], state["group"],
                            state["observed"], semantic_codec=codecs["semantic"],
                            local_codec=codecs["local"], anchor_eligible=state.get("anchor_eligible"),
                            relation_confidence=state.get("relation_confidence"),
                            semantic_measure_state={k.removeprefix('semantic_measure.'):v for k,v in state.items()
                                                    if k.startswith('semantic_measure.')} or None,
                            native_mode_state={k.removeprefix('native_modes.'):v for k,v in state.items()
                                               if k.startswith('native_modes.')} or None,
                            region_carrier_state={k.removeprefix('region_carrier.'):v for k,v in state.items()
                                                  if k.startswith('region_carrier.')} or None,
                            support_footprint_state={k.removeprefix('support_footprint.'):v for k,v in state.items()
                                                    if k.startswith('support_footprint.')} or None,
                            coverage_factor_state={k.removeprefix('coverage_factors.'):v for k,v in state.items()
                                                    if k.startswith('coverage_factors.')} or None,
                            extent_factor_state={k.removeprefix('extent_factors.'):v for k,v in state.items()
                                                 if k.startswith('extent_factors.')} or None,
                            extent_preparation_state={k.removeprefix('extent_preparation.'):v for k,v in state.items()
                                                      if k.startswith('extent_preparation.')} or None,
                            scope_topology_state={k.removeprefix('scope_topology.'):v for k,v in state.items()
                                                  if k.startswith('scope_topology.')} or None,
                            scope_constraint_state={k.removeprefix('scope_constraints.'):v for k,v in state.items()
                                                    if k.startswith('scope_constraints.')} or None,
                            scope_permission_state={k.removeprefix('scope_permissions.'):v for k,v in state.items()
                                                    if k.startswith('scope_permissions.')} or None,
                            evidence_state={k.removeprefix('evidence.'): v for k, v in state.items()
                                            if k.startswith('evidence.')} or None, **manifest["spec"])
    if preserve_dtypes:
        # Constructors historically normalize selected buffers/parameters to fp32.
        # The v2 lossless path restores serialized dtypes before strict copying.
        for key, value in state.items():
            module_name, _, leaf = key.rpartition(".")
            module = field.get_submodule(module_name) if module_name else field
            if leaf in module._parameters and module._parameters[leaf] is not None:
                parameter = module._parameters[leaf]
                module._parameters[leaf] = torch.nn.Parameter(parameter.to(value.dtype), requires_grad=parameter.requires_grad)
            elif leaf in module._buffers and module._buffers[leaf] is not None:
                module._buffers[leaf] = module._buffers[leaf].to(value.dtype)
    field.load_state_dict(state, strict=True)
    return field.to(device).eval()
