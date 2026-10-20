"""Prepare an explicit legacy replay config from a verified v2 field.

Does not launch a benchmark or modify any frozen configuration/output.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import torch

from .contracts import QueryPacket, Support
from .shared_storage import _read_manifest, _descriptor, export_field_v1
from .storage import load_field, sha256


def _parent(config, pointer):
    if not pointer.startswith("/") or pointer == "/":
        raise ValueError("field pointer must identify one existing JSON value")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer[1:].split("/")]
    node = config
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    key = int(parts[-1]) if isinstance(node, list) else parts[-1]
    return node, key


def prepare_replay(field_path, store_root, legacy_config, field_pointer,
                   expected_legacy_field, deployment, output_config, *, expected_source_sha256=None, reuse_deployment=False,
                   max_export_bytes=16*1024**3):
    source = Path(legacy_config)
    original_sha = sha256(source)
    config = json.loads(source.read_text())
    node, key = _parent(config, field_pointer)
    if node[key] != expected_legacy_field:
        raise ValueError("legacy field pointer does not match the explicitly selected source")
    target = Path(output_config)
    receipt_path = target.with_suffix(target.suffix + ".replay.json")
    if target.exists() or receipt_path.exists() or target.resolve() == source.resolve():
        raise FileExistsError("new replay configuration and receipt must not exist")
    _, shared = _read_manifest(field_path)
    recorded_source = shared.get("provenance", {}).get("source_field_sha256")
    if expected_source_sha256 is None or recorded_source != expected_source_sha256:
        raise ValueError("v2 migration does not bind the explicitly selected legacy source digest")
    deployment = Path(deployment)
    if reuse_deployment:
        if not deployment.is_dir():
            raise ValueError("explicit reuse requires an existing standalone deployment")
    else:
        # A caller chooses one fixed cache slot; we never rotate names or delete.
        estimate = shared["tensor_bytes"] + 65536*len(shared["tensors"]) + 1024**2
        if max_export_bytes < 0 or estimate > max_export_bytes:
            raise ValueError("standalone export exceeds explicit slot budget")
        ancestor = deployment.parent
        while not ancestor.exists():
            ancestor = ancestor.parent
        if shutil.disk_usage(ancestor).free < estimate + 1024**3:
            raise OSError("insufficient standalone cache space including reserve")
        export_field_v1(field_path, deployment, store_root=store_root)
    field, deployed = load_field(deployment, "cpu")
    if deployed.get("provenance", {}).get("parent_state_identity") != shared["state_identity"]:
        raise ValueError("standalone deployment belongs to another v2 state")
    if ({k: _descriptor(v)[0] for k, v in field.state_dict().items()} != shared["tensors"]
            or field.export_spec() != shared["spec"]):
        raise ValueError("legacy standalone state or spec differs")
    rows = torch.where(field.observed)[0]
    if not len(rows):
        raise ValueError("cold replay probe requires an observed element")
    packet = QueryPacket("point_3d", positive=Support(rows[:1], torch.ones(1)), exact_positive=True)
    expected = field.query(packet)
    with patch("builtins.open", side_effect=AssertionError("query file access")), \
         patch("io.open", side_effect=AssertionError("query file access")), \
         patch("torch.load", side_effect=AssertionError("query file access")):
        actual = field.query(packet)
    if not (torch.equal(expected.membership_scores, actual.membership_scores)
            and torch.equal(expected.selected, actual.selected)):
        raise ValueError("cold query differs")
    replacement = copy.deepcopy(config)
    replaced_node, replaced_key = _parent(replacement, field_pointer)
    replaced_node[replaced_key] = str(deployment.resolve())
    if sha256(source) != original_sha:
        raise ValueError("original configuration changed during preparation")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        json.dump(replacement, handle, indent=2)
        handle.write("\n")
    receipt = dict(schema="radio_gs.legacy_replay_adapter.v1",
                   original_config=str(source), original_config_sha256=original_sha,
                   original_config_unchanged=True, output_config=str(target),
                   output_config_sha256=sha256(target), field_pointer=field_pointer,
                   original_field=expected_legacy_field, explicit_override=str(deployment.resolve()),
                   expected_source_sha256=expected_source_sha256,
                   v2_state_identity=shared["state_identity"],
                   standalone_field_sha256=deployed["files"]["field.pt"]["sha256"],
                   raw_tensor_state_exact=True, spec_exact=True, cold_query_file_access_denied=True,
                   source_preserved=True, benchmark_launched=False,
                   other_config_values_unchanged=True, deployment_reused=reuse_deployment,
                   max_export_bytes=max_export_bytes,
                   cache_policy="explicit fixed slot; no automatic rotation, duplication, or deletion")
    with receipt_path.open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--store-root", required=True)
    parser.add_argument("--legacy-config", required=True)
    parser.add_argument("--field-pointer", required=True)
    parser.add_argument("--expected-legacy-field", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--reuse-deployment", action="store_true")
    parser.add_argument("--max-export-bytes", type=int, default=16*1024**3)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    print(json.dumps(prepare_replay(args.field, args.store_root, args.legacy_config,
        args.field_pointer, args.expected_legacy_field, args.deployment, args.output_config,
        expected_source_sha256=args.expected_source_sha256, reuse_deployment=args.reuse_deployment,
        max_export_bytes=args.max_export_bytes), indent=2))


if __name__ == "__main__":
    main()
