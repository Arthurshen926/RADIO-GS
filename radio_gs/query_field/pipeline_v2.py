"""Explicit v2 research pipeline. Legacy frozen entry points remain v1."""
import argparse
from functools import partial
import json
from .shared_storage import save_field_v2, load_field_v2, export_field_v1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "scannet-predict", "scannet-score", "export-v1", "migrate-v1"])
    parser.add_argument("--store-root")
    parser.add_argument("--field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations")
    parser.add_argument("--config")
    parser.add_argument("--queries")
    parser.add_argument("--encoder")
    parser.add_argument("--evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--query-program", default="original")
    parser.add_argument("--max-new-bytes", type=int)
    parser.add_argument("--reserve-bytes", type=int, default=1024**3)
    args = parser.parse_args(argv)
    if args.stage in {"train", "migrate-v1"} and args.store_root is None:
        parser.error("--store-root is required for shared research writes")
    save = partial(save_field_v2, store_root=args.store_root,
                   max_new_bytes=args.max_new_bytes, reserve_bytes=args.reserve_bytes)
    if args.stage == "train":
        from .train import fit
        with open(args.config) as handle:
            config = json.load(handle)
        result = fit(args.observations, args.output, config, args.device, save_function=save)
    elif args.stage == "scannet-predict":
        from .public_scannet import predict
        result = predict(args.field, args.queries, args.output, args.encoder, args.device,
                         args.query_program, field_loader=partial(load_field_v2, store_root=args.store_root))
    elif args.stage == "scannet-score":
        from .public_scannet import score
        result = score(args.field, args.evaluation, args.output)
    elif args.stage == "export-v1":
        result = export_field_v1(args.field, args.output, store_root=args.store_root)
    else:
        from .storage import load_field
        field, manifest = load_field(args.field)
        import torch
        from pathlib import Path
        from .shared_storage import _descriptor
        raw = torch.load(Path(args.field)/"field.pt", map_location="cpu", weights_only=True)
        if {k:_descriptor(v)[0] for k,v in raw.items()} != {k:_descriptor(v)[0] for k,v in field.state_dict().items()}:
            raise ValueError("legacy constructor changed raw state; migration refused")
        del raw
        result = save(field, args.output, dict(original_manifest=manifest,
                      source_field_sha256=manifest["files"]["field.pt"]["sha256"]))
        restored, _ = load_field_v2(args.output, store_root=args.store_root)
        import torch
        if field.export_spec() != restored.export_spec() or any(
                original.dtype != restored.state_dict()[key].dtype
                or not torch.equal(original, restored.state_dict()[key])
                for key, original in field.state_dict().items()):
            raise ValueError("migration state verification failed; original preserved")
        result = dict(result, migration_state_exact=True, original_preserved=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
