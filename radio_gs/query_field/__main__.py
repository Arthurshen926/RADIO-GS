"""Bounded field construction and fitting, separate from benchmark evaluation."""
import argparse
import json
from pathlib import Path

import torch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["compile","fit","evaluate","audit","smoke","diagnose"])
    parser.add_argument("--config",default="configs/query_field/figurines_pilot.json")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",default=4,type=int)
    parser.add_argument("--store-root", type=Path, help="Explicit shared v2 blob store; required for new fits")
    parser.add_argument("--legacy-v1", action="store_true", help="Explicit historical reconstruction using full v1 packages")
    parser.add_argument("--max-new-bytes", type=int, help="Per-save allocation budget for v2")
    parser.add_argument("--reserve-bytes", type=int, default=1024**3)
    args=parser.parse_args(argv)
    if args.legacy_v1 and args.stage != "fit":
        parser.error("--legacy-v1 applies only to an explicit historical fit")
    if args.legacy_v1 and args.store_root is not None:
        parser.error("--legacy-v1 and --store-root are mutually exclusive")
    if args.stage == "fit" and not args.legacy_v1 and args.store_root is None:
        parser.error("new fits use shared v2: provide --store-root, or explicitly request --legacy-v1 for historical reconstruction")
    torch.set_num_threads(args.threads)
    config=json.loads(Path(args.config).read_text())
    if args.stage=="compile":
        from .observations import compile_observations
        report=compile_observations(args.config,args.output/"observations",args.device)
        print(json.dumps({k:report[k] for k in ["elements","observed_elements","splits"]},indent=2))
    elif args.stage=="fit":
        from .train import fit
        from functools import partial
        from .shared_storage import save_field_v2
        writer = None if args.legacy_v1 else partial(save_field_v2, store_root=args.store_root,
                    max_new_bytes=args.max_new_bytes, reserve_bytes=args.reserve_bytes)
        report=fit(args.output/"observations",args.output/"models",config["training"],args.device,
                   save_function=writer,legacy_v1=args.legacy_v1)
        print(json.dumps(report,indent=2))
    elif args.stage=="evaluate":
        from .evaluate import evaluate
        print(json.dumps(evaluate(args.output,args.device,config.get('evaluation_stages',('untrained','trained')),store_root=args.store_root),indent=2))
    elif args.stage=="smoke":
        from .smoke import smoke
        print(json.dumps(smoke(args.output,args.device),indent=2))
    elif args.stage=="diagnose":
        from .diagnose import diagnose
        print(json.dumps(diagnose(args.output,args.device),indent=2))
    else:
        from .baseline import audit
        print(json.dumps(audit(args.output),indent=2))


if __name__=="__main__":
    main()
