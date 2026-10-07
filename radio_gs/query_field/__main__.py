"""Bounded field construction and fitting, separate from benchmark evaluation."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["compile","fit","evaluate","audit","smoke","diagnose"])
    parser.add_argument("--config",default="configs/query_field/figurines_pilot.json")
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--threads",default=4,type=int)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    config=json.loads(Path(args.config).read_text())
    if args.stage=="compile":
        from .observations import compile_observations
        report=compile_observations(args.config,args.output/"observations",args.device)
        print(json.dumps({k:report[k] for k in ["elements","observed_elements","splits"]},indent=2))
    elif args.stage=="fit":
        from .train import fit
        report=fit(args.output/"observations",args.output/"models",config["training"],args.device)
        print(json.dumps(report,indent=2))
    elif args.stage=="evaluate":
        from .evaluate import evaluate
        print(json.dumps(evaluate(args.output,args.device,config.get('evaluation_stages',('untrained','trained'))),indent=2))
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
