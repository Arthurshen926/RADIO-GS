"""Run RADIO-GS on explicit scene assets and runtime queries."""
import argparse
import json
from .runtime import PipelineConfig, build_plan, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--queries', required=True, help='Single query JSON, {queries: [...]}, or public query package directory for registered observation maps')
    parser.add_argument('--access', choices=['map-only', 'source-assisted'], default='source-assisted')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--native-python', default=PipelineConfig.native_python)
    parser.add_argument('--sam-python', default=PipelineConfig.sam_python)
    parser.add_argument('--checkpoint', default=PipelineConfig.checkpoint)
    parser.add_argument('--native-response', help='Explicit native query-response replay; never a fitted answer')
    parser.add_argument('--source-evidence', help='Explicit source detector observations to replay')
    parser.add_argument('--reference-tracks', help='Explicit source tracks for one reference query')
    parser.add_argument('--reference-evidence', help='Explicit reference-identified masks and unknown pixels; shared response and selection input')
    parser.add_argument('--provider-admitted', action='store_true', help='Source evidence was explicitly admitted upstream; bypass native identity admission')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--scope-transfer', action='store_true', help='Experimental one-hop physical scope correspondence on registered maps')
    parser.add_argument('--instance-scope', action='store_true', help='Experimental native region identity followed by physical instance scope')
    parser.add_argument('--member-chunk-size',type=int,help='Temporary member feature rows, without reducing output elements')
    parser.add_argument('--reference-completion',help='Explicit N14 signed-reference and target-RGB profile; source-only default unchanged')
    args = parser.parse_args()
    plan_only = args.plan
    del args.plan
    args.identity_admission = not args.provider_admitted
    del args.provider_admitted
    config = PipelineConfig(**vars(args))
    result = build_plan(config) if plan_only else run(config)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
