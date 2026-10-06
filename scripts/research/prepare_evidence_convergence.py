"""Declare complete matched conditions, with no stored query answers as inputs."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--condition',choices=['local','partial','ownership'],required=True)
    args=parser.parse_args();root=Path('output/evidence_convergence_20261005')
    recipe=json.loads(Path('output/surface_membership_study/learned_recipe.json').read_text())
    for job in recipe['jobs']:
        source=Path(job['scene']);spec=json.loads(source.read_text())
        destination=root/args.condition/'maps'/job['benchmark']/job['name'];destination.mkdir(parents=True,exist_ok=True)
        if job['benchmark']=='uqis':
            for name in spec['map_files']:
                if not (destination/name).exists():os.link(source.parent/name,destination/name)
            spec['conditional_membership']['checkpoint']=str((root/'conditional_query_elements.joblib').resolve())
            spec['surface_consistency']['score_rule']='unary'
            spec['scope_exclusion_policy']='covisible_completeness' if args.condition=='partial' else 'visible_complement'
        else:
            spec['instance_evidence_reliability']='positive_local' if args.condition=='ownership' else 'local'
            spec['instance_evidence_mass']='raw' if args.condition=='local' else 'matched'
        (destination/source.name).write_text(json.dumps(spec,indent=2)+'\n');job['scene']=str(destination/source.name)
    recipe.update(name='Evidence reliability and explicit set convergence: '+args.condition,default=False,
        validation=dict(status='frozen before execution',independent_final_test=False,
            baseline_public='configs/pipeline/system_v8.json',baseline_uqis='output/surface_membership_study/learned_recipe.json',
            source_reliability=('pixel-owning detector confidence; original view-level complement' if args.condition=='ownership' else
                'pixel-owning instance confidence; least reliable instance bounds complement strength'),
            point_calibration='same four independent rooms, complete deployment domain; no retraining',
            uqis_ranking='unary ranking and independent MAP boolean set',
            partial_scope_exclusions=args.condition=='partial',
            source_mass_matching=args.condition!='local',
            graph_unchanged=True,native_coordinates_unchanged=True,source_inputs_unchanged=True))
    path=root/(args.condition+'_recipe.json');path.write_text(json.dumps(recipe,indent=2)+'\n')
    print(path)


if __name__=='__main__':main()
