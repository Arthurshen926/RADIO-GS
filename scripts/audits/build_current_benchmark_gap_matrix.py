"""Explicit same-version coverage; never fill missing cells with legacy branches."""
import argparse
import json
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('paper/artifacts/current_benchmark_gap_matrix_20260918.json'));args=parser.parse_args()
contract_path=Path('paper/artifacts/joint_retrieval_contracts_20260915.json')
contracts=json.loads(contract_path.read_text())
rows=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 root=Path('output/v2_scope_rebuild_20260918')/scene if scene in ['figurines','ramen'] else Path('output/lerf_complete_v2_20260918')/scene
 for protocol in ['lerf_polygon_2d_development','lerf_public_3d','lerf_derived_reference']:
  task='text' if protocol=='lerf_polygon_2d_development' else 'reference'
  result=root/(task+'_online.json')
  complete=protocol!='lerf_public_3d' and result.exists()
  rows.append(dict(benchmark=protocol,scene=scene,method='v2_scope_96d',status='measured' if complete else 'not_run',
                   artifact=str(result) if complete else None,sha256=sha256_file(result) if complete else None,
                   missing=None if complete else 'Selected-only output implemented; released JPEG GT equivalence and same-contract score remain unverified' if protocol=='lerf_public_3d' else 'Current 96D reconstruction and protocol evaluation'))
for scene in contracts['scannet']['scenes']:
 for classes in ['19','15','10']:
  rows.append(dict(benchmark='scannet_ovs_'+classes,scene=scene['scene'],method='v2_scope_96d',status='not_run',
                   class_ids=contracts['scannet']['class_splits'][classes],artifact=None,missing='Current geometry/authority, 96D reconstruction, categorical point transfer and evaluation'))
for scene in ['fern','flower','fortress','horns_center','horns_left','leaves','orchids','trex']:
 rows.append(dict(benchmark='nvos_reference_signed_strict_unseen',scene=scene,method='v2_scope_96d',status='not_run',artifact=None,
                  missing='Same complete reconstruction recipe; existing shared image-branch pilot is not this full method'))
diagnostics=[]
for scene in ['figurines','ramen','teatime','waldo_kitchen']:
 path=(Path('output/current_selected_lerf_20260918')/scene/'report.json' if scene in ['figurines','ramen'] else Path('output/lerf_complete_v2_20260918')/scene/'selected_output/report.json')
 if path.exists():
  data=json.loads(path.read_text());diagnostics.append(dict(scene=scene,protocol='selected_3d_elements_projected_to_local_polygon_masks',artifact=str(path),sha256=sha256_file(path),mean_iou=data['mean_iou'],counts_as_official_public_completion=False))
report=dict(selected_output_diagnostics=diagnostics,schema='current_benchmark_gap_matrix.v1',independent_final_test=False,joint_method_complete=False,
            protocol_reference_sha256=sha256_file(contract_path),rows=rows,
            measured_cells=sum(r['status']=='measured' for r in rows),expected_cells=len(rows),
            acceptance='Complete every declared cell under its output contract. Do not average modalities/cohorts or substitute old branch scores. Source-proxy perfection is not an execution prerequisite.',
            source_control='Training entrypoints exclude benchmark labels; query/label access is allowed only by each explicit evaluation contract.',
            implementation_sha256=sha256_file(__file__))
out=args.output;assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print('coverage',report['measured_cells'],'/',len(rows))
