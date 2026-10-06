"""Freeze a reviewable method and code snapshot; no benchmark result promotion."""
import json,shutil,subprocess
from pathlib import Path
from radio_gs.utils.immutable_artifacts import sha256_file


def main():
 root=Path('output/frozen_cross_benchmark_method_20260918');root.mkdir(parents=True,exist_ok=False)
 files=sorted(set(Path('radio_gs').rglob('*.py'))|set(Path('scripts/audits').glob('*.py')))
 sources={}
 for p in files:
  target=root/'source'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target);sources[str(p)]=sha256_file(target)
 source_manifest=root/'source_manifest.json';source_manifest.write_text(json.dumps(sources,indent=2)+'\n')
 assets={}
 for scene in ['figurines','ramen']:
  original=Path('output/depth_opacity_pipeline_20260918')/scene/'reconstruction/field/bounded_fallback_admission'
  target=root/'checkpoints'/scene;target.mkdir(parents=True)
  for name in ['field.pt','report.json']:
   shutil.copy2(original/name,target/name);assets[str(target/name)]=sha256_file(target/name)
  for p in [Path('output/depth_opacity_repair_20260918')/scene/'geometry_report.json',Path('output/depth_opacity_repair_20260918')/scene/'geometry.pt']:
   assets[str(p)]=sha256_file(p)
 protocols={str(p):sha256_file(p) for p in map(Path,['paper/artifacts/evaluation_protocol_freeze_20260801.yaml','paper/artifacts/evaluation_protocol_registry_20260731.yaml','paper/artifacts/promptable_nvs_protocol_registry.yaml'])}
 report=dict(schema='frozen_cross_benchmark_method.v1',method_id='native96-depth-opacity-bounded-fallback-20260918',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),working_tree_snapshot_authoritative=True,source_manifest=str(source_manifest),source_manifest_sha256=sha256_file(source_manifest),assets=assets,evaluation_protocols=protocols,benchmarks=['LERF2D','LERF3D','ScanNet OVS','NVOS','UQIS'],query_types=['text','image','2d_prompt','3d_prompt'],recipe=dict(field='single96/native64+structure32',geometry='source-depth calibration and source-only opacity preprocessing, then frozen',teachers=['SigLIP','DINO','RADIO','SAM'],native_readout='shared teacher decoder/native query matching',fallback='finite signed native logit correction, source concept-withholding grid [0,.5,1,2,4]',admission='source-fitted with withheld-fallback counterfactuals'),positive_only_control_excluded=True,adaptation_policy='Only dataset I/O, geometry input tracks and frozen evaluator adapters; method changes require new identity. Do not stitch scene-best controls.',independent_test_claim=False,completed_checkpoint_scenes=['figurines','ramen'],other_benchmarks_completed=False,uqis_contract_status='pending readable existing manifest and frozen evaluator; do not invent substitute',input_tracks=['provided_geometry/RGB-D where authorized by contract','RGB+MoGe with any extra calibration geometry explicitly declared'])
 p=Path('paper/artifacts/frozen_cross_benchmark_method_20260918.json');assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(p,sha256_file(p),len(files),'source files')

if __name__=='__main__':main()
