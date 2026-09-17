"""Verify completed source controls and the rejected semantic admission evaluation."""
import json
from pathlib import Path
import torch
from radio_gs.utils.immutable_artifacts import sha256_file
from radio_gs.models.element_retrieval_field import ElementRetrievalField
result=dict(schema='scope_fusion_controls.v1',scenes={},promoted=False,retained='counterfactual_extent_benefit_admission')
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene;field=root/'field';out=root/'semantic_extent_benefit_admission_validation';candidate=field/'semantic_extent_benefit_admission';report=json.loads((candidate/'report.json').read_text());assert sha256_file(candidate/'field.pt')==report['field_sha256'];parent=torch.load(field/'joint_supervised_image_residual/field.pt',weights_only=False);new=torch.load(candidate/'field.pt',weights_only=False);assert sha256_file(field/'joint_supervised_image_residual/field.pt')==report['parent_field_sha256']
 model=ElementRetrievalField(**new['architecture']);model.load_state_dict(new['state_dict']);assert all(torch.equal(v,new['state_dict'][k]) for k,v in parent['state_dict'].items());assert set(new['state_dict'])-set(parent['state_dict'])=={'extent_benefit_admission.center','extent_benefit_admission.scale','extent_benefit_admission.coefficient'}
 execution=json.loads((out/'execution.json').read_text());assert execution['finished'] and execution['image_prompt_observations_exact']
 for stage in execution['stages']:assert stage['status']=='complete' and stage['returncode']==0 and sha256_file(out/(stage['name']+'.json'))==stage['output_sha256']
 cold=json.loads((out/'cold.json').read_text());assert cold['field_sha256']==report['field_sha256'] and cold['training_cache_reads_forbidden'] and cold['native_queries_encoded_online'];assert cold['modalities']['text']['concept_route_consistent'] and cold['modalities']['text']['posterior_max_difference']<.01 and cold['modalities']['prompt']['posterior_max_difference']==0
 previous_cold=json.loads((root/'counterfactual_extent_benefit_admission_validation/cold.json').read_text());assert cold['modalities']['image']==previous_cold['modalities']['image']
 sources=[Path('output/pixel_evidence_fusion_20260917')/scene/'caption_augmented.json',root/'robust_fusion_diagnosis/report.json',field/'source_extent_utility_tree/report.json',candidate/'report.json',out/'text.json',out/'reference.json',out/'execution.json',out/'cold.json']
 result['scenes'][scene]=dict(parent_tensors_exact=True,strict_cold_state_load=True,image_prompt_observations_exact=True,source_selected_penalty=report['selected']['penalty'],source_acceptance_passed=report['source_acceptance_passed'],rejected_for_shared_promotion=True,artifacts={str(p):sha256_file(p) for p in sources},cold=cold)
ceiling=Path('paper/artifacts/native_extent_selection_ceiling_20260917.json');result['selection_ceiling_sha256']=sha256_file(ceiling)
files=['radio_gs/models/extent_benefit_gate.py','radio_gs/models/element_retrieval_field.py','radio_gs/training/extent_utility_tree.py','radio_gs/training/robust_source_fusion.py','scripts/audits/prepare_pixel_evidence_reconstruction.py','scripts/audits/fit_semantic_extent_admission.py','scripts/audits/diagnose_native_extent_selection_ceiling.py',__file__];result['current_implementations']={p:sha256_file(p) for p in files}
result['implementation_note']='Fit snapshots precede the optional routing=None argument-validation fix. Current runtime also passes the new invalid-config regression test; source snapshots are preserved.'
Path('paper/artifacts/scope_fusion_controls_20260917.json').write_text(json.dumps(result,indent=2)+'\n');print('verified both scenes; retain previous gate')
