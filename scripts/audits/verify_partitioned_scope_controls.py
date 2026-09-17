"""Verify fold isolation, relation constraints, full denominators and retained state."""
import json
from pathlib import Path
from itertools import combinations
from radio_gs.training.scope_correspondence import partition_source_scopes
from radio_gs.utils.immutable_artifacts import sha256_file
result=dict(schema='partitioned_scope_controls.v1',promoted=False,benchmark_data_used_for_training=False,retained='counterfactual_extent_benefit_admission',scenes={})
for scene in ['figurines','ramen']:
 root=Path('output/pixel_evidence_rebuild_20260917')/scene;field=root/'field';info=json.loads((field/'teachers.json').read_text());frames=info['train_frames'];pairs=json.loads((root/'scope_depth_scope_pairs.json').read_text());artifacts={};metrics={};baseline=None;cohort=None
 for name in ['partitioned_scope_validation_20260918','local_partitioned_scope_validation_20260918','bridged_scope_validation_20260918','local_scope_abstention_20260918']:
  raw=root/name/'report.json';path=root/name/'full_cohort_report.json';d=json.loads(path.read_text());assert d['original_report_sha256']==sha256_file(raw);assert d['source_only'] and not d['benchmark_data_used']
  for p,h in d['inputs'].items():assert sha256_file(p)==h
  for p,h in d['implementations'].items():assert sha256_file(root/name/'sources'/Path(p).name)==h
  keys={(r['frame'],r['group']) for r in d['rows']};assert len(keys)==len(d['rows'])
  if cohort is None:cohort=keys
  else:assert keys==cohort
  values={(r['frame'],r['group']):r['merged']['iou'] for r in d['rows']}
  if baseline is None:baseline=values
  else:assert all(abs(values[k]-baseline[k])<1e-7 for k in keys)
  for plan in d['plans']:
   validation=set(frames[plan['fold']::4]);training=set(plan['training_frames']);assert not validation&training and validation==set(plan['validation_frames']);filtered=[r for r in pairs['scope_pairs'] if r['group']==plan['group'] and r['source_frame'] in training and r['target_frame'] in training];bridged=name=='bridged_scope_validation_20260918';recomputed=partition_source_scopes(plan['group'],training,filtered,allow_unknown_bridges=bridged);assert recomputed['clusters']==plan['clusters'] and recomputed['relations']==plan['relations']
   relation={(r['source'],r['target']):r['relation'] for r in plan['relations']}
   for cluster in plan['clusters']:
    for a,b in combinations(cluster,2):assert relation[min(a,b),max(a,b)] in (('support','unknown') if bridged else ('support',))
  metrics[name]=dict(macro=d['macro'],observations=len(d['rows']),zero_support_observations_added=d['zero_support_observations_added']);artifacts[str(path)]=sha256_file(path)
 oraclepath=root/'scope_hypothesis_ceiling_20260918/report.json';oracle=json.loads(oraclepath.read_text());assert oracle['diagnostic_only'] and not oracle['deployable'] and oracle['validation_masks_used_for_oracle_selection'];assert {(r['frame'],r['group']) for r in oracle['rows']}==cohort;assert abs(oracle['macro']['baseline']-sum(baseline.values())/len(baseline))<1e-7;artifacts[str(oraclepath)]=sha256_file(oraclepath)
 trackpath=root/'tracked_scope_labels_20260918.json';track=json.loads(trackpath.read_text());assert not track['benchmark_data_used'] and not track['moge_depth_used'] and not track['gaussian_correspondence_used']
 for p,h in track['inputs'].items():assert sha256_file(p)==h
 artifacts[str(trackpath)]=sha256_file(trackpath);pos=track['positive_group_tracks'];track_summary={k:track[k] for k in ['multi_view_group_tracks','group_tracks_with_positive','conflicting_group_tracks','groups_with_positive_tracks','total_groups']};track_summary.update(unique_positive_track_ids=len({r['point_id'] for r in pos}),unique_conflicting_track_ids=len({r['point_id'] for r in pos if r['conflict']}))
 replaypath=root/'native_scope_dedup_replay_20260918/report.json';replay=json.loads(replaypath.read_text());assert replay['source_only'] and not replay['benchmark_data_used'];assert replay['native_scope_assignment_errors']['native_and_coarse']==0
 for p,h in replay['inputs'].items():assert sha256_file(p)==h
 for p,h in replay['implementations'].items():assert sha256_file(replaypath.parent/'sources'/Path(p).name)==h
 assert sha256_file(replaypath.parent/'replay.pt')==replay['replay_sha256'];artifacts[str(replaypath)]=sha256_file(replaypath)
 retained=field/'counterfactual_extent_benefit_admission';receipt=json.loads((retained/'report.json').read_text());assert sha256_file(retained/'field.pt')==receipt['field_sha256'];result['scenes'][scene]=dict(native_scope_replay={k:replay[k] for k in ['accepted_detections','legacy_regions','corrected_regions','native_scope_assignment_errors']},metrics=metrics,oracle=oracle['macro'],track_summary=track_summary,retained_sha256=receipt['field_sha256'],artifacts=artifacts)
files=['radio_gs/utils/native_scope_identity.py','scripts/audits/ground_source_caption_phrases.py','scripts/audits/replay_native_scope_dedup.py','radio_gs/training/scope_correspondence.py','radio_gs/training/scope_vote_abstention.py','radio_gs/training/scope_validation_cohort.py','scripts/audits/validate_partitioned_scope_fusion.py','scripts/audits/validate_local_scope_abstention.py','scripts/audits/complete_scope_validation_cohort.py','scripts/audits/audit_scope_hypothesis_ceiling.py','scripts/audits/audit_tracked_scope_labels.py',__file__];result['current_implementations']={p:sha256_file(p) for p in files};result['notes']='All started source controls completed. No hypothesis policy improves both scenes; no new full-field retraining or real-query evaluation was launched. Original conditional reports remain immutable; full-cohort reports restore zero-support misses.'
Path('paper/artifacts/partitioned_scope_controls_20260918.json').write_text(json.dumps(result,indent=2)+'\n');print('verified full cohorts, training-only scope decisions, candidate ceilings, track evidence and unchanged retained models')
