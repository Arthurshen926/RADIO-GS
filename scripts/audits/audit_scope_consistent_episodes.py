"""Verify conservative scope admission against sealed source supervision."""
import json
from pathlib import Path
import torch
from radio_gs.training.reliable_correspondence import reliable_correspondence_tasks
from radio_gs.training.retrieval_supervision import validate_episode_split
from radio_gs.utils.immutable_artifacts import sha256_file


def graph_conflicts(data, split):
    views = data['proposal_views']
    parent = list(range(len(views)))
    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    edges = {tuple(sorted((e['query_region'], e['target_region']))) for e in data['episodes'] if e['split']==split and e['modality']=='image'}
    for a, b in sorted(edges):
        parent[find(a)] = find(b)
    groups = {}
    for i in range(len(parent)):
        groups.setdefault(find(i), []).append(i)
    collisions = []
    for group in groups.values():
        for a in group:
            for b in group:
                if b >= a or views[a] != views[b]:
                    continue
                left, right = data['masks'][a], data['masks'][b]
                iou = float((left & right).sum() / (left | right).sum().clamp_min(1))
                collisions.append(dict(regions=[b,a], view=int(views[a]), mask_iou=iou))
    return dict(edges=len(edges), groups=sum(len(g)>1 for g in groups.values()),
                same_view_pairs=len(collisions), nonduplicate_pairs=sum(r['mask_iou']<.9 for r in collisions),
                disjoint_pairs=sum(r['mask_iou']==0 for r in collisions), collisions=collisions)


def main():
    torch.set_num_threads(4)
    base = Path('/mnt/pool/sqy/results/RADIO-GS/output')
    output = Path('paper/artifacts/source_scope_consistency_audit_20260916.json')
    if output.exists():
        raise FileExistsError(output)
    results = {}
    for scene in ['figurines','ramen']:
        oldroot = base/'preserved_source_inventory_20260916'/scene
        newroot = base/'scope_consistent_episodes_20260916'/scene
        old, new = [torch.load(root/'episodes.pt', weights_only=False) for root in [oldroot,newroot]]
        for root in [oldroot,newroot]:
            assert sha256_file(root/'episodes.pt') == json.loads((root/'episodes.json').read_text())['sha256']
        info = json.loads((newroot/'teachers.json').read_text())
        assert old['geometry_binding'] == new['geometry_binding']
        assert old['teacher_manifest_sha256'] == new['teacher_manifest_sha256']
        for key in ['masks','proposal_views','seen']:
            assert torch.equal(old[key],new[key]), key
        assert old['queries'].keys() == new['queries'].keys()
        for key, value in old['queries'].items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value,new['queries'][key],atol=0,rtol=0)
        transport_error = float((old['region_probability']-new['region_probability']).abs().max())
        assert transport_error <= 2**-11
        validate_episode_split(new['episodes'], new['proposal_views'], info['frames'],info['train_frames'],info['validation_frames'])
        comparisons = {}
        for split in ['train','validation']:
            before, after = graph_conflicts(old,split),graph_conflicts(new,split)
            assert after['nonduplicate_pairs']==0
            comparisons[split] = dict(before=before,after=after)
        indexed = [(i,e) for i,e in enumerate(old['episodes']) if e['split']=='train']
        _, strict = reliable_correspondence_tasks(indexed,old['region_probability'].float(),old['proposal_views'],old['seen'])
        # Actual retained model lineage did not enable the older strict admission.
        oldreport = json.loads((oldroot/'preserved96_protected/report.json').read_text())
        results[scene] = dict(comparisons=comparisons, strict_legacy_accepted_pairs=strict['accepted_pairs'],
            parent_training_used_strict_admission=oldreport['correspondence_admission'] is not None,
            source_regions_unchanged=True,query_tensors_unchanged=True,transport_max_error=transport_error,
            counts=new['counts'],correspondence_consistency=new['correspondence_consistency'],
            old_episode_sha256=sha256_file(oldroot/'episodes.pt'),new_episode_sha256=sha256_file(newroot/'episodes.pt'),
            parent_field_sha256=sha256_file(oldroot/'exact_render_query_threshold96/field.pt'))
        print(scene, 'old/new/strict edges',comparisons['train']['before']['edges'],comparisons['train']['after']['edges'],strict['accepted_pairs'],flush=True)
    result=dict(schema='scope_consistency_audit.v1',scenes=results,benchmark_data_used=False,
                physical_instance_identity_verified=False,trained_candidate=False,
                old_strict_admission_was_previously_rejected=True,
                implementation_sha256=sha256_file(__file__))
    output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
