"""Offline functional compilation; benchmark labels and queries are not inputs."""
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from .compiled_forest import compile_forest, CompiledForest
from .compile_paired_evidence import csr_state
from .contracts import Support
from .joint_observation import JointObservationEvidence, KIND
from .storage import load_field_versioned, sha256, field_identity, deployment_bytes
from .research_storage import research_writer, export_receipt


def observation_state(membership, visibility, frames, descriptors):
    """Collapse linear native response into row moments, retaining its norm."""
    b, v = membership.tocsr().astype(np.float32), visibility.tocsr().astype(np.float32)
    frames = np.asarray(frames, np.int64)
    if b.shape[0] != v.shape[0] or len(frames) != b.shape[1] or len(descriptors) != b.shape[1]:
        raise ValueError('source observation arrays differ')
    blocks = []
    for view in range(v.shape[1]):
        ids = np.flatnonzero(frames == view); block = b[:,ids]
        if (block - sparse.diags(v[:,view].toarray().ravel()) @ block).nnz:
            raise ValueError('source membership outside declared visibility')
        mass = np.asarray(block.sum(1)).ravel()
        blocks.append(sparse.diags(1/np.maximum(mass,1)) @ block)
    if not np.array_equal(frames,np.sort(frames)): raise ValueError('source regions must be grouped by view')
    association = sparse.hstack(blocks,format='csr')
    mass = np.asarray(association.sum(1)).ravel()
    transfer = (sparse.diags(1/np.maximum(mass,1)) @ association).tocsr()
    native = np.array(descriptors,np.float32,copy=True)
    native /= np.maximum(np.linalg.norm(native,axis=1,keepdims=True),1e-8)
    moment = np.asarray(transfer @ native,dtype=np.float32)
    state = dict(views=torch.from_numpy(frames), number_of_views=torch.tensor(v.shape[1]),
        native_moment=torch.from_numpy(moment), native_supported=torch.from_numpy(mass>0),
        visible_fraction=torch.from_numpy(np.asarray(v.sum(1)).ravel()/v.shape[1]))
    for prefix, matrix in [('membership',b),('visibility',v),('association',association)]:
        state.update(csr_state(matrix,prefix))
    return state, transfer


@torch.no_grad()
def compile_scene(job, output, budgets, checkpoint, excluded_frames, *, store_root=None, legacy_v1=False):
    """Each budget yields two independent exports differing only in the head."""
    save=research_writer(store_root=store_root,legacy_v1=legacy_v1)
    import joblib
    from radio_gs.querying.direct_anchor_membership import DirectAnchorEvidence, FEATURE_NAMES
    from radio_gs.querying.anchor_observation_evidence import anchor_membership
    out = Path(output); out.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    field, parent = load_field_versioned(job['protected_parent'],store_root=store_root)
    if field.evidence is not None: raise ValueError('reference requires the protected prior without relation payload')
    before = {k: v.detach().clone() for k,v in field.state_dict().items()}
    observation = json.loads((Path(job['observations'])/'observations.json').read_text())
    geometry = torch.load(observation['geometry']['path'],map_location='cpu',weights_only=True)
    if sha256(observation['geometry']['path']) != observation['geometry']['sha256']:
        raise ValueError('geometry receipt changed')
    ids = geometry['source_vertex_ids'].numpy()
    directory = Path(job['legacy_map']).parent; legacy=json.loads(Path(job['legacy_map']).read_text())
    xyz=np.load(directory/'mesh_xyz.npy')
    if len(np.unique(ids)) != len(ids) or not np.array_equal(field.xyz.numpy(),xyz[ids]):
        raise ValueError('explicit source-to-Gaussian carrier correspondence failed')
    if not legacy['query_independent'] or legacy['labels_used'] or legacy['query_manifest_read']:
        raise ValueError('legacy source observations are not query/label independent')
    source_frames=legacy['source_frames']; offsets=np.load(directory/'region_offsets.npy')
    membership=sparse.load_npz(directory/'membership.npz'); visibility=sparse.load_npz(directory/'visibility.npz')
    descriptors=np.load(directory/'descriptors.npy')
    saved=joblib.load(checkpoint)
    if saved['feature_names'] != FEATURE_NAMES: raise ValueError('member feature contract changed')
    if job['scene'] in set(saved['fit']+saved['calibration']): raise ValueError('benchmark scene in member supervision')
    forest=compile_forest(saved['estimator'],saved['threshold'])
    source_receipts=[dict(path=str(directory/p),sha256=sha256(directory/p)) for p in
        ['membership.npz','visibility.npz','descriptors.npy','region_offsets.npy','mesh_xyz.npy']]
    reports=[]
    for budget, frames in budgets.items():
        if set(frames)&set(excluded_frames) or set(frames)-set(source_frames):
            raise ValueError('illegal source budget')
        columns=np.array([i for i,f in enumerate(source_frames) if f in set(frames)],np.int64)
        regions=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in columns])
        groups=np.repeat(np.arange(len(columns)),[offsets[i+1]-offsets[i] for i in columns])
        b=membership[:,regions].tocsr(); v=visibility[:,columns].tocsr(); d=descriptors[regions]
        state, transfer=observation_state(b,v,groups,d)
        # The exact source vertex map selects rows, never interpolates relations.
        for prefix in ['membership','visibility','association']:
            matrix={'membership':b,'visibility':v}.get(prefix)
            if matrix is None:
                matrix=sparse.csr_matrix((state[prefix+'_value'].numpy(),state[prefix+'_index'].numpy(),
                    state[prefix+'_pointer'].numpy()),shape=b.shape)
            state.update(csr_state(matrix[ids],prefix))
        for key in ['native_moment','native_supported','visible_fraction']: state[key]=state[key][ids].clone()
        relation=JointObservationEvidence(state,len(ids))
        reference=DirectAnchorEvidence(b,v,groups,xyz,d,transfer)
        anchors=np.unique(np.linspace(0,len(ids)-1,16).round().astype(int))
        rows=np.unique(np.r_[np.linspace(0,len(ids)-1,512).round().astype(int),anchors])
        source_replay=[]; tree=CompiledForest(forest)
        for anchor in anchors:
            support=Support(torch.tensor([int(anchor)]),torch.ones(1))
            actual=relation.features(field,support,torch.from_numpy(rows))
            expected=reference.encode(int(ids[anchor]),ids[rows])
            native,_=anchor_membership(b,v,groups,int(ids[anchor]))
            error=float(np.abs(actual.numpy()-expected).max())
            source_error=float(np.abs(actual[:,0].numpy()-native[ids[rows]]).max())
            exact_tree=saved['estimator'].predict_proba(expected)[:,1].astype(np.float32)
            same_features=tree.probability(torch.from_numpy(expected)).numpy()
            deployed=tree.probability(actual).numpy()
            operator_error=float(np.abs(same_features-exact_tree).max())
            disagreements=int(((deployed>saved['threshold'])!=(exact_tree>saved['threshold'])).sum())
            if max(error,source_error)>1e-5 or operator_error>1e-7 or disagreements:
                raise ValueError(f'functional compilation failed: {error}, {source_error}, {operator_error}, {disagreements}')
            source_replay.append(dict(anchor=int(anchor),source_anchor=int(ids[anchor]),rows=len(rows),
                feature_error=error, source_only_error=source_error, forest_operator_error=operator_error,
                forest_function_error=float(np.abs(deployed-exact_tree).max()),selection_disagreements=disagreements))
        for decision in ['source','supervised']:
            arm=budget+'_'+decision
            field.evidence=relation; field.evidence_kind=KIND
            relation.forest=CompiledForest(forest) if decision=='supervised' else None
            exported=save(field,out/arm/'field',dict(
                parent_field_sha256=field_identity(parent),
                source_map_sha256=sha256(job['legacy_map']), source_frames=frames,
                development_frames_excluded=excluded_frames, evidence_function='exact binary joint observation',
                membership_decision='compiled independent-room forest' if decision=='supervised' else 'visible conditional co-membership',
                additional_3d_labels=decision=='supervised', training_steps=0,
                member_checkpoint_sha256=sha256(checkpoint) if decision=='supervised' else None,
                supervision_rooms=saved['fit'] if decision=='supervised' else [],
                calibration_rooms=saved['calibration'] if decision=='supervised' else [],
                legacy_member_threshold=saved['threshold'] if decision=='supervised' else .5,
                scores='odds recentered at inherited member threshold; not calibrated probability',
                native_query_channels_unchanged=True, native_local_patch_function_preserved=False,
                source_query_inputs_used=False, uncompressed_functional_reference=True))
            if any(not torch.equal(field.state_dict()[k],value) for k,value in before.items()):
                raise ValueError('compiler rewrote parent native/geometry state')
            reports.append(dict(arm=arm,frames=frames,regions=len(regions),source_replay=source_replay,
                tensor_bytes=exported['tensor_bytes'],export=export_receipt(exported),
                field_schema=exported['schema'],deployment_bytes=deployment_bytes(exported),
                parent_state_unchanged=True,complete_carrier_rows=len(ids),extra_supervision=decision=='supervised'))
            print('exported joint function',job['scene'],arm,len(frames),len(regions),flush=True)
        field.evidence=None; field.evidence_kind='none'
    report=dict(scene=job['scene'],records=reports,source_receipts=source_receipts,
        parent_manifest=parent,supervision=dict(condition=saved['condition'],fit=saved['fit'],calibration=saved['calibration']),
        source_vertex_mapping_exact=True,benchmark_labels_read=False,benchmark_queries_read=False,
        runtime_source_access=False,seconds=time.monotonic()-started)
    (out/'construction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
