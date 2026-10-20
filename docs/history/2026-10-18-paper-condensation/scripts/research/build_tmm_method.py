"""Bind the condensed method to code and retained evidence; no new inference.

This is a claim ledger, not an assertion that every adapter already implements
one successful admission algorithm. It keeps positive controls, provider gains,
functional compatibility, diagnostic results and unverified transfer separate.
"""
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = REPO / 'paper/tmm_response'
STUDY = REPO / 'output/tmm_response_20261016/study'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def code_binding(path, symbol, role, status):
    nodes = ast.parse((REPO / path).read_text()).body
    node = None
    for part in symbol.split('.'):
        node = next(n for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
        nodes = node.body
    return dict(path=path, symbol=symbol, line=node.lineno, sha256=sha(REPO / path),
                role=role, status=status)


def make_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(11.8, 3.4))
    ax.set_xlim(0, 15.8); ax.set_ylim(-.15, 4.3); ax.axis('off')

    def box(x, y, width, height, title, detail, color):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.08',
            linewidth=1, edgecolor='#4A5568', facecolor=color))
        ax.text(x+width/2, y+height*.71, title, ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(x+width/2, y+height*.32, detail, ha='center', va='center', fontsize=8)

    def arrow(a, b, **extra):
        ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->', color='#4A5568', lw=1.2, **extra))

    box(.15, 1.8, 2.65, 1.55, 'Native teacher evidence',
        'Query + native response\nAvailable observations', '#EDF2F7')
    box(3.5, 1.8, 3.2, 1.55, 'Query applicability',
        'Does this observation\nrefer to the requested target?', '#D9EDE9')
    box(7.25, 1.8, 3.2, 1.55, 'Spatial authority',
        'Support / exclude / unknown\nReliability and visible domain', '#D9EDE9')
    box(11.15, 1.8, 2.35, 1.55, 'Geometric fusion',
        'Admitted responses\nRetain unsupported prior', '#E0EBF7')
    ax.text(14.8, 2.57, 'Fixed task\nreadout', ha='center', va='center', fontsize=9)
    for a,b in [((2.85,2.57),(3.4,2.57)), ((6.77,2.57),(7.15,2.57)),
                ((10.52,2.57),(11.05,2.57)), ((13.57,2.57),(14.2,2.57))]: arrow(a,b)
    ax.plot([3.5,3.5,13.5,13.5],[3.55,3.75,3.75,3.55], color='#367D70', lw=1)
    ax.text(8.5, 4.02, 'Query-conditioned evidence-to-response method (our v10 implementation)',
            fontsize=10, ha='center', color='#235D53')
    box(11.15, .05, 2.35, 1.25, 'Complete extent',
        'Silhouette selection /\nconditional membership', '#E0EBF7')
    ax.plot([8.85,8.85],[1.72,.68], color='#4A5568', lw=1)
    arrow((8.85,.68),(11.05,.68))
    arrow((13.57,.68),(14.8,.68))
    arrow((14.8,.68),(14.8,2.02))
    ax.text(5.9,.68,'Typed realizations; shared evidence contract',ha='center',fontsize=8)
    fig.tight_layout(pad=.2)
    (PAPER / 'figures').mkdir(exist_ok=True)
    for suffix in ('pdf', 'png'):
        fig.savefig(PAPER / f'figures/method_overview.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def control_table(feedback):
    columns = [('lerf','semantic_iou'), ('lerf','ap'), ('nvos','iou'), ('scannet','miou_19'),
               ('uqis','text_ap'), ('uqis','image_ap'), ('uqis','point_2d_ap'), ('uqis','point_3d_ap')]
    maps = {name:{(r['family'],r['metric']):r['values_percent'] for r in d['rows']}
            for name,d in feedback.items()}
    reference = [maps['stage_reconstruction_v4'][key]['baseline'] for key in columns]
    for mapping in maps.values():
        assert all(abs(mapping[key]['baseline']-value) < 1e-8 for key,value in zip(columns,reference))
    variants = [
        ('post_response_v1','partial','Post: active partial'),
        ('post_response_v1','complement','Post: exterior negative'),
        ('post_response_v2','observable','Post: available domain'),
        ('post_response_v3','scope','Post: scope agreement'),
        ('post_response_v3','matched','Post: matched budget'),
        ('stage_reconstruction_v4','diagonal','Stage: diagonal'),
        ('stage_reconstruction_v4','rendered_squared','Stage: squared'),
        ('stage_reconstruction_v4','rendered_huber','Stage: Huber')]
    lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
        r'Condition & L-IoU & L-AP & N-IoU & S-mIoU19 & U-text & U-image & U-2D & U-3D \\',
        r'\midrule', 'Retained v10 & '+' & '.join(f'{x:.2f}' for x in reference)+r' \\']
    for name,arm,label in variants:
        values = [f'{maps[name][key][arm]:.2f}' +
            (r'$^\dagger$' if name=='stage_reconstruction_v4' and key[0]=='uqis' else '') for key in columns]
        lines.append(label+' & '+' & '.join(values)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'tables/condensed_controls.tex').write_text('\n'.join(lines)+'\n')


def build():
    evidence = read(PAPER / 'evidence.json')
    assert evidence['complete_jobs'] == 31 and not evidence['new_common_core_all_tasks_validated']
    for path, receipt in evidence['source_files'].items():
        assert sha(path) == receipt['sha256'], path
    effects = {e['name']: e for e in evidence['paired_effects']}
    for e in effects.values():
        assert abs(e['delta_pp'] - 100*(e['after'] - e['before'])) < 1e-10
    feedback = {}
    for name in ('post_response_v1', 'post_response_v2', 'post_response_v3', 'stage_reconstruction_v4'):
        item = read(PAPER / f'{name}_feedback.json')
        assert item['summary_sha256'] == sha(STUDY / name / 'summary.json')
        assert not item['default_changed'] and not item['independent_test']
        feedback[name] = item
    analysis = read(PAPER / 'stage_reconstruction_v4_analysis.json')
    assert analysis['summary_sha256'] == feedback['stage_reconstruction_v4']['summary_sha256']
    assert sum(v['squared_objective_reduced_jobs'] for v in analysis['families'].values()) == 22
    assert analysis['uqis_compatibility_only'] and not analysis['new_default_selected']
    specs = [
        ('radio_gs/querying/source_identity.py', 'native_identity_posterior', 'query-conditioned identity support', 'positive_local_control'),
        ('scripts/research/condition_source_identity.py', 'main', 'candidate rejection retained as unknown', 'positive_local_control'),
        ('radio_gs/querying/typed_evidence.py', 'ViewEvidence', 'positive/negative/visible observation contract', 'existing_contract'),
        ('radio_gs/querying/response_refinement.py', 'fuse_admitted_evidence', 'admitted visibility fusion', 'functional_compatibility'),
        ('radio_gs/querying/response_reconstruction.py', 'AdmittedResponseSystem', 'matched reconstruction operators', 'functional_compatibility_and_controls'),
        ('radio_gs/querying/reference_response.py', 'main', 'our v10 reference evidence and robust rendered response', 'implemented_in_house_method'),
        ('radio_gs/pipeline/subset.py', 'main', 'our v10 independent opacity selection with admitted-pixel silhouette losses and footprint cost', 'implemented_in_house_method'),
        ('radio_gs/querying/local_appearance_match.py', 'match_regions', 'our native patch-level image correspondence', 'positive_local_control'),
        ('radio_gs/querying/coverage_evidence.py', 'coverage_evidence', 'confidence-weighted source coverage for controlled NVOS study', 'candidate_not_promoted'),
        ('radio_gs/querying/image_conditioned_response.py', 'response_prompt', 'rendered query response as a frozen image-decoder prompt', 'view_conditioned_candidate_not_promoted'),
        ('radio_gs/querying/image_conditioned_response.py', 'lift_image_response', 'actual bounded Gaussian response writeback with unsupported-element preservation', 'view_conditioned_candidate_not_promoted'),
        ('radio_gs/querying/signed_reference_affinity.py', 'patch_authority', 'official signed scribble versus pseudo-mask reference authority', 'negative_control_not_promoted'),
        ('radio_gs/querying/signed_reference_affinity.py', 'signed_reference_probability', 'frozen descriptor margin with reference-only scale; incomplete instance transfer', 'negative_control_not_promoted'),
        ('radio_gs/querying/transport_residual.py', 'transfer_reference_delta', 'matched reference intervention around incumbent response', 'candidate_not_promoted'),
        ('radio_gs/querying/response_crop.py', 'response_crop_box', 'complete-response local target re-decoding window', 'stopped_iou_ap_tradeoff_control'),
        ('radio_gs/querying/response_crop.py', 'restore_response_crop', 'preserve every outside probability during local image replacement', 'stopped_iou_ap_tradeoff_control'),
        ('radio_gs/querying/multiview_response.py', 'fit_multiview_response', 'matched source-only and source-target response fusion', 'source_only_positive_control_joint_not_promoted'),
        ('radio_gs/querying/reference_admission.py', 'reference_choices', 'reference feasibility or predicted quality selects complete response', 'candidate_not_promoted'),
        ('radio_gs/querying/reference_completion.py', 'signed_scribble_points', 'original signed query coverage for reference completion', 'candidate_not_promoted'),
        ('radio_gs/querying/reference_completion.py', 'enforce_reference_scribbles', 'explicit reference annotations preserve unknown responses', 'candidate_not_promoted'),
        ('radio_gs/querying/instance_extent.py', 'scribble_fields', 'visible signed reference coverage; unmarked coverage stays unknown', 'candidate_not_promoted'),
        ('radio_gs/querying/instance_extent.py', 'admit_component_additions', 'instance-authorized additive region response', 'candidate_not_promoted'),
        ('radio_gs/querying/response_calibration.py', 'SourceReadoutHistogram.result', 'source-only response decision calibration; same rule for every query', 'candidate_not_promoted'),
        ('radio_gs/querying/local_response_credit.py', 'LocalResponseCredit', 'source Huber path credit for local response edits', 'candidate_not_promoted'),
        ('radio_gs/querying/local_response_credit.py', 'matched_uniform_edit', 'positive/negative edit-mass matched control', 'diagnostic_control'),
        ('radio_gs/querying/response_permissions.py', 'admit_response_observation', 'explicit directional permissions', 'diagnostic_candidate_not_promoted'),
        ('radio_gs/querying/local_element_fusion.py', 'compile_local_evidence', 'both native category priors and production source weights', 'preserved_provider'),
        ('radio_gs/querying/query_conditioned_membership.py', 'QueryMembershipEvidence.query_features', 'optional native evidence and complete-carrier context', 'optional_candidate_not_v10_default'),
        ('radio_gs/querying/query_conditioned_membership.py', 'QueryConditionedMembership.infer', 'optional query-conditioned membership; absent from all nine v10 maps', 'optional_candidate_not_v10_default'),
        ('radio_gs/querying/direct_anchor_membership.py', 'DirectAnchorMembership.infer', 'complete membership for actual point input', 'preserved_provider'),
        ('radio_gs/querying/direct_anchor_membership.py', 'DirectAnchorEvidence.encode', '15-dimensional complete-carrier anchor evidence', 'implemented_in_house_method'),
        ('radio_gs/querying/anchor_observation_evidence.py', 'joint_anchor_evidence', 'view-balanced nested-mask ambiguity and common visibility', 'implemented_in_house_method'),
        ('radio_gs/querying/direct_anchor_membership.py', 'DirectAnchorMembership.infer_region', 'retained image path reuses infer; estimated anchor is also forced positive', 'preserved_provider_with_unresolved_anchor_authority'),
        ('radio_gs/querying/estimated_anchor_authority.py', 'estimated_anchor_patch', 'isolate only the estimated-anchor positive override', 'causal_audit_not_promoted'),
        ('radio_gs/querying/query_witness_anchor.py', 'QueryWitnessAnchor.infer', 'other-view query witness center with matched within-view permutation', 'targeted_mechanism_not_promoted'),
        ('radio_gs/querying/complete_response_verification.py', 'CompleteResponseVerifier.verify', 'query verification among unmodified complete response candidates', 'targeted_mechanism_not_promoted'),
    ]
    replacement_path = PAPER/'response_replacement_feedback.json'
    replacement = read(replacement_path) if replacement_path.exists() else None
    if replacement is not None:
        for path, digest in replacement['evidence'].items():
            assert sha(path) == digest, path
        specs += [
            ('radio_gs/querying/response_replacement.py', 'RegionResponseEvidence.encode',
             'teacher-order-invariant complete-response evidence signatures', 'learned_candidate_not_promoted'),
            ('radio_gs/querying/response_replacement.py', 'relative_utilities',
             'antisymmetric learned actual-set utility differences', 'learned_candidate_not_promoted'),
            ('radio_gs/querying/response_replacement.py', 'calibrate_margin',
             'separate calibration of one empirical retain/replace guard', 'learned_candidate_not_promoted'),
            ('radio_gs/querying/response_replacement.py', 'choose_response',
             'whole-answer replacement with exact incumbent retention on abstention', 'learned_candidate_not_promoted'),
        ]
    value_path = PAPER/'response_value_feedback.json'
    if value_path.exists():
        specs += [('radio_gs/querying/response_value.py', 'set_features',
                   'candidate-set context without modality labels; order equivariance', 'controlled_architecture_candidate'),
                  ('radio_gs/querying/response_value.py', 'relative_utilities',
                   'consistent differences of a common complete-response value', 'controlled_architecture_candidate'),
                  ('radio_gs/querying/response_value.py', 'training_values',
                   'query-centered actual complete-answer utility targets', 'controlled_architecture_candidate')]
    transfer_path=PAPER/'multitask_transfer_feedback.json'
    if transfer_path.exists():
        specs += [('radio_gs/querying/category_response_transfer.py','complete_menu_candidates',
                   'original whole-menu providers with native and fusion output parity checks','verified_provider_preparation'),
                  ('radio_gs/querying/category_response_transfer.py','menu_signature',
                   '39-component signature averaged over the complete menu; explicit visibility-domain shift','transfer_candidate'),
                  ('radio_gs/querying/multitask_response_value.py','protected_output_rows',
                   'shared menu decision and one calibration margin protecting every room/output cell','joint_training_protocol'),
                  ('radio_gs/response_study/transfer_diagnosis.py','menu_capacity',
                   'whole-menu capacity and regret without submenu winner assembly','post_evaluation_diagnostic')]
    utility_path=PAPER/'utility_diagnosis_feedback.json'
    if utility_path.exists():
        specs += [('radio_gs/querying/centered_response_value.py','CenteredResponseValue',
                   'fixed relative-only context control; common-offset invariant inputs','controlled_architecture_candidate'),
                  ('radio_gs/response_study/utility_diagnosis.py','gain_audit',
                   'fixed-model available utility versus decision regret','post_fit_diagnostic')]
    bindings = [code_binding(*s) for s in specs]
    for binding in bindings:
        binding['implementation_ownership'] = 'ours; external teacher weights remain externally attributed'
    residual_path=PAPER/'residual_utility_feedback.json'
    if residual_path.exists():
        bindings += [code_binding(*s) for s in [
            ('radio_gs/querying/residual_response_value.py','out_of_room_prior',
             'room-grouped cross-fitting of the structural prior','controlled_architecture_candidate'),
            ('radio_gs/querying/residual_response_value.py','fit_residual_models',
             'query residual with matched no-null capacity and weight control','controlled_architecture_not_promoted'),
            ('radio_gs/response_study/residual_diagnosis.py','decision_attribution',
             'separate fixed-policy ranking regret from signed retention cost','post_fit_diagnostic')]]
    identity = effects['Query-conditioned identity admission']
    response = effects['Admitted source response fusion']
    claims = [
        dict(id='C1', claim='Query-conditioned candidate admission improves text selected-set projection.',
            status='supported_local_control', effect=identity,
            limit='LERF text candidates; does not establish the same rule across teachers or query types.'),
        dict(id='C2', claim='Using already admitted observations in the response recovers semantic accuracy.',
            status='supported_output_integration_control', effect=response,
            limit='Gain is not attributable solely to a new fusion equation and is not additive with C1.'),
        dict(id='C3', claim='Extracted cores preserve original solver threshold/category behavior.',
            status='functional_compatibility', source_jobs=22,
            numeric_checks=analysis['families'], uqis='unchanged complete-output compatibility',
            limit='Original solvers differ; no common new rule is proven by extraction.'),
        dict(id='C4', claim='Lower source reconstruction loss is insufficient for higher downstream accuracy.',
            status='supported_diagnostic_control', source_jobs_with_lower_loss=22,
            reference='stage_reconstruction_v4_feedback.json',
            limit='Does not uniquely identify identity, extent, geometry or view-consistency errors.'),
        dict(id='H1', claim='One transferable admission rule improves strong frozen providers across query types.',
            status='open', missing=['validated shared admission gains on map-only UQIS',
                'positive query-specific effect under matched controls', 'at least two appropriate teacher configurations',
                'untouched validation scenes after rule freeze']),
    ]
    claims += [dict(id='V10',claim='Our implemented v10 method is evaluated across all five outputs.',
                    status='implemented_and_scored',jobs=31,
                    limit='Typed implementations and differing input contracts are disclosed; not a new L11 gain'),
               dict(id='C5',claim='Native local correspondence improves image identity matching.',
                    status='supported_local_control',effect=effects['Native local correspondence']),
               dict(id='C6',claim='Complete-carrier membership recovers extent beyond regional readout.',
                    status='supported_local_control',effect=effects['Complete-carrier membership reuse'])]
    from radio_gs.response_study.registry import provider_audit, check_anchor_audit
    providers = provider_audit(REPO, read(REPO / 'configs/pipeline/system_v10.json'))
    anchor_checks = check_anchor_audit(STUDY)
    anchor = read(STUDY/'estimated_anchor_authority_v1/summary.json')
    (PAPER/'estimated_anchor_authority_feedback.json').write_text(json.dumps(anchor,indent=2)+'\n')
    diagnosis_path = STUDY/'estimated_anchor_authority_v1/anchor_diagnosis.json'
    diagnosis = read(diagnosis_path)
    correct = [r['unforced_probability'] for r in diagnosis['rows'] if r['anchor_correct']]
    incorrect = [r['unforced_probability'] for r in diagnosis['rows'] if not r['anchor_correct']]
    # Pairwise AUROC with half credit for ties; descriptive, no fitted threshold.
    auc = sum((p > n) + .5*(p == n) for p in correct for n in incorrect)/(len(correct)*len(incorrect))
    self_score = dict(queries=len(correct)+len(incorrect), identity_correctness_auroc=auc,
        correct_mean_self_probability=sum(correct)/len(correct),
        incorrect_mean_self_probability=sum(incorrect)/len(incorrect),
        diagnostic_only=True, threshold_fitted=False, independent_test=False,
        source=str(diagnosis_path), source_sha256=sha(diagnosis_path))
    (PAPER/'anchor_self_score_diagnosis.json').write_text(json.dumps(self_score,indent=2)+'\n')
    witness = read(PAPER/'query_witness_anchor_feedback.json')
    assert witness['complete'] and not witness['default_changed'] and not witness['all31_fresh_inference']
    assert witness['summary_sha256'] == sha(STUDY/'query_witness_anchor_v1/summary.json')
    response_verification = read(PAPER/'complete_response_verification_feedback.json')
    assert response_verification['complete'] and not response_verification['default_changed']
    assert response_verification['summary_sha256'] == sha(STUDY/'complete_response_verification_v1/summary.json')
    record = dict(schema='tmm.condensed_method.v1', method='Query-conditioned evidence fusion for 3D responses',
        method_implementation='our v10', historical_components_owned_by_method=True,
        mechanism=['query applicability', 'spatial evidence authority', 'geometric response fusion'],
        scope_fixed=True, mathematical_contract_defined=True, universal_admission_rule_finalized=False,
        single_shared_algorithm_all_five_validated=False, inference_default_changed=False,
        protocol='all-five feedback; declared source-assisted versus map-only inputs; no task winner stitching',
        code_bindings=bindings, claims=claims, effective_provider_audit=providers,
        estimated_anchor_audit=dict(verification=anchor_checks,
            effect_image_ap_pp=anchor['effects']['image_ap']['delta_pp'],
            effect_image_iou_pp=anchor['effects']['image_selected_iou']['delta_pp'],
            source=str(STUDY/'estimated_anchor_authority_v1/summary.json'),
            source_sha256=sha(STUDY/'estimated_anchor_authority_v1/summary.json'),
            self_score_identity_diagnosis=self_score,
            interpretation='removing the single-vertex override is insufficient; conditional member confidence is not query-identity confidence',
            conditional_anchor_strata_not_causal=True),
        frozen_provider_effects=[effects['Native local correspondence'],effects['Complete-carrier membership reuse']],
        query_witness_study=dict(source='query_witness_anchor_feedback.json',
            source_sha256=sha(PAPER/'query_witness_anchor_feedback.json'),
            effect_image_iou_pp=witness['effects']['query_witness_minus_baseline']['image_selected_iou']['delta_pp'],
            matched_control_effect=witness['effects']['query_witness_minus_permuted_witness']['image_selected_iou'],
            promoted=False,all31_fresh_inference=False,
            interpretation='weighted spatial recentering loses complete-response quality; query-specific benefit is not demonstrated'),
        complete_response_verification_study=dict(source='complete_response_verification_feedback.json',
            source_sha256=sha(PAPER/'complete_response_verification_feedback.json'),
            effect_image_iou_pp=response_verification['effects']['verified_minus_baseline']['image_selected_iou']['delta_pp'],
            finite_pool_decomposition=response_verification['finite_pool_decomposition'],
            promoted=False,all31_fresh_inference=False,oracle_is_not_a_method=True,
            interpretation='frozen proposals contain better complete answers, but rank-contrast selection does not preserve baseline accuracy'),
        retired_from_method_body=['active-only residual normalization','scope agreement as a default gate',
            'universal replacement inverse solver'],
        build_runs_benchmark_inference=False,
        verified_historical_source_files=len(evidence['source_files']),
        source_sha256={str(p.relative_to(REPO)):sha(p) for p in [PAPER/'evidence.json',
            PAPER/'stage_reconstruction_v4_analysis.json',PAPER/'estimated_anchor_authority_feedback.json',
            PAPER/'anchor_self_score_diagnosis.json',PAPER/'query_witness_anchor_feedback.json',
            PAPER/'complete_response_verification_feedback.json',PAPER/'sections/method.tex',PAPER/'sections/replacement_method.tex',
            PAPER/'main.tex',PAPER/'diagnostics.tex',PAPER/'sections/related_work.tex',
            Path(__file__).resolve(), *[PAPER/f'{name}_feedback.json' for name in feedback]]})
    if replacement is not None:
        record['active_algorithm_candidate'] = dict(name=replacement['method'],
            architecture='frozen complete responses -> observable signatures -> learned relative utility -> retain or replace',
            training_development_gate=replacement['training']['development_gate'],
            uqis_transfer=replacement['transfer']['uqis_macro'],
            source_task_transfer_pending=True, promoted=False,
            note='historical partial-evidence positive controls are not gains of this learned policy')
        for path in [replacement_path, PAPER/'replacement.tex', REPO/'configs/research/response_replacement.json',
                     REPO/'docs/adr/0023-learned-complete-response-replacement.md']:
            record['source_sha256'][str(path.relative_to(REPO))] = sha(path)
    joint_path = PAPER/'joint_replacement_feedback.json'
    if joint_path.exists():
        joint = read(joint_path)
        for path, digest in joint['evidence'].items():
            assert sha(path) == digest, path
        record['active_algorithm_candidate'] = dict(name=joint['method'],
            architecture='frozen complete responses -> observable signatures -> one jointly trained utility model -> one retain/replace margin',
            training_development_gate=joint['training']['development_gate'],
            uqis_transfer=joint['transfer']['uqis_macro'],
            exact_preserved_candidate_pools=joint['exact_candidate_pools_reused'],
            finite_pool_diagnosis=joint['finite_pool_diagnosis'],
            source_task_transfer_pending=True, promoted=False,
            note='whole-cohort transfer remains open; no modality stitching or independent final-test claim')
        for path in [joint_path, REPO/'configs/research/response_replacement_joint.json',
                     REPO/'radio_gs/querying/response_supervision.py',
                     REPO/'scripts/research/compile_tmm_image_supervision.py',
                     REPO/'scripts/research/train_tmm_joint_replacement.py']:
            record['source_sha256'][str(path.relative_to(REPO))] = sha(path)
    if value_path.exists():
        value = read(value_path)
        for path, digest in value['evidence'].items():
            assert sha(path) == digest, path
        record['active_algorithm_candidate'] = dict(name=value['method'],
            architecture='preserved complete candidate set -> query/response signatures -> set-conditioned utility potential -> shared retain/replace rule',
            training_development_gate=value['training']['development_gate'],
            uqis_transfer=value['transfer']['uqis_macro'], exact_preserved_candidate_pools=134,
            controlled_joint_pairwise_parent=value['controlled_joint_pairwise_parent'],
            source_task_transfer_pending=True, promoted=False,
            note='candidate-order and comparison consistency are algebraic properties, not proof of accuracy')
        for path in [value_path, REPO/'configs/research/response_value.json',
                     REPO/'radio_gs/querying/response_value.py', REPO/'scripts/research/train_tmm_response_value.py']:
            record['source_sha256'][str(path.relative_to(REPO))] = sha(path)
    cross_path = PAPER/'cross_task_feedback.json'
    if cross_path.exists():
        cross = read(cross_path)
        for path, digest in cross['source_hashes'].items():
            assert sha(path) == digest, path
        record['cross_task_boundary'] = dict(
            source_decisions=cross['source_audit']['decisions'],
            source_candidate_capacity=cross['historical_joint_metric_oracle']['macro'],
            independent_target_rooms=cross['independent_target_preparation']['rooms'],
            missing_calibration_classes=cross['independent_target_preparation']['missing_calibration_classes'],
            utility_training_ready=False, source_learner_validated=False,
            oracle_is_diagnostic=True, new_five_task_performance=False)
        for path in [cross_path, PAPER/'tables/cross_task_capacity.tex',
                     REPO/'configs/research/source_decision_audit.json',
                     REPO/'radio_gs/response_study/decision_evidence.py',
                     REPO/'scripts/research/audit_tmm_source_decisions.py',
                     REPO/'scripts/research/prepare_tmm_cross_task_targets.py',
                     REPO/'scripts/research/summarize_tmm_cross_task.py']:
            record['source_sha256'][str(path.relative_to(REPO))] = sha(path)
    if transfer_path.exists():
        transfer=read(transfer_path)
        for path,digest in transfer['receipts'].items():
            assert sha(REPO/path)==digest,path
        record['cross_task_progress']=dict(prepared=transfer.get('prepared'),
            source_rooms_complete=transfer['source_rooms_complete'],
            model_fitted=transfer['model_fitted'],fresh_joint_transfer_complete=transfer['fresh_joint_transfer_complete'],
            outputs=transfer['outputs'],limitations=transfer['limitations'],
            all_five_transfer_complete=False,promoted=False)
        if transfer['fresh_joint_transfer_complete']:
            record['active_algorithm_candidate']['latest_multitask_transfer']=transfer['outputs']
            record['active_algorithm_candidate']['latest_training_run']='response_value_multitask_v1'
        for path in [transfer_path,PAPER/'category_supervision_feedback.json',
                     REPO/'configs/research/response_value_multitask.json',
                     REPO/'scripts/research/prepare_tmm_category_transfer.py',
                     REPO/'scripts/research/evaluate_tmm_multitask_transfer.py',
                     REPO/'scripts/research/diagnose_tmm_multitask_transfer.py',
                     REPO/'scripts/research/summarize_tmm_multitask_transfer.py']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    if utility_path.exists():
        utility=read(utility_path)
        for path,digest in utility['receipts'].items():
            assert sha(REPO/path)==digest,path
        record['joint_utility_diagnosis']=dict(completed=utility['completed'],
            development={k:dict(gate=v['gate'],task_gates=v['task_gates'],margins=v['margins'])
                         for k,v in utility['development'].items()},
            benchmark_capacity=utility.get('benchmark_capacity'),
            target_signatures_inspected_before_control_design=True,
            calibration_and_development_previously_opened=True,
            all_five_mechanism_closed=False,promoted=False)
        for path in [utility_path,REPO/'configs/research/response_value_centered.json',
                     REPO/'radio_gs/querying/centered_response_value.py',
                     REPO/'scripts/research/train_tmm_centered_value.py',
                     REPO/'scripts/research/audit_tmm_multitask_utility.py',
                     REPO/'scripts/research/summarize_tmm_utility_diagnosis.py']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    if residual_path.exists():
        residual=read(residual_path)
        for path,digest in residual['receipts'].items():
            assert sha(REPO/path)==digest,path
        record['residual_utility_control']=dict(development_gate=residual['development_gate'],
            same_choice_as_no_null=residual['same_choice_as_no_null'],
            query_control_effects=residual['query_control_effects'],
            matched_full_no_null_trees=128,parent_single_potential_trees=64,
            constraint_accuracy_contribution_established=False,all_five_transfer_complete=False,promoted=False)
        for path in [residual_path,PAPER/'sections/residual_control.tex',PAPER/'tables/residual_transfer.tex',
                     PAPER/'tables/residual_summary.tex',REPO/'configs/research/response_value_residual.json',
                     REPO/'radio_gs/querying/residual_response_value.py',
                     REPO/'radio_gs/response_study/residual_diagnosis.py',
                     REPO/'scripts/research/train_tmm_residual_value.py',
                     REPO/'scripts/research/audit_tmm_residual_value.py',
                     REPO/'scripts/research/summarize_tmm_residual_value.py']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    comparisons = read(PAPER/'paper_experiments.json')
    for path,digest in comparisons['receipts'].items():
        assert sha(REPO/path)==digest,path
        record['source_sha256'][path]=digest
    record['paper_comparisons']=dict(rows=len(comparisons['rows']),
        local_reproduction_artifacts_verified=comparisons['local_reproduction_artifacts_verified'],
        correction_ids=[c['id'] for c in comparisons['corrections']],
        coverage=comparisons['coverage'],all_five_mechanism_complete=False,
        external_baselines_retrained=False)
    correctness = STUDY/'protocol_correctness_audit_v2'
    lineage = STUDY/'protocol_input_lineage_audit_v2'
    audit = read(correctness/'summary.json')
    assert audit['complete'] and not audit['new_model_gain_claim']
    record['protocol_correctness_audit'] = dict(
        lerf=audit['lerf']['macro'],scannet=audit['scannet']['macro'],nvos=audit['nvos']['macro'],
        semantic_rerender=False,global_no_leakage_proof=False,oracle_deployable=False)
    for path in [correctness/'summary.json',correctness/'prediction_seal.json',
                 lineage/'summary.json',lineage/'nvos_error_diagnosis.png',
                 REPO/'scripts/research/audit_tmm_protocol_correctness.py',
                 REPO/'scripts/research/audit_tmm_protocol_lineage.py',
                 REPO/'docs/experiments/2026-10-16-tmm-protocol-correctness-audit.md']:
        record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    for path in [PAPER/'paper_experiments.json',PAPER/'artifacts/paper_comparisons.csv',
                 REPO/'scripts/research/build_tmm_paper_experiments.py',
                 *[PAPER/'sections'/name for name in ('introduction.tex','experiments.tex',
                    'discussion.tex','references.tex')]]:
        record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    membership=read(PAPER/'artifacts/inherited_membership_spec.json')
    assert sha(membership['checkpoint'])==membership['checkpoint_sha256']
    assert not membership['benchmark_split_overlap'] and not membership['new_training']
    for entry in membership['active_maps'].values():
        assert sha(REPO/entry['path'])==entry['sha256']
        record['source_sha256'][entry['path']]=entry['sha256']
    record['source_sha256'][membership['checkpoint']]=membership['checkpoint_sha256']
    record['inherited_membership_spec']=membership
    record['source_sha256']['paper/tmm_response/artifacts/inherited_membership_spec.json']=sha(PAPER/'artifacts/inherited_membership_spec.json')
    record['active_algorithm_candidate']['role'] = 'optional extension, not the entire paper method'
    record['in_house_component_effects'] = evidence['paired_effects'] + evidence['supporting_provider_effects']
    record['source_sha256']['docs/adr/0024-v10-method-ownership-and-nvos-evidence.md'] = sha(REPO/'docs/adr/0024-v10-method-ownership-and-nvos-evidence.md')
    nvos_path=PAPER/'nvos_coverage_feedback.json'
    if nvos_path.exists():
        nvos=read(nvos_path)
        record['nvos_evidence_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            ranking=nvos.get('ranking'),
            inherited_readouts=nvos.get('inherited_readouts',{}).get('macro'),
            field_by_readout=nvos.get('readout_factorial',{}).get('macro'),
            fresh_tasks=8,other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for path,digest in nvos['receipts'].items():
            assert sha(path)==digest,path
            record['source_sha256'][path]=digest
        for path in [nvos_path,PAPER/'tables/nvos_coverage.tex',PAPER/'artifacts/nvos_coverage.csv',
                     REPO/'scripts/research/summarize_tmm_nvos_coverage.py',
                     REPO/'radio_gs/response_study/reference_study.py',
                     REPO/'docs/experiments/2026-10-17-nvos-evidence-and-v10-method.md',
                     PAPER/'artifacts/nvos_environment_20261017.json',
                     PAPER/'sections/nvos_coverage.tex']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
        if (PAPER/'tables/nvos_inherited_readouts.tex').exists():
            record['source_sha256']['paper/tmm_response/tables/nvos_inherited_readouts.tex']=sha(PAPER/'tables/nvos_inherited_readouts.tex')
        if (PAPER/'tables/nvos_readout_factorial.tex').exists():
            record['source_sha256']['paper/tmm_response/tables/nvos_readout_factorial.tex']=sha(PAPER/'tables/nvos_readout_factorial.tex')
    source_readout_path=PAPER/'nvos_source_readout_feedback.json'
    if source_readout_path.exists():
        nvos=read(source_readout_path)
        record['nvos_source_readout_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            thresholds=nvos['thresholds'],shared_final_response=True,fresh_tasks=8,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for path,digest in nvos['receipts'].items():
            assert sha(path)==digest,path
            record['source_sha256'][path]=digest
        for path in [source_readout_path,PAPER/'tables/nvos_source_readout.tex',
                     PAPER/'artifacts/nvos_source_readout.csv',PAPER/'sections/nvos_source_readout.tex',
                     PAPER/'figures/nvos_source_criteria.pdf',PAPER/'figures/nvos_source_criteria.png',
                     REPO/'radio_gs/response_study/source_readout_study.py',
                     REPO/'scripts/research/summarize_tmm_nvos_source_readout.py',
                     REPO/'docs/adr/0025-source-only-response-readout-calibration.md',
                     REPO/'docs/experiments/2026-10-17-nvos-source-readout.md']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
        if (PAPER/'tables/nvos_score_changes.tex').exists():
            record['source_sha256']['paper/tmm_response/tables/nvos_score_changes.tex']=sha(PAPER/'tables/nvos_score_changes.tex')
    local_credit_path=PAPER/'nvos_local_credit_feedback.json'
    if local_credit_path.exists():
        nvos=read(local_credit_path)
        record['nvos_local_credit_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='held_local',fresh_tasks=8,images=48,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for path,digest in nvos['receipts'].items():
            assert sha(path)==digest,path
            record['source_sha256'][path]=digest
        for path in [local_credit_path,PAPER/'tables/nvos_local_credit.tex',
                     PAPER/'tables/nvos_local_credit_macro.tex',PAPER/'tables/nvos_reference_scope.tex',PAPER/'artifacts/nvos_local_credit.csv',
                     PAPER/'sections/nvos_local_credit.tex',PAPER/'sections/nvos_local_credit_results.tex',
                     PAPER/'sections/nvos_local_credit_summary.tex',
                     REPO/'radio_gs/response_study/local_credit_study.py',
                     REPO/'scripts/research/summarize_tmm_nvos_local_credit.py',
                     REPO/'docs/adr/0026-local-response-edit-credit.md',
                     REPO/'docs/experiments/2026-10-17-nvos-local-credit.md']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    image_response_path=PAPER/'nvos_image_response_feedback.json'
    if image_response_path.exists():
        nvos=read(image_response_path)
        record['nvos_image_response_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='rgb_mask_lift',target_rgb_arms=nvos['target_rgb_arms'],fresh_tasks=8,images=56,
            view_conditioned_not_query_global=True,other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for path,digest in nvos['receipts'].items():
            assert sha(path)==digest,path
            record['source_sha256'][path]=digest
        for path in [image_response_path,PAPER/'tables/nvos_image_response.tex',
                     PAPER/'artifacts/nvos_image_response.csv',PAPER/'sections/nvos_image_response.tex',
                     REPO/'radio_gs/response_study/image_response_study.py',
                     REPO/'scripts/research/summarize_tmm_nvos_image_response.py',
                     REPO/'docs/adr/0027-image-conditioned-query-response.md',
                     REPO/'docs/experiments/2026-10-17-nvos-image-response.md']:
            record['source_sha256'][str(path.relative_to(REPO))]=sha(path)
    for suffix,primary,adr in [('partial','positive_core_lift','0028-partial-image-response-permissions'),
                               ('unboxed','unboxed_sparse_lift','0029-identity-without-inherited-box'),
                               ('memory','sequence_memory_lift','0030-reference-memory-response-transport')]:
        path=PAPER/f'nvos_{suffix}_response_feedback.json'
        if not path.exists():continue
        nvos=read(path)
        record[f'nvos_{suffix}_response_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm=primary,target_rgb_used=True,fresh_tasks=8,images=56,
            view_conditioned_not_query_global=True,average_precision_measured=False,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [path,PAPER/f'tables/nvos_{suffix}_response.tex',PAPER/f'artifacts/nvos_{suffix}_response.csv',
                       PAPER/f'sections/nvos_{suffix}_response.tex',REPO/f'scripts/research/summarize_tmm_nvos_{suffix}_response.py',
                       REPO/f'docs/adr/{adr}.md',REPO/f'docs/experiments/2026-10-17-nvos-{suffix}-response.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    native_path=PAPER/'nvos_native_writeback_feedback.json'
    if native_path.exists():
        nvos=read(native_path)
        record['nvos_native_writeback_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='sequence_memory_native',target_rgb_used=True,fresh_tasks=8,images=56,
            view_conditioned_not_query_global=True,average_precision_measured=False,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [native_path,PAPER/'tables/nvos_native_writeback.tex',PAPER/'artifacts/nvos_native_writeback.csv',
                       PAPER/'sections/nvos_native_writeback.tex',REPO/'scripts/research/summarize_tmm_nvos_native_writeback.py',
                       PAPER/'artifacts/nvos_memory_native_storage.json',
                       REPO/'radio_gs/response_study/native_writeback_study.py',REPO/'radio_gs/rendering/gsplat_footprint_cuda.py',
                       REPO/'docs/adr/0031-native-grid-response-writeback.md',REPO/'docs/experiments/2026-10-17-nvos-native-writeback.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    affinity_path=PAPER/'nvos_signed_affinity_feedback.json'
    if affinity_path.exists():
        nvos=read(affinity_path)
        record['nvos_signed_affinity_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='signed_native',target_rgb_used=True,negative_scribble_used=True,frozen_dino_added=True,
            fresh_tasks=8,images=72,view_conditioned_not_query_global=True,average_precision_measured=True,
            other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [affinity_path,PAPER/'tables/nvos_signed_affinity.tex',PAPER/'artifacts/nvos_signed_affinity.csv',
                       PAPER/'artifacts/nvos_affinity_storage.json',
                       PAPER/'sections/nvos_signed_affinity.tex',REPO/'scripts/research/summarize_tmm_nvos_signed_affinity.py',
                       REPO/'radio_gs/response_study/signed_affinity_study.py',REPO/'radio_gs/querying/signed_reference_affinity.py',
                       REPO/'docs/adr/0032-signed-reference-affinity.md',REPO/'docs/experiments/2026-10-17-nvos-signed-affinity.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    direct_path=PAPER/'nvos_direct_affinity_feedback.json'
    if direct_path.exists():
        nvos=read(direct_path)
        record['nvos_direct_affinity_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='signed_direct_native',target_rgb_used=True,negative_scribble_used=True,exact_parent_controls=True,
            fresh_tasks=8,images=48,view_conditioned_not_query_global=True,average_precision_measured=True,
            other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [direct_path,PAPER/'tables/nvos_direct_affinity.tex',PAPER/'artifacts/nvos_direct_affinity.csv',
                       PAPER/'sections/nvos_direct_affinity.tex',REPO/'scripts/research/summarize_tmm_nvos_direct_affinity.py',
                       REPO/'radio_gs/response_study/direct_affinity_study.py',REPO/'docs/adr/0033-direct-affinity-writeback.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    extent_path=PAPER/'nvos_instance_extent_feedback.json'
    if extent_path.exists():
        nvos=read(extent_path)
        record['nvos_instance_extent_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            primary_arm='scribble_anchor_native',target_rgb_used=True,negative_scribble_used=True,
            exact_parent_image_proposals=True,fresh_tasks=8,images=72,view_conditioned_not_query_global=True,
            average_precision_measured=True,other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [extent_path,PAPER/'tables/nvos_instance_extent.tex',PAPER/'artifacts/nvos_instance_extent.csv',
                       PAPER/'artifacts/nvos_instance_extent_storage.json',PAPER/'sections/nvos_instance_extent.tex',PAPER/'sections/nvos_instance_extent_formulation.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_instance_extent.py',REPO/'radio_gs/response_study/instance_extent_study.py',
                       REPO/'radio_gs/querying/instance_extent.py',REPO/'docs/adr/0034-instance-constrained-extent.md',
                       REPO/'docs/experiments/2026-10-17-nvos-instance-extent.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    completion_path=PAPER/'nvos_reference_completion_feedback.json'
    if completion_path.exists():
        nvos=read(completion_path)
        record['nvos_reference_completion_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            previous_candidate_effects=nvos['previous_candidate_effects'],primary_arm='constrained_memory_native',
            target_rgb_used=True,negative_scribble_used=True,fresh_tasks=8,images=72,
            view_conditioned_not_query_global=True,average_precision_measured=True,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [completion_path,PAPER/'tables/nvos_reference_completion.tex',PAPER/'artifacts/nvos_reference_completion.csv',
                       PAPER/'artifacts/nvos_reference_completion_storage.json',PAPER/'sections/nvos_reference_completion.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_reference_completion.py',REPO/'radio_gs/response_study/reference_completion_study.py',
                       REPO/'radio_gs/querying/reference_completion.py',REPO/'docs/adr/0035-reference-completion-before-transport.md',
                       REPO/'docs/experiments/2026-10-18-nvos-reference-completion.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    delta_path=PAPER/'nvos_reference_delta_feedback.json'
    if delta_path.exists():
        nvos=read(delta_path)
        record['nvos_reference_delta_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            previous_candidate_effects=nvos['previous_candidate_effects'],primary_arm='probability_native',
            target_rgb_used=True,negative_scribble_used=True,fresh_tasks=8,images=56,
            view_conditioned_not_query_global=True,average_precision_measured=True,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [delta_path,PAPER/'tables/nvos_reference_delta.tex',PAPER/'artifacts/nvos_reference_delta.csv',
                       PAPER/'artifacts/nvos_reference_delta_storage.json',PAPER/'sections/nvos_reference_delta.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_reference_delta.py',REPO/'radio_gs/response_study/reference_delta_study.py',
                       REPO/'radio_gs/querying/transport_residual.py',REPO/'docs/adr/0036-paired-reference-intervention-residual.md',
                       REPO/'docs/experiments/2026-10-18-nvos-reference-completion.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    admission_path=PAPER/'nvos_reference_admission_feedback.json'
    if admission_path.exists():
        nvos=read(admission_path)
        record['nvos_reference_admission_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            previous_candidate_effects=nvos['previous_candidate_effects'],primary_arm='feasible_native',
            target_rgb_used=True,negative_scribble_used=True,fresh_tasks=8,images=56,
            view_conditioned_not_query_global=True,average_precision_measured=True,other_jobs_unchanged=23,
            default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [admission_path,PAPER/'tables/nvos_reference_admission.tex',PAPER/'artifacts/nvos_reference_admission.csv',
                       PAPER/'artifacts/nvos_reference_admission_storage.json',PAPER/'sections/nvos_reference_admission.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_reference_admission.py',REPO/'radio_gs/response_study/reference_admission_study.py',
                       REPO/'radio_gs/querying/reference_admission.py',REPO/'docs/adr/0037-source-evidence-selects-reference-repair.md',
                       REPO/'docs/experiments/2026-10-18-nvos-reference-completion.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    multiview_path=PAPER/'nvos_multiview_completion_feedback.json'
    if multiview_path.exists():
        nvos=read(multiview_path)
        record['nvos_multiview_completion_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            stopping_decision=nvos['stopping_decision'],primary_arm='completed_joint_native',
            source_consistency_cohort=nvos['source_consistency_cohort'],
            matched_source_completion_leave_one_scene_out=nvos['matched_source_completion_leave_one_scene_out'],
            target_rgb_used=True,source_only_controls_exclude_target_evidence=True,
            negative_scribble_used=True,fresh_tasks=8,images=48,average_precision_measured=True,
            other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [multiview_path,PAPER/'tables/nvos_multiview_completion.tex',PAPER/'artifacts/nvos_multiview_completion.csv',
                       PAPER/'artifacts/nvos_multiview_completion_storage.json',PAPER/'sections/nvos_multiview_completion.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_multiview_completion.py',REPO/'radio_gs/response_study/multiview_completion_study.py',
                       REPO/'radio_gs/querying/multiview_response.py',REPO/'docs/adr/0038-completed-reference-multiview-fusion.md',
                       REPO/'docs/experiments/2026-10-18-nvos-multiview-completion.md',
                       PAPER/'artifacts/nvos_external_comparison_20261018.json']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    redecode_path=PAPER/'nvos_target_redecode_feedback.json'
    if redecode_path.exists():
        nvos=read(redecode_path)
        record['nvos_target_redecode_study']=dict(run=nvos['run'],macro=nvos['macro'],effects=nvos['effects'],
            stopping_decision=nvos['stopping_decision'],primary_arm='crop_native',target_rgb_used=True,
            negative_scribble_used=True,fresh_tasks=8,images=56,average_precision_measured=True,
            other_jobs_unchanged=23,default_changed=False,independent_test=False)
        for receipt,digest in nvos['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [redecode_path,PAPER/'tables/nvos_target_redecode.tex',PAPER/'artifacts/nvos_target_redecode.csv',
                       PAPER/'artifacts/nvos_target_redecode_storage.json',PAPER/'sections/nvos_target_redecode.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_target_redecode.py',REPO/'radio_gs/response_study/target_redecode_study.py',
                       REPO/'radio_gs/querying/response_crop.py',REPO/'docs/adr/0039-query-local-target-redecoding.md',
                       REPO/'docs/experiments/2026-10-18-nvos-target-redecode.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    precision_path=PAPER/'nvos_score_precision_feedback.json'
    if precision_path.exists():
        audit=read(precision_path)
        record['nvos_score_precision_audit']=dict(run=audit['run'],macro=audit['macro'],effects=audit['effects'],
            hard_mask_changes=audit['hard_mask_changes'],diagnostic_only=True,actual_3d_rerun=False,default_changed=False)
        for receipt,digest in audit['receipts'].items():
            assert sha(receipt)==digest,receipt
            record['source_sha256'][receipt]=digest
        for source in [precision_path,PAPER/'tables/nvos_score_precision.tex',PAPER/'artifacts/nvos_score_precision.csv',
                       PAPER/'artifacts/nvos_score_precision_storage.json',PAPER/'sections/nvos_score_precision.tex',
                       REPO/'scripts/research/summarize_tmm_nvos_score_precision.py',REPO/'radio_gs/response_study/score_precision_audit.py',
                       REPO/'docs/experiments/2026-10-18-nvos-score-precision.md']:
            record['source_sha256'][str(source.relative_to(REPO))]=sha(source)
    (PAPER / 'method_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    lines = [r'\begin{tabular}{lll}', r'\toprule',
        r'Claim & Evidence & Scope/status \\', r'\midrule',
        f"Query-conditioned admission & Projection IoU +{identity['delta_pp']:.2f} pp & LERF text control" + r' \\',
        f"Admitted response use & Semantic IoU +{response['delta_pp']:.2f} pp & LERF output integration" + r' \\',
        r'Functional preservation & 22 source jobs; UQIS unchanged & Compatibility, not new gain \\',
        r'Fitting versus task quality & 22/22 lower source objectives & Diagnostic, mixed task effects \\',
        r'Common admission transfer & Matched cross-provider tests & Open \\',
        r'\bottomrule', r'\end{tabular}']
    (PAPER / 'tables/claim_evidence.tex').write_text('\n'.join(lines)+'\n')
    control_table(feedback)
    make_figure()
    make_replacement_figure()
    print(json.dumps(dict(verified_historical_files=len(evidence['source_files']),
        verified_studies=len(feedback),code_bindings=len(bindings),claims=len(claims),
        common_admission_finalized=False,new_inference=False),indent=2))


def make_replacement_figure():
    """Current algorithm; keep the old admission figure for its archived paper."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 3.3))
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.1); ax.axis('off')
    specs = [(.1, 2.7, 'Complete candidates', 'Frozen teachers + decoders\nIncumbent + intact alternatives'),
             (3.25, 2.7, 'Response signatures', 'Query ranks + visible overlap\nNo target labels at inference'),
             (6.4, 3.0, 'Shared scalar utility', r'$z_k=g(x_k,\bar{x},x_k-\bar{x})$'+'\n'+r'$\widehat{\Delta}_k=z_k-z_0$'),
             (9.85, 2.7, 'One replacement rule', 'Verification available\n'+r'$\max_k\widehat{\Delta}_k>\tau$'+'\nElse retain incumbent')]
    for x, width, title, detail in specs:
        ax.add_patch(FancyBboxPatch((x, 1.65), width, 1.45, boxstyle='round,pad=.06',
            facecolor='#EDF4F8' if x < 6 else '#DDEEE9', edgecolor='#46566A', linewidth=1))
        ax.text(x+width/2, 2.77, title, ha='center', va='center', fontsize=9, weight='bold')
        ax.text(x+width/2, 2.19, detail, ha='center', va='center', fontsize=8)
    for a, b in [(2.86, 3.17), (6.01, 6.32), (9.46, 9.77), (12.61, 13.0)]:
        ax.annotate('', xy=(b, 2.36), xytext=(a, 2.36), arrowprops=dict(arrowstyle='->', color='#46566A'))
    ax.text(14, 2.36, 'One intact\nanswer', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(7.5, 3.65, 'Learn when replacing an existing complete answer improves task utility',
            ha='center', fontsize=11, weight='bold')
    ax.text(7.9, .65, 'Separate fit rooms\nActual candidate utility', ha='center', fontsize=8)
    ax.text(11.2, .65, 'Separate calibration rooms\nOne shared margin', ha='center', fontsize=8)
    for x in (7.9, 11.2):
        ax.annotate('', xy=(x, 1.56), xytext=(x, 1.05),
                    arrowprops=dict(arrowstyle='->', color='#46566A', linestyle='--'))
    ax.text(3, .65, 'Fixed query providers and complete answers\nNo feature-field or decoder retraining',
            ha='center', fontsize=8)
    fig.tight_layout(pad=.2)
    for suffix in ('pdf', 'png'):
        fig.savefig(PAPER/f'figures/replacement_overview.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    build()
