"""Joint-policy results, explicit whole-cohort gaps and reproducible method note."""
import csv
import argparse
import json
from pathlib import Path

from radio_gs.querying.response_study_adapters import SYSTEM, read
from scripts.research.replay_tmm_response import STUDY, sha
from scripts.research.summarize_tmm_response_study import paired
from radio_gs.querying.response_replacement import choose_response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--value', action='store_true', help='report the controlled set-utility architecture test')
    args = parser.parse_args()
    data = STUDY/'response_replacement_images_v2'
    train = STUDY/('response_value_train_v1' if args.value else 'response_replacement_joint_v1')
    transfer = STUDY/('response_value_transfer_v1' if args.value else 'response_replacement_joint_transfer_v1')
    prefix = 'response_value' if args.value else 'joint_replacement'
    training = read(train/'summary.json'); measured = read(transfer/'summary.json')
    cal = read(train/'calibration.json'); old = read(STUDY/'response_replacement_transfer_v1/summary.json')
    manifest = read(transfer/'manifest.json'); base = {r['query_id']: r['iou'] for r in read(transfer/'uqis_baseline_sets.json')['per_query']}
    receipts = [r for job in manifest['jobs'] for r in read(transfer/(job['name']+'.json'))['rows']]
    modality = {r['query_id']: r['modality'] for r in receipts}
    changes = {}; effects = {}; calibration = {}
    full_sets = read(transfer/'uqis_full_sets.json')
    permutation = read(transfer/'uqis_permuted_sets.json')
    for m in ('text', 'image'):
        records = [r for r in full_sets['per_query'] if modality.get(r['query_id']) == m]
        delta = [r['iou']-base[r['query_id']] for r in records]
        changes[m] = dict(improved=sum(d > 1e-12 for d in delta), harmed=sum(d < -1e-12 for d in delta),
                          unchanged=sum(abs(d) <= 1e-12 for d in delta))
        values = {r['query_id']: r['iou'] for r in records}
        replaced = [r for r in receipts if r['modality'] == m and r['arms']['full']['candidate'] != 0]
        calibration[m] = dict(replaced=len(replaced),
            predicted_gain=sum(r['arms']['full']['gains'][r['arms']['full']['candidate']] for r in replaced)/len(replaced) if replaced else None,
            actual_gain=sum(values[r['query_id']]-base[r['query_id']] for r in replaced)/len(replaced) if replaced else None)
        left = full_sets['modalities'][m]['per_scene']; right = permutation['modalities'][m]['per_scene']
        effects[m] = paired([left[s]['iou']-right[s]['iou'] for s in left])
    # Reuse labelled candidate utilities ONLY after checking the new inference
    # re-created the exact same complete probabilities AND actual selected sets.
    capacity_root = STUDY/'response_replacement_capacity_v1'
    pool = {r['query_id']: r for r in read(capacity_root/'summary.json')['rows']}
    previous = {r['query_id']: r for job in manifest['jobs']
                for r in read(STUDY/'response_replacement_transfer_v1'/(job['name']+'.json'))['rows']}
    scores = {r['query_id']: r['iou'] for r in full_sets['per_query']}
    capacity = {}
    for r in receipts:
        old_r = previous[r['query_id']]
        if any(r[k] != old_r[k] for k in ('candidates', 'candidate_probability_sha256', 'candidate_selection_sha256')):
            raise ValueError('joint-transfer candidate pool differs from retained capacity audit')
        if abs(pool[r['query_id']]['candidate_iou'][r['arms']['full']['candidate']]-scores[r['query_id']]) > 1e-12:
            raise ValueError('new actual set does not match the exact retained candidate utility')
    for m in ('text', 'image'):
        rows = [pool[r['query_id']] for r in receipts if r['modality'] == m]
        scenes = sorted({r['scene'] for r in rows})
        oracle = sum(sum(r['oracle_iou'] for r in rows if r['scene'] == s)/sum(r['scene'] == s for r in rows) for s in scenes)/len(scenes)
        baseline = measured['uqis_macro']['baseline'][m+'_selected_iou']
        actual = measured['uqis_macro']['full'][m+'_selected_iou']
        capacity[m] = dict(oracle_iou=oracle, available_gain=oracle-baseline, decision_regret=oracle-actual,
                           regret_reduction_from_text_only=actual-old['uqis_macro']['full'][m+'_selected_iou'])
    # Matched decision controls are recoverable from SEALED gains and exact
    # candidate-set utilities. This is post-scoring diagnosis, not a new tuned
    # model or a replay of unavailable dense AP predictions.
    zero_control = {}
    for m in ('text', 'image'):
        local = [r for r in receipts if r['modality'] == m]
        per_scene = {}
        for arm in ('full', 'query_blind', 'permuted'):
            grouped = {}
            for r in local:
                candidate = choose_response(r['arms'][arm]['gains'], 0., evidence_available=r['evidence']['verification_views'] > 0)
                utility = pool[r['query_id']]
                grouped.setdefault(utility['scene'], []).append(utility['candidate_iou'][candidate])
            per_scene[arm] = {s: sum(v)/len(v) for s,v in grouped.items()}
        macro = {a: sum(v.values())/len(v) for a,v in per_scene.items()}
        if abs(macro['full']-measured['uqis_macro']['unrestricted'][m+'_selected_iou']) > 1e-12:
            raise ValueError('recovered zero-margin full utility differs from fresh actual-set scoring')
        zero_control[m] = dict(scene_macro_iou=macro,
            full_minus_permuted=paired([per_scene['full'][s]-per_scene['permuted'][s] for s in per_scene['full']]),
            full_minus_query_blind=paired([per_scene['full'][s]-per_scene['query_blind'][s] for s in per_scene['full']]))
    references = read(SYSTEM/'summary.json')['macro']
    feedback = dict(method='set-conditioned complete-response utility' if args.value else 'jointly supervised complete-response replacement', training=training, transfer=measured,
        calibration={k: v['selection'] for k, v in cal.items()}, repair_and_harm=changes,
        query_correspondence_effects=effects, replacement_gain_calibration=calibration,
        exact_candidate_pools_reused=134, finite_pool_diagnosis=capacity,
        matched_zero_margin_controls=dict(results=zero_control, post_scoring_diagnostic=True,
            source='sealed all-candidate predicted gains and exact A07 candidate IoUs',
            dense_ap_not_available_for_additional_controls=True, policy_not_refitted=True),
        whole_cohort=[
            dict(output='LERF semantic2D', value=references['lerf']['semantic_iou'], status='source adapter and independent supervision pending', fresh=False),
            dict(output='LERF selected-set 2D projection', value=references['lerf']['selected_projection_iou'], status='independent selection/projection adapter pending', fresh=False),
            dict(output='NVOS', value=references['nvos']['iou'], status='reference-authority transfer pending', fresh=False),
            dict(output='ScanNet OVS', value=references['scannet']['miou_19'], status='whole category-menu decision and utility supervision pending', fresh=False),
            dict(output='UQIS', value=None, status='134 text/image interventions, 268 queries scored', fresh=True)],
        evidence={str(p): sha(p) for p in [data/'summary.json', train/'manifest.json', train/'model.joblib',
            train/'calibration.json', train/'summary.json', transfer/'manifest.json', transfer/'summary.json', transfer/'predictions_complete.json', capacity_root/'summary.json']},
        all_five_mechanism_closed=False, promoted=False, independent_final_test=False)
    feedback['full_matches_query_blind_candidates'] = all(r['arms']['full']['candidate'] == r['arms']['query_blind']['candidate'] for r in receipts)
    joint_parent = None
    if args.value:
        parent_path = STUDY/'response_replacement_joint_transfer_v1/summary.json'
        joint_parent = read(parent_path)
        feedback['controlled_joint_pairwise_parent'] = joint_parent['uqis_macro']
        feedback['evidence'][str(parent_path)] = sha(parent_path)
    paper = Path('paper/tmm_response')
    (paper/(prefix+'_feedback.json')).write_text(json.dumps(feedback, indent=2)+'\n')
    metrics = []
    for arm, values in measured['uqis_macro'].items():
        for metric, value in values.items():
            metrics.append(dict(arm=arm, metric=metric, percent=100*value,
                delta_pp=100*(value-measured['uqis_macro']['baseline'][metric]),
                intervention=metric.startswith(('text_', 'image_')), independent_test=False))
    with (paper/'artifacts'/('response_value_metrics.csv' if args.value else 'joint_metrics.csv')).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    image_summary = read(data/'summary.json')
    lines = ['# '+('候选集合共同效用：完整响应决策的架构对照' if args.value else '真实图像与文本联合监督的完整响应决策'), '',
        '日期：2026-10-16。自动生成；默认仍为 v10。', '',
        '本轮补齐决策层的功能监督：输入已有完整候选及原教师证据，直接学习真实答案替换收益。',
        '没有改动特征场、教师坐标或完整成员／作用域解码器。模型看不到模态标签、场景名或 benchmark 名。', '',
        '## 数据与方法', '',
        f"独立 29 房间共生成 {sum(image_summary['scenes'].values())} 条真实 RGB 图像查询；",
        f"训练／校准／开发查询数分别为 {training['training_queries']}、{training['calibration_queries']}、{training['development_queries']}。",
        '图像是原 RGB-D 中的自然裁剪，用官方实例标注仅完成独立训练数据的构造及收益监督；不向模型提供裁剪掩码。',
        '按独立构造标注的几何可见性覆盖选至多四个参考视角，其时间／位姿近邻从所有候选提供器输入中共同排除；',
        '保留原 SAM 区域和 SigLIP 坐标，流式重算原 DINO 局部 Chamfer 匹配，不以池化特征冒充局部对应。',
        '几何仍为完整扫描官方网格；它不是查询视角完全不可知的重建实验。自动裁剪未做全队列人工视觉筛查。',
        '已抽查首个固定训练房间的 12 个裁剪，像素哈希均可复现；部分图像模糊、遮挡或越界补黑。该抽查不代表全体图像审核，也未据此修改构造参数。',
        '最初固定时间分位的准备运行因遗漏物体导致空房间，已在模型拟合前停止并留档；最终协议要求至少两个保留目标视角、50% 目标可见覆盖。',
        '三组房间与 benchmark 场景无交集。成员提供器也使用过 22 个训练房间；开发四房间早已打开。', '',
        '文本使用之前封存的全部候选监督，按实际部署把作用域答案放在候选零；图像以直接成员答案为零。',
        '树模型容量、39 维基础签名、四身份×双后端候选和全局替换规则固定；效用估计结构见下文。每个房间、模态、查询等权。',
        '一个全局门槛同时约束三个校准房间×两个模态的六个分组；允许全部拒绝，不能用文字收益抵消图像损害。',
        '这一约束只是有限校准集上的经验约束，不提供分布外无损保证。', '',
        '## 校准和开发', '',
        f"全策略门槛：{cal['full']['selection']['margin']}；去查询特征控制：{cal['query_blind']['selection']['margin']}。",
        f"开发检查：{training['modality_gates']}；整体通过：{training['development_gate']}。", '',
        '| 条件 | 模态 | 原答案 IoU % | 输出 IoU % | Δ pp | 改善／损害 |', '|---|---|---:|---:|---:|---:|']
    for arm, value in training['development'].items():
        for m, r in value['by_modality'].items():
            v = r['scene_macro']
            lines.append(f"| {arm} | {m} | {100*v['baseline_iou']:.4f} | {100*v['selected_iou']:.4f} | {100*v['delta']:+.4f} | {r['improved']}／{r['harmed']} |")
    if args.value:
        lines += ['', '本对照只更改效用估计结构：每个候选输入自身签名、集合均值和两者之差；',
            '监督目标为候选实际 IoU 减去该查询候选池的平均 IoU。预测所有候选的共同效用，再以效用差决定替换。',
            '它保证候选顺序等变、三者比较的环路和为零；不保证真实排序正确。监督样本、完整候选、教师、',
            '39 维签名、树模型容量及六分组校准规则均与联合成对模型相同。',
            '架构由成对模型在独立校准集上的高置信度有害替换触发，在打开该轮新 benchmark 结果前固定。',
            '不能把失败唯一归因于比较不一致，也不能把这种代数一致性当作经验精度证明。']
    lines += ['', '## 同一个模型、同一个门槛的 UQIS 迁移', '',
        '| 条件 | 文本 IoU % | 图像 IoU % | 文本 AP % | 图像 AP % |', '|---|---:|---:|---:|---:|']
    comparisons = [('v10', measured['uqis_macro']['baseline']), ('此前文本监督 L02', old['uqis_macro']['full'])]
    if joint_parent is not None:
        comparisons.append(('同监督成对模型 L04', joint_parent['uqis_macro']['full']))
    for name, values in comparisons + list(measured['uqis_macro'].items())[1:]:
        lines.append('| '+name+' | '+' | '.join(f'{100*values[k]:.4f}' for k in ('text_selected_iou', 'image_selected_iou', 'text_ap', 'image_ap'))+' |')
    for m, r in changes.items():
        lines += ['', f"{m}：改善 {r['improved']}，损害 {r['harmed']}，不变 {r['unchanged']}。",
                  f"full − query-permuted 的场景均值差：{effects[m]['delta_pp']:+.4f} pp，描述性区间 {effects[m]['scene_bootstrap95_pp']}。"]
    if feedback['full_matches_query_blind_candidates']:
        lines += ['', '**full 与去显式 query 特征控制选择了完全相同的候选；当前结果不能证明显式查询签名的独立贡献。**']
    lines += ['', '这些是已打开 benchmark 上的开发性迁移结果，不能称为独立最终检验。',
        '同时变化的是监督模态、图像视角隔离、训练组权重与校准分组，因此与 L02 的差值不能单独归因为多教师监督。',
        '控制保持同一候选池；查询打乱只改变区域与查询证据的对应，不能把完整后端本身的能力算作新学习器贡献。', '',
        '134 个候选池的连续响应和实际选集哈希均与 A07 完全一致，新增结果重新推理和评分；因此可复用封存的候选效用做精确归因。', '',
        '| 模态 | 有限池标注选优 % | 可用增益 pp | 当前决策损失 pp | 相对旧决策减少的损失 pp |', '|---|---:|---:|---:|---:|']
    for m, r in capacity.items():
        lines.append(f"| {m} | {100*r['oracle_iou']:.4f} | {100*r['available_gain']:.4f} | {100*r['decision_regret']:.4f} | {100*r['regret_reduction_from_text_only']:+.4f} |")
    lines += ['', '标注选优只用于诊断，绝不是自动方法成绩。', '',
        '为区分查询对应作用与拒绝门槛的作用，再从封存的所有候选收益及精确候选 IoU 计算共同零门槛控制；',
        '这是评分后的诊断，不重新拟合模型。额外控制只有实际集合 IoU，没有重新生成密集 AP。', '',
        '| 模态 | 查询模型、零门槛 IoU % | 去显式 query 特征、零门槛 IoU % | 对应打乱、零门槛 IoU % |', '|---|---:|---:|---:|']
    for m, r in zero_control.items():
        lines.append('| '+m+' | '+' | '.join(f"{100*r['scene_macro_iou'][a]:.4f}" for a in ('full', 'query_blind', 'permuted'))+' |')
    lines += ['',
        '## 五项输出完成度', '', '| 输出 | 参考指标 % | 本轮状态 |', '|---|---:|---|']
    for r in feedback['whole_cohort']:
        number = f"{100*r['value']:.4f}" if r['value'] is not None else '见上表'
        lines.append(f"| {r['output']} | {number} | {r['status']} |")
    lines += ['', '不能将四项历史参考值与 UQIS 的新结果拼成跨五 benchmark 的新方法成绩。',
        '剩余实质缺口是源辅助任务的合法观察签名和完整类别菜单的独立收益监督；不是继续对已打开查询调门槛。', '',
        '## 复现与保存', '', '```bash',
        'python scripts/research/compile_tmm_image_supervision.py --run NEW_IMAGES',
        ('python scripts/research/train_tmm_response_value.py --run NEW_JOINT' if args.value else 'python scripts/research/train_tmm_joint_replacement.py --images NEW_IMAGES --run NEW_JOINT'),
        'python scripts/research/evaluate_tmm_response_replacement.py --training-run NEW_JOINT --run NEW_TRANSFER',
        '```', '',
        '保存逐查询候选特征、实际收益、构造坐标、输入哈希、源码快照和小模型。原始扫描、教师、地图均按路径复用；',
        '不保存重复 RGB 裁剪、DINO 密集特征或密集完整响应。裁剪由保留原始扫描和坐标复现。',
        f"新决策模型 { (train/'model.joblib').stat().st_size:,} bytes；本轮持久密集预测 0 bytes。"]
    name = '2026-10-16-tmm-response-value.md' if args.value else '2026-10-16-tmm-joint-response-replacement.md'
    Path('docs/experiments',name).write_text('\n'.join(lines)+'\n')
    table = [r'\begin{tabular}{lrrrr}', r'\toprule', r'Policy & Text IoU & Image IoU & Text AP & Image AP \\', r'\midrule']
    for name, values in [('v10', measured['uqis_macro']['baseline']), ('Set utility' if args.value else 'Joint pairwise', measured['uqis_macro']['full']), ('Query permuted', measured['uqis_macro']['permuted'])]:
        table.append(name+' & '+' & '.join(f'{100*values[k]:.2f}' for k in ('text_selected_iou', 'image_selected_iou', 'text_ap', 'image_ap'))+r' \\')
    table += [r'\bottomrule', r'\end{tabular}']
    (paper/'tables'/(prefix+'.tex')).write_text('\n'.join(table)+'\n')
    sentence = []
    for m in ('text', 'image'):
        values = measured['uqis_macro']; gain = 100*(values['full'][m+'_selected_iou']-values['baseline'][m+'_selected_iou'])
        r = changes[m]; effect = effects[m]
        sentence.append(f"For {m}, actual set IoU changes by {gain:+.2f} percentage points, "
            f"with {r['improved']} improved and {r['harmed']} harmed queries. "
            f"The full-minus-permuted scene-macro difference is {effect['delta_pp']:+.2f} points "
            f"(descriptive interval [{effect['scene_bootstrap95_pp'][0]:+.2f}, {effect['scene_bootstrap95_pp'][1]:+.2f}]).")
    sentence.append('The predeclared joint development gate '+('passes.' if training['development_gate'] else 'does not pass.')+
        ' All 134 complete candidate pools replay exactly. No full-five or independent final-test claim is made.')
    if feedback['full_matches_query_blind_candidates']:
        sentence.append('The full and query-feature-blind policies choose identical candidates, so the measured gain does not establish an independent benefit of the explicit query signature.')
    if args.value:
        sentence.append('Under matched zero-margin diagnosis, full-minus-permuted set IoU is '
            f"{zero_control['text']['full_minus_permuted']['delta_pp']:+.3f} points for text and "
            f"{zero_control['image']['full_minus_permuted']['delta_pp']:+.3f} for images; both descriptive intervals cross zero. "
            'This post-scoring diagnostic reuses sealed gains and exact candidate utilities; it does not add dense AP results.')
    (paper/'tables'/('response_value_summary.tex' if args.value else 'joint_summary.tex')).write_text('\n'.join(sentence)+'\n')
    print(json.dumps(dict(margins=feedback['calibration'], uqis=measured['uqis_macro']['full'], repair=changes), indent=2))


if __name__ == '__main__':
    main()
