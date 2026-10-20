"""Render sealed joint utility results, the fixed context control and attribution."""
import json
from pathlib import Path
from radio_gs.response_study.registry import read, sha


def main():
    registry = read('configs/research/tmm_response.json')
    study = Path(registry['study_root']); paper = Path(registry['paper_root'])
    names = dict(joint='response_value_multitask_v1', relative='response_value_centered_v1',
                 audit='multitask_utility_audit_v1', transfer='response_value_multitask_transfer_v1',
                 capacity='response_value_multitask_diagnosis_v1')
    data = {k: read(study/v/'summary.json') for k, v in names.items()
            if (study/v/'summary.json').exists()}
    feedback = dict(completed=list(data), default_changed=False, independent_final_test=False,
                    all_five_mechanism_closed=False, receipts={})
    extra_receipts = []
    lines = ['# 联合效用与相对差异对照：结果和失败归因', '',
        '由封存结果自动生成。主线保留完整答案、原教师和提供器；模型只决定是否替换。',
        '同一个模型和统一门限服务所有已接通任务。开发与 benchmark 早已打开，默认仍为 v10。', '',
        '## 独立房间开发：两种输入结构', '',
        'joint 使用候选签名、集合均值和相对差异；relative 只使用相对差异。',
        '其余教师、候选、数据划分、样本权重、模型容量和 15 分组校准保持一致。',
        'relative 的设计和代码在 joint 结果产生前封存；设计时读取过无标签目标签名。', '',
        '| 模型 | 条件 | 输出 | 原答案 % | 输出 % | Δ pp | 改善/损害 |',
        '|---|---|---|---:|---:|---:|---:|']
    feedback['development'] = {}
    tex = [r'\begin{tabular}{llrrr}', r'\toprule',
           r'Input & Task & Base & Full & $\Delta$ (pp) \\', r'\midrule']
    for key in ('joint', 'relative'):
        if key not in data:
            continue
        result = data[key]; root = study/names[key]
        calibration = read(root/'calibration.json')
        feedback['development'][key] = dict(margins={a: r['selection']['margin'] for a, r in calibration.items()},
            gate=result['development_gate'], task_gates=result['modality_gates'], results=result['development'])
        if key == 'relative' and calibration['full']['selection']['margin'] is None:
            feedback['relative_transfer_status'] = dict(full_policy='always retain incumbent by construction',
                fresh_benchmark_inference=False, benchmark_metrics_claimed=False,
                additional_dense_replay_required_to_establish_full_policy_identity=False)
        for arm, tasks in result['development'].items():
            for output, r in {**tasks['by_modality'], **{'menu'+k: v for k, v in tasks['category_outputs'].items()}}.items():
                v = r['scene_macro']
                lines.append(f"| {key} | {arm} | {output} | {100*v['baseline_iou']:.4f} | "
                             f"{100*v['selected_iou']:.4f} | {100*v['delta']:+.4f} | {r['improved']}/{r['harmed']} |")
                if arm == 'full' and output in ('text', 'image', 'category'):
                    tex.append(f"{key} & {output} & {100*v['baseline_iou']:.2f} & "
                               f"{100*v['selected_iou']:.2f} & {100*v['delta']:+.2f}" + r' \\')
        lines += ['', f"{key}：统一门限 {calibration['full']['selection']['margin']}；"
                  f"开发门槛通过 {result['development_gate']}；分任务检查 {result['modality_gates']}。", '']
    tex += [r'\bottomrule', r'\end{tabular}']
    (paper/'tables/multitask_development.tex').write_text('\n'.join(tex)+'\n')
    if 'audit' in data:
        audit = data['audit']; feedback['fixed_model_diagnosis'] = audit
        lines += ['## 固定模型的误差与输入域', '',
            '以下范围由独立训练房间定义；目标仅使用签名，不读取 benchmark GT。',
            '越界与树重要性只能描述分布差异和模型使用情况，不能单独证明错误原因。', '',
            '| 签名 | 训练范围 | Gaussian 范围 | 超出训练范围的候选/30 |', '|---|---|---|---:|']
        for r in sorted(audit['category_fullmap_range']['features'], key=lambda r: -r['outside_fit_count'])[:8]:
            lines.append(f"| {r['feature']} | {r['fit_min']:.4f}–{r['fit_max']:.4f} | "
                         f"{r['target_min']:.4f}–{r['target_max']:.4f} | {r['outside_fit_count']} |")
        lines += ['', '| 数据 | 条件 | 任务 | 候选可用增益 pp | 决策损失 pp | 实际增益 pp |',
                  '|---|---|---|---:|---:|---:|']
        for split, arms in audit['independent_outputs'].items():
            for arm, tasks in arms.items():
                for task, result in tasks.items():
                    v = result['scene_macro']
                    lines.append(f"| {split} | {arm} | {task} | {100*v['available_gain']:.4f} | "
                                 f"{100*v['regret']:.4f} | {100*v['actual_gain']:+.4f} |")
        lines += ['', '分解恒等式为：实际增益 = 完整候选池可用增益 − 决策损失。',
                  '有界 oracle 只用于归因，不是本方法结果；所有数值先按房间聚合。', '']
    if 'transfer' in data:
        transfer = data['transfer']; feedback['transfer'] = transfer
        from scripts.research.summarize_tmm_response_study import paired
        root = study/names['transfer']
        extra_receipts += [root/('uqis_'+a+'_sets.json') for a in ('baseline', 'full', 'query_blind', 'permuted')]
        extra_receipts.append(study/'response_value_transfer_v1/summary.json')
        full = read(root/'uqis_full_sets.json'); base = read(root/'uqis_baseline_sets.json')
        baseline = {r['query_id']: r['iou'] for r in base['per_query']}
        delta = [r['iou']-baseline[r['query_id']] for r in full['per_query']]
        feedback['changed_actual_sets'] = dict(improved=sum(d>1e-12 for d in delta),
            harmed=sum(d<-1e-12 for d in delta), unchanged=sum(abs(d)<=1e-12 for d in delta))
        feedback['query_control_effects'] = {}
        for arm in ('query_blind', 'permuted'):
            control = read(root/('uqis_'+arm+'_sets.json'))
            left = full['modalities']['image']['per_scene']; right = control['modalities']['image']['per_scene']
            feedback['query_control_effects'][arm] = paired([left[s]['iou']-right[s]['iou'] for s in left])
        tex = [r'\begin{tabular}{lrrrrrrr}', r'\toprule',
            r'Policy & S19 & S15 & S10 & Text IoU & Image IoU & Text AP & Image AP \\', r'\midrule']
        lines += ['## 同一 joint 模型和门限的实际 benchmark 迁移', '',
            '| 条件 | ScanNet19 % | ScanNet15 % | ScanNet10 % | 文本 IoU % | 图像 IoU % | 文本 AP % | 图像 AP % |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ('baseline', 'full', 'query_blind', 'permuted', 'unrestricted'):
            c = transfer['category_macro'][arm]; u = transfer['uqis_macro'][arm]
            values = [c[k] for k in ('miou_19', 'miou_15', 'miou_10')]
            values += [u[k] for k in ('text_selected_iou', 'image_selected_iou', 'text_ap', 'image_ap')]
            lines.append('| '+arm+' | '+' | '.join(f'{100*v:.4f}' for v in values)+' |')
            tex.append(arm.replace('_', ' ')+' & '+' & '.join(f'{100*v:.2f}' for v in values)+r' \\')
        tex += [r'\bottomrule', r'\end{tabular}']
        (paper/'tables/multitask_transfer.tex').write_text('\n'.join(tex)+'\n')
        lines += ['', '全 10 个完整菜单与 134 个文本/图像答案先封存，再评分全部 268 个 UQIS 查询。',
                  '点查询保留原答案；各类别子菜单使用同一个候选。relative 尚无本轮 benchmark 推理。', '']
        for arm, effect in feedback['query_control_effects'].items():
            lines += [f"图像 IoU full − {arm}：{effect['delta_pp']:+.4f} pp，"
                      f"描述性场景 bootstrap 95% 区间 {effect['scene_bootstrap95_pp']}。", '']
        lines += [f"全队列实际选集改善/损害/不变：{feedback['changed_actual_sets']}。", '']
        previous = read(study/'response_value_transfer_v1/summary.json')['uqis_macro']['full']
        feedback['previous_instance_only_image_iou'] = previous['image_selected_iou']
        lines += [f"此前仅文本/图像联合模型 L06 的图像 IoU 为 {100*previous['image_selected_iou']:.4f}%；"
                  '本轮没有刷新这一最好结果，且未将历史较优分项拼入统一策略。', '']
    if 'capacity' in data:
        capacity = data['capacity']
        feedback['benchmark_capacity'] = {k: capacity[k] for k in
            ('category_macro', 'instance_macro', 'category_same_choice_as_blind', 'category_same_choice_as_permuted')}
        lines += ['## Benchmark 候选能力与决策损失', '',
            '| 输出 | 可用增益 pp | 决策损失 pp | 实际增益 pp |', '|---|---:|---:|---:|']
        r = capacity['category_macro']['capacity']
        lines.append(f"| 完整菜单三输出均值 | {100*r['available_mean_gain']:.4f} | {100*r['decision_mean_regret']:.4f} | {100*r['selected_mean_gain']:+.4f} |")
        for task, r in capacity['instance_macro'].items():
            lines.append(f"| {task} | {100*r['available_gain']:.4f} | {100*r['decision_regret']:.4f} | "
                         f"{100*(r['selected_iou']-r['baseline_iou']):+.4f} |")
        lines += ['', f"10 个类别决策中，与 query-blind 相同 {capacity['category_same_choice_as_blind']}，"
                  f"与查询置乱相同 {capacity['category_same_choice_as_permuted']}。", '']
    lines += ['## 尚未闭环的范围', '',
        'LERF semantic 2D、独立选集的二维投影和 NVOS 尚无此模型的有效迁移结果。',
        'NVOS 现有每视角单个目标区域不满足区域签名的比较条件，需补齐适用的观察接口。',
        '训练 mesh 与 benchmark Gaussian 的可见域不同；六个类别未被校准房间覆盖；',
        'AP 和独立 LERF 选集没有直接效用监督。不能将保留原输出、兼容性或开发检查等同于机制成功。', '']
    if 'relative_transfer_status' in feedback:
        lines += ['relative 的门限为 reject-all，因此正式策略按定义始终返回原答案；不再浪费一次稠密重跑。',
                  '这是一条决策恒等性结论，不记录成新的 benchmark 实测或精度提升。', '']
    feedback['receipts'] = {str(study/names[k]/'summary.json'): sha(study/names[k]/'summary.json') for k in data}
    feedback['receipts'].update({str(p): sha(p) for p in extra_receipts})
    (paper/'utility_diagnosis_feedback.json').write_text(json.dumps(feedback, indent=2)+'\n')
    Path('docs/experiments/2026-10-16-tmm-joint-utility-diagnosis.md').write_text('\n'.join(lines)+'\n')
    print('joint utility report:', list(data))


if __name__ == '__main__':
    main()
