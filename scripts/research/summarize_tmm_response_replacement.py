"""Build the learned-policy research report from retained measured results."""
import csv
import json
from pathlib import Path

from scripts.research.replay_tmm_response import STUDY, sha
from radio_gs.querying.response_study_adapters import SYSTEM, read


def main():
    train = STUDY/'response_replacement_train_v2'; transfer = STUDY/'response_replacement_transfer_v1'
    training = read(train/'summary.json'); measured = read(transfer/'summary.json')
    calibration = read(train/'calibration.json'); manifest = read(train/'manifest.json')
    reference = read(SYSTEM/'summary.json')['macro']
    capacity_path = STUDY/'response_replacement_capacity_v1/summary.json'
    capacity = read(capacity_path) if capacity_path.exists() else None
    feedback = dict(method='learned complete-response replacement', training=training, transfer=measured,
        calibration={k: v['selection'] for k, v in calibration.items()},
        evidence={str(p): sha(p) for p in (train/'manifest.json', train/'summary.json', train/'calibration.json',
            train/'model.joblib', transfer/'manifest.json', transfer/'predictions_complete.json', transfer/'summary.json')},
        default_changed=False, all_five_mechanism_closed=False, independent_test=False)
    if capacity is not None:
        feedback['capacity_attribution'] = capacity['macro']
        feedback['fixed_provider_metrics'] = capacity['fixed_provider_metrics']
        feedback['evidence'][str(capacity_path)] = sha(capacity_path)
    paper = Path('paper/tmm_response')
    (paper/'response_replacement_feedback.json').write_text(json.dumps(feedback, indent=2)+'\n')
    rows = []
    for arm, result in training['development'].items():
        m = result['scene_macro']
        rows.append(dict(cohort='independent_NR3D_development_4_rooms', arm=arm, metric='actual_set_iou',
            baseline=100*m['baseline_iou'], value=100*m['selected_iou'], delta_pp=100*m['delta'],
            intervention=True, independent_final_test=False))
    for arm, values in measured['uqis_macro'].items():
        if arm == 'baseline':
            continue
        for key, value in values.items():
            baseline = measured['uqis_macro']['baseline'][key]
            rows.append(dict(cohort='UQIS_9_rooms', arm=arm, metric=key, baseline=100*baseline,
                value=100*value, delta_pp=100*(value-baseline), intervention=key.startswith(('text_', 'image_')),
                independent_final_test=False))
    with (paper/'artifacts/learned_metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    repair = {}
    base = {r['query_id']: r for r in read(transfer/'uqis_baseline_sets.json')['per_query']}
    for arm in ('full', 'query_blind', 'permuted', 'unrestricted'):
        values = read(transfer/f'uqis_{arm}_sets.json')['per_query']
        # Query IDs and modalities come from prediction receipts, not a guessed
        # naming convention or category label.
        modality = {r['query_id']: r['modality'] for job in read(transfer/'manifest.json')['jobs']
                    for r in read(transfer/(job['name']+'.json'))['rows']}
        repair[arm] = {}
        for m in ('text', 'image'):
            changes = [r['iou']-base[r['query_id']]['iou'] for r in values if modality.get(r['query_id']) == m]
            repair[arm][m] = dict(improved=sum(d > 1e-12 for d in changes), harmed=sum(d < -1e-12 for d in changes),
                                 unchanged=sum(abs(d) <= 1e-12 for d in changes))
    feedback['repair_and_harm'] = repair
    receipts = [r for job in read(transfer/'manifest.json')['jobs']
                for r in read(transfer/(job['name']+'.json'))['rows']]
    selected_iou = {r['query_id']: r['iou'] for r in read(transfer/'uqis_full_sets.json')['per_query']}
    feedback['replacement_gain_calibration'] = {}
    for m in ('text', 'image'):
        changed = [r for r in receipts if r['modality'] == m and r['arms']['full']['candidate'] != 0]
        feedback['replacement_gain_calibration'][m] = dict(queries=len(changed),
            predicted_gain=sum(r['arms']['full']['gains'][r['arms']['full']['candidate']] for r in changed)/len(changed),
            actual_gain=sum(selected_iou[r['query_id']]-base[r['query_id']]['iou'] for r in changed)/len(changed),
            unit='query mean among replaced answers; not scene macro or a causal stratum')
    from scripts.research.summarize_tmm_response_study import paired
    feedback['query_correspondence_effects'] = {}
    for m in ('text', 'image'):
        left = read(transfer/'uqis_full_sets.json')['modalities'][m]['per_scene']
        right = read(transfer/'uqis_permuted_sets.json')['modalities'][m]['per_scene']
        feedback['query_correspondence_effects'][m] = paired([left[s]['iou']-right[s]['iou'] for s in left])
    (paper/'response_replacement_feedback.json').write_text(json.dumps(feedback, indent=2)+'\n')
    lines = [
        '# 完整响应替换：独立训练与跨查询迁移', '',
        '日期：2026-10-16。分支 `codex/tmm-query-response`。本报告由保留结果自动生成。', '',
        '## 本轮实质变化', '',
        '把手工验证分数改为学习“替换完整答案的实际收益”。两个完整提供器、教师和几何冻结；',
        '每个查询至多四个区域身份，每个身份交由两个原后端解码。原答案明确保留，',
        '同一模型比较全部完整候选，超过独立校准的单一门槛才替换。没有重训特征场或成员分类器。', '',
        f"训练 {training['training_queries']} 条自然表达／22 房间，校准 {training['calibration_queries']} 条／3 房间，",
        f"开发 {training['development_queries']} 条／4 房间。当前 benchmark 场景与三组房间均不重合。",
        '冻结成员提供器也使用过这 22 个训练房间；四个开发房间此前已打开，不能称为独立最终测试。',
        '完整模型拟合前已固定候选、特征、学习器和门槛规则；开发及 UQIS 不再选门槛。', '',
        '## 独立房间开发结果', '',
        '| 决策 | 原直接成员答案 IoU | 输出 IoU | Δ pp | 改善／损害查询 |',
        '|---|---:|---:|---:|---:|',
    ]
    for arm, r in training['development'].items():
        m = r['scene_macro']
        lines.append(f"| {arm} | {100*m['baseline_iou']:.4f} | {100*m['selected_iou']:.4f} | {100*m['delta']:+.4f} | {r['improved']}／{r['harmed']} |")
    main_result = training['development']['full']['scene_macro']
    lines += ['', f"有限候选标注选优 IoU：{100*main_result['oracle_iou']:.4f}%；它只用于诊断，不是自动成绩。",
              f"预先声明的开发门槛通过：**{training['development_gate']}**。",
              '该门槛同时要求胜过原答案、去掉显式 query 特征的学习器、同视角 query 对应打乱控制，并至少三房间不退化。', '',
              '## 同一模型的 UQIS 迁移', '',
              '134 条文字／图像查询先完成预测封存，再对全部 268 条查询评分；实际点输出精确保留。',
              '文字使用原 SigLIP 证据，图像使用原 SigLIP 与局部 DINO 排名证据；没有教师替换或额外源图像访问。',
              '校准时原答案是直接成员后端，文字原答案是作用域后端，图像增加了第二教师；这些均属显式迁移条件。', '',
              '| 决策 | 文字 IoU | 图像 IoU | 文字 AP | 图像 AP |', '|---|---:|---:|---:|---:|']
    for arm, values in measured['uqis_macro'].items():
        numbers = [values[k]*100 for k in ('text_selected_iou', 'image_selected_iou', 'text_ap', 'image_ap')]
        lines.append('| '+arm+' | '+' | '.join(f'{n:.4f}' for n in numbers)+' |')
    lines += ['', '| 查询 | full 改善 | full 损害 | 不变 |', '|---|---:|---:|---:|']
    for m, r in repair['full'].items():
        lines.append(f"| {m} | {r['improved']} | {r['harmed']} | {r['unchanged']} |")
    lines += ['', '被替换答案的收益预测校准（查询均值，区别于上表的场景均值）：', '']
    for m, r in feedback['replacement_gain_calibration'].items():
        lines.append(f"- {m} 的 {r['queries']} 次替换：预测平均 {100*r['predicted_gain']:+.4f} pp，实际平均 {100*r['actual_gain']:+.4f} pp。")
    lines += ['', '图像替换收益的符号发生系统性偏差；这定位了当前决策接口的迁移失败，但不能仅凭这些数据唯一归因为训练模态缺失。',
              '下一步应在独立房间补足真实图像／多教师的完整答案收益监督，并检验同一策略的收益与损害，不能在这 67 条图像查询上扫门槛。']
    lines += ['', '同视角对应打乱保留了同一候选池、教师分数集合及几何。下面区间只是当前九房间的描述性 bootstrap，不是独立最终测试：', '']
    for m, r in feedback['query_correspondence_effects'].items():
        interval = r['scene_bootstrap95_pp']
        lines.append(f"- {m}：full − permuted = {r['delta_pp']:+.4f} pp，区间 [{interval[0]:+.4f}, {interval[1]:+.4f}]。")
    if capacity is not None:
        lines += ['', '## 复用已有完整能力与新决策的收益分离', '',
                  'A07 在预测封存后重新计算全部 134 个候选池，连续分数与实际集合的哈希均精确复现。',
                  '固定对照对文字和图像都使用“原身份不变、只换另一个完整后端”的同一规则。',
                  '标注选优仅用于有限候选诊断；所有标注在候选重算封存后读取。', '',
                  '| 查询 | v10 | 同身份换后端 | 学习决策 | 有限池标注选优 | 学习决策 − 固定对照 |',
                  '|---|---:|---:|---:|---:|---:|']
        for m, r in capacity['macro'].items():
            lines.append(f"| {m} | {100*r['baseline_iou']:.4f} | {100*r['fixed_provider_iou']:.4f} | {100*r['selected_iou']:.4f} | {100*r['oracle_iou']:.4f} | {100*r['learned_minus_fixed_provider']:+.4f} pp |")
        lines += ['', '精确分解：实际结果 = 原答案 + 候选可用增益 − 决策损失。',
                  '“保留完整后端”已通过数值复现验证；“正确估计替换收益”仍需跨输入类型验证。',
                  '固定对照的收益属于已有能力迁移，不能全部归入新学习器。', '']
    lines += ['', '## 全部 benchmark 的实际覆盖', '',
              '| 输出 | v10 参考（%） | 本轮状态 |', '|---|---:|---|',
              f"| LERF semantic IoU | {100*reference['lerf']['semantic_iou']:.4f} | 历史参考；新机制源观察适配待完成 |",
              f"| LERF selected projection IoU | {100*reference['lerf']['selected_projection_iou']:.4f} | 历史参考；必须单独处理实际选集，不是真实 3D GT IoU |",
              f"| NVOS IoU | {100*reference['nvos']['iou']:.4f} | 历史参考；参考权限的迁移待完成 |",
              f"| ScanNet mIoU19 | {100*reference['scannet']['miou_19']:.4f} | 历史参考；完整类别竞争的迁移待完成 |",
              '| UQIS | 见上表 | 文字／图像实际干预；四种查询重新评分 |', '',
              '**没有将未修改的历史值当作新机制跨五任务成功，也没有拼接每任务最优配置。默认仍是 v10。**', '',
              '## 复现与保存', '', '```bash',
              'python -m scripts.research.train_tmm_response_replacement --run NEW_TRAIN_RUN --workers 4',
              'python -m scripts.research.evaluate_tmm_response_replacement --training-run NEW_TRAIN_RUN --run NEW_TRANSFER_RUN',
              'python -m radio_gs.response_study verify', '```', '',
              '训练记录、配置、源码快照、输入哈希、小模型、逐查询指标保存；完整响应仅在有上限的内存中存在。',
              f"模型：{(train/'model.joblib').stat().st_size:,} bytes；查询预测峰值：{measured['transient_prediction_bytes']:,} bytes。",
              '持久完整预测：0 bytes。未复制教师、地图或字段，也未删除其他历史资产。', '',
              '第一版准备运行在拟合前停止，以纳入两个原提供器；保留 stop receipt，不计为一个模型搜索结果。',
              '架构与条件性结论见 `docs/adr/0023-learned-complete-response-replacement.md` 和',
              '`paper/tmm_response/replacement.tex`；全部机器可读结果见 `response_replacement_feedback.json`。']
    Path('docs/experiments/2026-10-16-tmm-learned-response-replacement.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(training_gate=training['development_gate'], margins=feedback['calibration'],
                          uqis_full=measured['uqis_macro']['full'], repair=repair['full']), indent=2))


if __name__ == '__main__':
    main()
