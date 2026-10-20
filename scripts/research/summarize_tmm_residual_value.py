"""Generate a controlled residual-utility result, including negative evidence."""
import json
from pathlib import Path
from radio_gs.response_study.registry import read,sha
from scripts.research.summarize_tmm_response_study import paired


def main():
    registry=read('configs/research/tmm_response.json');study=Path(registry['study_root']);paper=Path(registry['paper_root'])
    names=dict(training='response_value_residual_v1',transfer='response_value_residual_transfer_v1',
               capacity='response_value_residual_diagnosis_v1',audit='residual_utility_audit_v1')
    data={k:read(study/v/'summary.json') for k,v in names.items()}
    paths=[study/v/'summary.json' for v in names.values()]
    root=study/names['transfer'];train=study/names['training'];cal=read(train/'calibration.json')
    paths += [train/'calibration.json',train/'fit_diagnostics.json',root/'ram_released.json',root/'predictions_complete.json']
    arms=('full','query_blind','permuted','unrestricted','unregularized');transfer=data['transfer'];audit=data['audit']
    feedback=dict(default_changed=False,independent_final_test=False,all_five_transfer_complete=False,
        development_gate=data['training']['development_gate'],margins={a:r['selection']['margin'] for a,r in cal.items()},
        transfer=transfer,audit=audit,query_control_effects={},changed_actual_sets={})
    scores={a:read(root/('uqis_'+a+'_sets.json')) for a in ('baseline',)+arms}
    paths += [root/('uqis_'+a+'_sets.json') for a in scores]
    for arm in ('query_blind','permuted','unregularized'):
        feedback['query_control_effects'][arm]={}
        for task in ('text','image'):
            full=scores['full']['modalities'][task]['per_scene'];control=scores[arm]['modalities'][task]['per_scene']
            feedback['query_control_effects'][arm][task]=paired([full[s]['iou']-control[s]['iou'] for s in full])
    base={r['query_id']:r['iou'] for r in scores['baseline']['per_query']}
    for arm in arms:
        delta=[r['iou']-base[r['query_id']] for r in scores[arm]['per_query']]
        feedback['changed_actual_sets'][arm]=dict(improved=sum(d>1e-12 for d in delta),
            harmed=sum(d<-1e-12 for d in delta),unchanged=sum(abs(d)<=1e-12 for d in delta))
    matches=0;decisions=0
    for job in read(root/'manifest.json')['jobs']:
        p=root/(job['benchmark']+'_'+job['name']+'.json');paths.append(p);report=read(p)
        for row in report['rows'] if job['benchmark']=='uqis' else [report]:
            decisions+=1;matches+=row['arms']['full']['candidate']==row['arms']['unregularized']['candidate']
    feedback['same_choice_as_no_null']=dict(matches=matches,decisions=decisions)
    lines=['# 查询证据残差效用：机制对照与误差归因','',
        '由 L10/D05/L11/A12/A13 封存记录自动生成。所有开发集、校准集和 benchmark 均已打开；不是独立最终测试。',
        '本轮不改变特征场、教师、完整候选或默认 v10。研究变量是：查询证据能否修正结构先验的效用误差。','',
        '## 单一机制与公平对照','',
        '效用为 V = b(s) + r(s,q)。先用结构签名拟合 b，再用按房间五折交叉拟合的 b 预测构造残差目标。',
        '每个房间的全部文本、图像、类别样本同时留出；22 个训练房间、3 个校准房间、4 个开发房间不变。',
        '完整模型为真实查询学习效用残差，为区域对应关系置乱的查询约束零修正。零修正不等于零实际效用。',
        '无置乱约束对照（unregularized）使用同一结构先验、同一残差树容量和样本权重，仅将置乱样本换成真实样本副本。',
        '二者各有 128 棵树，比 L07 的 64 棵多；因此主要机制比较是 full 与同容量对照，而不是把与 L07 的差异全归因于约束。',
        '各模型独立校准一个全任务共用门限，仍保护 15 个房间／输出单元；另报告零门限，区分模型排序与门限影响。','',
        f"门限：{feedback['margins']}。开发门槛通过：{feedback['development_gate']}。",'',
        '## 全队列联合迁移','',
        '| 条件 | ScanNet19 % | ScanNet15 % | ScanNet10 % | 文本 IoU % | 图像 IoU % | 文本 AP % | 图像 AP % |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    tex=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'Policy & S19 & Text IoU & Image IoU & Text AP & Image AP \\',r'\midrule']
    for arm in ('baseline',)+arms:
        c=transfer['category_macro'][arm];u=transfer['uqis_macro'][arm]
        values=[c[k] for k in ('miou_19','miou_15','miou_10')]+[u[k] for k in ('text_selected_iou','image_selected_iou','text_ap','image_ap')]
        lines.append('| '+arm+' | '+' | '.join(f'{100*v:.4f}' for v in values)+' |')
        tex.append(arm.replace('_',' ')+' & '+' & '.join(f'{100*values[i]:.2f}' for i in (0,3,4,5,6))+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (paper/'tables/residual_transfer.tex').write_text('\n'.join(tex)+'\n')
    lines += ['', '10 个类别菜单与 134 个文本／图像候选决策全部先封存，再评分全部 268 个 UQIS 查询。',
        '每个类别场景的 19/15/10 输出使用同一个完整菜单。所有实际点输出精确保留；不按任务拼接最优对照。','']
    lines += [f'full 与同容量无置乱约束对照在 {matches}/{decisions} 个完整答案决策上相同。',
        '正式图像 IoU/AP 恢复到 L06 的 40.9859% / 60.5771%，没有超过既有最好值；本轮不能把恢复归功于置乱约束。','']
    for arm,values in feedback['changed_actual_sets'].items():lines.append(f'- {arm}：全 268 条实际选集的改善／损害／不变为 {values}。')
    lines += ['', '| 比较（图像 IoU） | full − 对照 pp | 描述性 95% 场景区间 |','|---|---:|---|']
    for arm,effects in feedback['query_control_effects'].items():
        e=effects['image'];lines.append(f"| {arm} | {e['delta_pp']:+.4f} | {e['scene_bootstrap95_pp']} |")
    lines += ['', '## 开发集与约束作用','',
        '| 条件 | 文本增益 pp | 图像增益 pp | 完整菜单均值增益 pp |','|---|---:|---:|---:|']
    for arm,result in data['training']['development'].items():
        values=[result['by_modality'][t]['scene_macro']['delta'] for t in ('text','image','category')]
        lines.append('| '+arm+' | '+' | '.join(f'{100*v:+.4f}' for v in values)+' |')
    fit=audit['fit'];reduction=1-fit['null_zero_correction_mse']/fit['unregularized_null_correction_mse']
    u=transfer['uqis_macro'];c=transfer['category_macro'];effects=feedback['query_control_effects']
    summary_tex=(f"The shuffled-training correction MSE falls by {100*reduction:.2f}\\%, yet the declared development gate fails. "
        f"The full policy yields image IoU/AP {100*u['full']['image_selected_iou']:.2f}/{100*u['full']['image_ap']:.2f}\\%, "
        f"matching the previous instance-only best. It improves {feedback['changed_actual_sets']['full']['improved']} query "
        f"and harms {feedback['changed_actual_sets']['full']['harmed']}. "
        f"The full and no-null policies choose the same answer in {matches}/{decisions} decisions, so the null penalty "
        "does not explain the deployed gain. At zero margin, text and image IoU reach "
        f"{100*u['unrestricted']['text_selected_iou']:.2f}/{100*u['unrestricted']['image_selected_iou']:.2f}\\%, "
        f"but menu mIoU19 falls from {100*c['baseline']['miou_19']:.2f} to {100*c['unrestricted']['miou_19']:.2f}\\%. "
        "This shared-policy tradeoff remains unresolved; the control is not promoted.\n")
    (paper/'tables/residual_summary.tex').write_text(summary_tex)
    lines += ['',f"训练置乱样本的修正 MSE：约束模型 {fit['null_zero_correction_mse']:.8g}，无约束对照 "
        f"{fit['unregularized_null_correction_mse']:.8g}，下降 {100*reduction:.2f}%。这不是精度。",'',
        '| 数据 | 模型 | 任务 | 真实证据修正 MSE | 置乱证据修正 MSE | 对应关系造成的相对修正差 MSE |',
        '|---|---|---|---:|---:|---:|']
    for split,models in audit['null_behavior'].items():
        for arm,tasks in models.items():
            for task,row in tasks.items():
                r=row['scene_macro'];lines.append(f"| {split} | {arm} | {task} | {r['aligned_mse']:.8f} | {r['null_mse']:.8f} | {r['correspondence_gain_mse']:.8f} |")
    lines += ['', '## 候选能力、排序和保留门限的分解','',
        '可用增益 = 排序损失 + 有符号保留代价 + 实际增益。排序损失由相同模型在零门限下计算；',
        '保留代价 = 零门限实际增益 − 正式门限实际增益。负值表示门限阻止了损害，不能一概称为门限过严。',
        '无有效观察时零门限也保留原答案，这种未实现的能力不计为门限代价。所有 oracle 仅作有限候选诊断。','',
        '| 数据 | 模型 | 任务 | 可用增益 pp | 排序损失 pp | 保留代价 pp | 实际增益 pp |',
        '|---|---|---|---:|---:|---:|---:|']
    for split,models in audit['outputs'].items():
        for arm in ('full','query_blind','permuted','unregularized'):
            for task,r in models[arm].items():
                d=r['decomposition'];lines.append('| '+split+' | '+arm+' | '+task+' | '+' | '.join(
                    f'{100*d[k]:+.4f}' for k in ('available_gain','ranking_regret','retention_cost','actual_gain'))+' |')
    reference=read(registry['reference_summary'])['macro'];paths.append(Path(registry['reference_summary']))
    lines += ['', '## 五输出覆盖与下一步边界','',
        '| 输出 | 本轮状态 |','|---|---|',
        '| LERF semantic 2D | 新机制迁移待完成；原参考 '+f"{100*reference['lerf']['semantic_iou']:.4f}% 不作本轮实测 |",
        '| LERF 独立选集二维投影 | 新机制迁移待完成，不能称真 3D GT IoU |',
        '| NVOS | 每视角单目标区域不能提供当前签名要求的区域竞争证据，需合法的参考条件化观察接口 |',
        '| ScanNet OVS | 本轮 10 场景、完整菜单三个输出实测；所有控制共用候选 |',
        '| UQIS | 9 场景、268 查询实测；文本／图像干预，真实点原答案保留 |','',
        '结论以开发集及查询对照为准：本轮没有闭合查询证据贡献，不晋升默认。',
        '约束确实限制了置乱修正，但这不证明它能识别真实候选的修复／损害。后续优先检验观察证据的可辨识性和跨载体适用性，',
        '并补齐 LERF/NVOS 的合法观察及完整答案接口；不继续在已打开 benchmark 上扫描门限或加树。',
        '类别候选上界很小、mesh/Gaussian 域有差异、校准缺 6 类、AP 没有直接效用监督仍是独立缺口。',
        '本文仍是工作稿；完整统一迁移、查询特异增益和独立测试均未完成。','',
        '## 复现与存储','',
        f"本轮五条件稠密预测峰值 {read(root/'predictions_complete.json')['combined_transient_prediction_bytes']:,} bytes，"
        f"已释放：{read(root/'ram_released.json')['all_dense_arrays_released']}；持久稠密预测为 0。",'',
        '运行依赖原教师和受保护地图资产；封存源码、模型、配置、输入哈希和紧凑指标齐全，重算必须使用新运行目录。','']
    feedback['receipts']={str(p):sha(p) for p in sorted(set(paths))}
    (paper/'residual_utility_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
    Path('docs/experiments/2026-10-16-tmm-residual-utility.md').write_text('\n'.join(lines)+'\n')
    print('residual utility report complete; development gate:',feedback['development_gate'])


if __name__=='__main__':main()
