"""Post-score class error summaries, without changing a prediction or model."""
import numpy as np
from scripts.research.run_query_field_semantic_measure import ROOT,read,write,receipt
from pathlib import Path


def main():
    source=ROOT/'scoring_complete.json';rows=read(source)['results'];classes={}
    for scene in rows:
        for cls,item in scene['metrics']['reliable_typed']['19']['per_class'].items():
            if item['iou'] is None:continue
            group=classes.setdefault(cls,dict(name=item['name'],parent=[],new=[]))
            group['new'].append(item['iou'])
            group['parent'].append(scene['metrics']['parent']['19']['per_class'][cls]['iou'])
    result=[dict(class_id=k,name=v['name'],present_scenes=len(v['new']),
                 parent_iou=float(np.mean(v['parent'])),current_iou=float(np.mean(v['new']))) for k,v in classes.items()]
    result.sort(key=lambda x:x['current_iou'])
    report=dict(aggregation='per-class mean over GT-present scenes;not global scene macro',
                label_use='post-score attribution only;no model selection or modification',classes=result,
                evidence=[receipt(source),receipt(Path(__file__))])
    write(ROOT/'diagnostics/class_attribution.json',report)
    write('docs/experiments/2026-10-09-semantic-measure-class-attribution.json',report)


if __name__=='__main__':main()
