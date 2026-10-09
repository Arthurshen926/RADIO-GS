"""Collect retained NVOS foreground_iou keys and actual signed-function diagnostics."""
from scripts.research.run_query_field_signed_regions import ROOT,PREVIOUS,ARMS,read,write,receipt
from scripts.research.summarize_query_field_relations import paired

def main(root=ROOT,document_prefix='signed-region'):
    ROOT=root
    d=read(ROOT/'design.json');read(ROOT/'scoring_complete.json');ARMS=d['arms'];per={a:{} for a in ARMS};diagnostics={a:{} for a in ARMS};baseline=[]
    for j in d['jobs']:
        evaluation=read(ROOT/'predictions'/j['name']/'evaluation.json')
        prior=read(PREVIOUS/'confidence_mixture_function/predictions/nvos'/j['name']/'evaluation.json')
        if prior['gt_sha256']!=evaluation['gt_sha256']:raise ValueError('retained target changed')
        for row in evaluation['results']:
            arm=row['job'].split('__')[-1]
            per[arm][j['name']]=dict(response_iou=row['retained_response']['foreground_iou'],selection_iou=row['foreground_iou'],
                precision=row['precision'],recall=row['recall'],ap=row['retained_response']['average_precision'])
            prediction=read(ROOT/'predictions'/j['name']/row['job']/'prediction.json')
            diagnostics[arm][j['name']]=dict(diagnostics=prediction['diagnostics'],prompt_agreement=prediction['prompt_agreement'],selected=prediction['selected'])
            if arm=='parent':baseline.append(dict(case=j['name'],response_delta=row['retained_response']['foreground_iou']-prior['results'][0]['retained_response']['foreground_iou'],selection_delta=row['foreground_iou']-prior['results'][0]['foreground_iou']))
    macro={a:{k:sum(v[k] for v in per[a].values())/8 for k in next(iter(per[a].values()))} for a in ARMS}
    comparisons={a+'__'+b:{k:paired({s:v[k] for s,v in per[a].items()},{s:v[k] for s,v in per[b].items()},'NVOS '+k)
        for k in macro[a]} for a,b in [(ARMS[0],ARMS[1]),(ARMS[0],ARMS[2]),(ARMS[1],ARMS[2])]}
    report=dict(complete=True,cases=8,primary=d['primary'],macro=macro,per_case=per,diagnostics=diagnostics,comparisons=comparisons,
        baseline_reproduction=baseline,source_frames=120,source_frame_resolution_unmodified=True,whole_five_task=False,model_promotion=False,
        training_steps=0,extra_state_bytes=0,implementation_commit=d['implementation_commit'],
        collector=receipt(__file__),reporting_repair='retained scorer uses foreground_iou, not iou; no prediction/scoring changes')
    write(ROOT/'summary.json',report);write('docs/experiments/2026-10-09-'+document_prefix+'-results.json',report);print(macro,flush=True)

if __name__=='__main__':main()
