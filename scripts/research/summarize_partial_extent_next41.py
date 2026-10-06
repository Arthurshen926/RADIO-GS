"""Summarize fixed benchmark cohorts separately from assisted source controls."""
import json
from pathlib import Path
from scripts.research.summarize_source_extent_next40 import text_results


if __name__=='__main__':
    base=Path('output/partial_extent_next41')
    result=dict(development=True,promotion=False,
                parent=text_results('output/taskwise_scope_composition_next38',calibrated=True),
                next40=text_results('output/odds_kernel_extent_next40'))
    for mode in ['reselected','conditional','visibility_prior']:result[mode]=text_results(base/mode)
    selections=[json.loads(p.read_text()) for p in sorted((base/'reselected').glob('*/selection.json'))]
    result['ridge_changes']=[dict(key=d['key'],before=d['old_ridge'],after=d['selected']['ridge']) for d in selections if d['old_ridge']!=d['selected']['ridge']]
    result['source_controls']={name:json.loads((base/(name+'.json')).read_text())['macro'] for name in ['reviewed_identity','reviewed_identity_visibility','typed_relations','reviewed_identity_enclosures']}
    result['source_limitations']='Assistant-reviewed source identity/enclosures, teacher pixel masks; leave one source view out of fit, original map saw views. Not benchmark scores or human-certified complete GT.'
    (base/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['macro'] for k,v in result.items() if isinstance(v,dict) and 'macro' in v},indent=2))
    print(json.dumps(result['source_controls'],indent=2))
