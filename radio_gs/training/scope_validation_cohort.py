"""Count heldout scopes with no training evidence as misses, not exclusions."""
import copy


def complete_scope_cohort(rows, frames, group_frames, arms, folds=4):
    output=copy.deepcopy(rows);seen=set()
    for row in output:
        key=(row['frame'],row['group'])
        if key in seen:raise ValueError('Duplicate validation observation')
        seen.add(key)
    expected=set();added=0
    for fold in range(folds):
        heldout=set(frames[fold::folds]);training=set(frames)-heldout
        for group,source_frames in enumerate(group_frames):
            has_evidence=bool(training&set(source_frames))
            for frame in heldout&set(source_frames):
                key=(frame,group);expected.add(key)
                if key in seen:continue
                if has_evidence:raise ValueError('Missing prediction despite available training evidence')
                output.append(dict(frame=frame,fold=fold,group=group,has_training_evidence=False,
                    **{arm:dict(iou=0.,precision=0.,recall=0.,positive_support=0.) for arm in arms}))
                added+=1
    if seen-expected:raise ValueError('Observation outside validation cohort')
    output.sort(key=lambda r:(r['fold'],r['frame'],r['group']))
    return output,added
