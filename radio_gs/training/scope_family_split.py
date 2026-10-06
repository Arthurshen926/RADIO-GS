"""Conservative validation families; connected evidence is never split.

Families are contamination groups, not claims of physical instance identity.
Over-grouping reduces usable validation and must be reported, not fixed by
cutting inconvenient edges after looking at validation scores.
"""

def scope_family_split(query_ids, edges, folds=4):
    edges=list(edges)
    ids=sorted(set(map(int,query_ids)))
    if folds<2:raise ValueError('At least two folds required')
    parent={i:i for i in ids}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in edges:
        if a not in parent or b not in parent:raise ValueError('Unknown query endpoint')
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    components={}
    for i in ids:components.setdefault(find(i),[]).append(i)
    families=sorted(components.values(),key=lambda x:(-len(x),x[0]));assigned=[[] for _ in range(folds)];sizes=[0]*folds
    for family in families:
        fold=min(range(folds),key=lambda f:(sizes[f],f));assigned[fold].extend(family);sizes[fold]+=len(family)
    membership={q:f for f,qs in enumerate(assigned) for q in qs}
    assert all(membership[a]==membership[b] for a,b in edges)
    # A count of tiny components must not disguise one family containing almost
    # the entire dataset. Keep that family intact and flag the unusable split.
    enough_families=len(families)>=folds
    dominated=bool(ids) and max(sizes)>len(ids)/2
    return dict(families=families,folds=[sorted(qs) for qs in assigned],
                usable=enough_families and not dominated,
                enough_families=enough_families,dominated_by_one_fold=dominated,
                query_to_fold=membership)
