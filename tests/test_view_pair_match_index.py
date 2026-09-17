from radio_gs.scripts.compile_three_query_episodes import matches_by_view_pair


def test_view_pair_index_preserves_exact_candidates_and_order():
    views=[0,1,0,2,1,2]
    matches=[(0,1,.9),(2,4,.8),(1,0,.9),(3,5,.7),(0,3,.6),(0,1,.5),(4,2,.8)]
    grouped=matches_by_view_pair(matches,views)
    for q,t,_ in matches:
        old=[m for m in matches if views[m[0]]==views[q] and views[m[1]]==views[t] and m[0]!=q]
        new=[m for m in grouped[(views[q],views[t])] if m[0]!=q]
        assert new==old
    assert matches_by_view_pair([],[])=={}
