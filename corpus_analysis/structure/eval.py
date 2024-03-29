import numpy as np
from itertools import groupby, product
from mir_eval.hierarchy import lmeasure, evaluate

#add 0 at beginning, target_time at end
def adjust_start_end(hierarchy, target_time):
    ivls, labels = list(hierarchy[0]), list(hierarchy[1])
    for k in range(len(ivls)):
        if len(ivls[k]) == 0:
            ivls[k], labels[k] = np.array([[0, target_time]]), [0]
        elif ivls[k][0][0] > 0:
            ivls[k] = np.insert(ivls[k], 0, [0, ivls[k][0][0]], axis=0)
            labels[k] = np.insert(labels[k], 0, -1)
        if ivls[k][-1][1] < target_time:
            ivls[k] = np.append(ivls[k], [[ivls[k][-1][1], target_time]], axis=0)
            labels[k] = np.append(labels[k], -1)
        elif max(ivls[k][-1]) > target_time:
            while ivls[k][-1][0] >= target_time:
                ivls[k] = ivls[k][:-1]
                labels[k] = labels[k][:-1]
            ivls[k][-1][-1] = target_time
    return ivls, labels

def simplify(hierarchy):
    levels = list(zip(*hierarchy))
    simplified = []
    for l in levels:
        grouped = [list(g) for k,g in groupby(zip(l[0].tolist(), l[1]),
            lambda s: s[1])]
        simpl = list(zip(*[[[g[0][0][0], g[-1][0][1]], g[0][1]]
            for g in grouped]))
        simplified.append((np.array(simpl[0]), simpl[1]))
    return tuple(zip(*simplified))

def evaluate_hierarchy(refint, reflab, estint, estlab):
    #estint, estlab = estint[:8], estlab[:8]#estint[3:6], estlab[3:6]
    #print('CUT')
    refmax = np.max(np.concatenate(refint))
    #evaluation algo fails if times/framecount not same
    estint, estlab = simplify(adjust_start_end((estint, estlab), refmax))
    try:
        return lmeasure(refint, reflab, estint, estlab)
    except ValueError as e:
        print(e)
        print(refint, reflab)
        print(estint, estlab)

def evaluate_hierarchy_varlen(reference_n_estimate):
    reference, estimate = reference_n_estimate
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    #rethink this multiplication...
    return lm#lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]

#see if subdivisions affect lmeasure
def test(frame_size=None):
    # ref_i = [[[0, 30], [30, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    # ref_l = [['A', 'A'], ['a', 'b', 'a', 'c']]
    # est_i = [[[0, 45], [45, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    # est_l = [['A', 'A'], ['a', 'a', 'b', 'b']]
    # est2_i = [[[0, 15], [15, 30], [30, 45], [45, 60]], [[0, 5],[5,10],[10,15], [15, 30], [30, 45], [45, 60]]]
    # est2_l = [['A', 'A', 'A', 'A'], ['a', 'a', 'a', 'a', 'b', 'b']]
    # 
    # scores = evaluate(ref_i, ref_l, est_i, est_l)
    # print(scores)
    # scores = evaluate(ref_i, ref_l, est2_i, est2_l)
    # print(scores)
    # 
    ref_i = [[[0, 30], [30, 40]], [[0, 10], [10, 20], [20, 30], [30, 40]]]
    ref_l = [['A', 'A'], ['a', 'b', 'c', 'a']]
    est_i = [[[0, 30], [30, 40]], [[0, 10], [10, 20], [20, 30], [30, 40]]]
    est_l = [['A', 'A'], ['a', 'b', 'b', 'b']]
    
    scores = evaluate(ref_i, ref_l, est_i, est_l)
    print(scores)
    
    print()
    
    ref_i = [[[0, 30]], [[0, 10], [10, 20], [20, 30]]]
    est_i = [[[0, 30]], [[0, 10], [10, 20], [20, 30]]]
    ref_l = [ ['A'], ['a', 'b', 'a'] ]
    est_l = [ ['B'], ['a', 'b', 'b'] ]

    scores = evaluate(ref_i, ref_l, est_i, est_l)
    print(scores)

#test()
#simplify(([np.array([[0,1]]),np.array([[0,0.5],[0.5,0.75],[0.75,1]])], [[0],[0,0,1]]))