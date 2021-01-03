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
            labels[k] = [-1]+labels[k]
        if ivls[k][-1][1] < target_time:
            ivls[k] = np.append(ivls[k], [[ivls[k][-1][1], target_time]], axis=0)
            labels[k] = labels[k]+[-1]
        elif ivls[k][-1][1] > target_time:
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
    refmax = np.max(np.concatenate(refint))
    #evaluation algo fails if times/framecount not same
    #estint, estlab = adjust_start_end((estint, estlab), refmax)
    estint, estlab = simplify(adjust_start_end((estint, estlab), refmax))
    return lmeasure(refint, reflab, estint, estlab)

def evaluate_hierarchy_varlen(reference_n_estimate):
    reference, estimate = reference_n_estimate
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    #rethink this multiplication...
    return lm#lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]

#see if subdivisions affect lmeasure
def test(frame_size=None):
    ref_i = [[[0, 30], [30, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    ref_l = [['A', 'B'], ['a', 'b', 'a', 'c']]
    est_i = [[[0, 45], [45, 60]], [[0, 15], [15, 30], [30, 45], [45, 60]]]
    est_l = [['A', 'B'], ['a', 'a', 'b', 'b']]
    est2_i = [[[0, 15], [15, 30], [30, 45], [45, 60]], [[0, 5],[5,10],[10,15], [15, 30], [30, 45], [45, 60]]]
    est2_l = [['A', 'A', 'A', 'B'], ['a', 'a', 'a', 'a', 'b', 'b']]

    scores = evaluate(ref_i, ref_l, est_i, est_l)
    print(scores)
    scores = evaluate(ref_i, ref_l, est2_i, est2_l)
    print(scores)

#test()
# test(0.1)
# test(0.5)
# test(1)
#simplify(([np.array([[0,1]]),np.array([[0,0.5],[0.5,0.75],[0.75,1]])], [[0],[0,0,1]]))