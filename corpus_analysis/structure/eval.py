import numpy as np
from mir_eval.hierarchy import lmeasure

#add 0 at beginning, target_time at end
def adjust_start_end(hierarchy, target_time):
    ivls, labels = list(hierarchy[0]), list(hierarchy[1])
    for k in range(len(ivls)):
        print(labels[k])
        if len(ivls[k]) == 0:
            ivls[k], labels[k] = np.array([[0, target_time]]), ['0']
        elif ivls[k][0][0] > 0:
            ivls[k] = np.insert(ivls[k], 0, [0, ivls[k][0][0]], axis=0)
            labels[k] = ['-1']+labels[k]
        if ivls[k][-1][1] < target_time:
            ivls[k] = np.append(ivls[k], [[ivls[k][-1][1], target_time]], axis=0)
            labels[k] = labels[k]+['-1']
    return ivls, labels

def evaluate_hierarchy(refint, reflab, estint, estlab):
    refmax = np.max(np.concatenate(refint))
    #evaluation algo fails if times/framecount not same
    estint, estlab = adjust_start_end((estint, estlab), refmax)
    return lmeasure(refint, reflab, estint, estlab)

def evaluate_hierarchy_varlen(reference_n_estimate):
    reference, estimate = reference_n_estimate
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    #rethink this multiplication...
    return lm#lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]