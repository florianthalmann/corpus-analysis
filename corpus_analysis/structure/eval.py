import numpy as np
from mir_eval.hierarchy import lmeasure

def get_intervals(levels, grid=None):
    basis = grid if grid is not None else np.arange(len(levels[0]))
    level = np.stack([basis[:-1], basis[1:]]).T
    return np.array([level for l in levels])

def evaluate_hierarchy(reference, refint, estimate, estgrid):
    ref, est = reference, estimate
    estint = get_intervals(est, estgrid)
    return lmeasure(refint, ref, estint, est)

def evaluate_hierarchy_varlen(reference_n_estimate):
    reference, estimate = reference_n_estimate
    sw = np.array(smith_waterman(reference[-1,:], estimate[-1,:])[0])
    ref, est = reference[:,sw[:,0]], estimate[:,sw[:,1]]
    lm = lmeasure(get_intervals(ref), ref, get_intervals(est), est)
    #rethink this multiplication...
    return lm#lm[0]*len(sw)/len(reference[0]), lm[1]*len(sw)/len(estimate[0]), lm[2]