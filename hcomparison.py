import numpy as np

def meet(v,w):
    same_labels = np.where(np.logical_and(v == w, v != -1, w != -1))[0]
    if len(same_labels) > 0:
        return np.max(same_labels)
    return -1

def get_meet_matrix(hlabels):
    return np.array([[meet(x,y) if j > i else -1 for j,y in enumerate(hlabels)]
        for i,x in enumerate(hlabels)])

def get_meet_triples(hlabels):
    meets = get_meet_matrix(hlabels)
    comp_sets = [meets[i,i+1:][:,None] > meets[i,i+1:] for i in range(meets.shape[0])]
    triples = np.concatenate([np.insert(np.add(np.nonzero(np.triu(c)), i+1), 0, i, axis=0).T
        for i,c in enumerate(comp_sets)])
    return triples

def get_relative_meet_triples(hlabels):
    #print(hlabels[:20])
    #return count too!!!
    triples = get_meet_triples(hlabels)
    #print(len(triples))
    #print(triples[:50])
    return triples

get_relative_meet_triples(np.array([[0,-1,-1],[  1, 182, 201],[  2, 182, 201],
[  3, 182, 201],[  4, 201,  -1],[  5, 201,  -1],[  6, 201,  -1],[  7, 201,  -1],
[  8, 201,  -1],[  9, 201,  -1],[  1, 182, 201],[  2, 182, 201],[  3, 182, 201],
[  4, 201,  -1]]))