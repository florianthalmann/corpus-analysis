import numpy as np
from .util import plot_matrix

def get_meet_matrix(hlabels):
    matrix = np.zeros((len(hlabels), len(hlabels)), dtype=int)
    for level, labels in enumerate(hlabels.T):
        same_label = np.triu(np.equal.outer(labels, labels), k=1)
        matrix[np.where(same_label)] = level
    return matrix

def get_meet_triples(hlabels):
    meets = get_meet_matrix(hlabels)
    #print(len(np.nonzero(meets)[0]))
    plot_matrix(meets)
    comp_sets = [meets[i,i+1:][:,None] > meets[i,i+1:] for i in range(meets.shape[0])]
    triples = np.concatenate([np.insert(np.add(np.nonzero(np.triu(c, k=1)), i+1), 0, i, axis=0).T
        for i,c in enumerate(comp_sets)])
    #print(len(triples))
    return triples

def get_relative_meet_triples(hlabels):
    #print(hlabels[:5])
    #return count too!!!
    triples = get_meet_triples(hlabels)
    #print(triples[:10], len(triples))
    relative = np.subtract(triples.T, triples.T[0]).T
    matrix = np.zeros((len(hlabels), len(hlabels)), dtype=int)
    for r in triples:
        matrix[r[0],r[1]] += 1
    plot_matrix(matrix)
    #unique = list(set([tuple(r) for r in relative]))
    #print(unique[:10], len(unique))
    
    #print(len(triples))
    #print(triples[:50])
    #return triples
    return matrix

# get_relative_meet_triples(np.array([[271,0,0,0],[271,201,182,1],[271,201,182,2],
# [271,201,182,3],[271,201,201,4],[271,201,201,5],[271,201,201,6],[271,201,201,7],
# [271,201,201,8],[271,201,201,9],[271,201,182,1],[271,201,182,2],[271,201,182,3],
# [271,201,201,4],[271,201,201,5],[271,201,201,6],[271,201,201,7],[271,201,201,8],
# [271,201,201,9],[271,201,182,1]]))