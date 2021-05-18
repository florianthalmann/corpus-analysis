import numpy as np
import sortednp as snp
from ..util import plot_matrix, load_json, profile

def get_meet_matrix(hlabels):
    matrix = np.zeros((hlabels.shape[1], hlabels.shape[1]), dtype=int)
    for level, labels in enumerate(hlabels):#.T):
        same_label = np.triu(np.equal.outer(labels, labels), k=1)
        matrix[np.where(same_label)] = level
    return matrix

def to_triples(a):
    return a.view(dtype=np.dtype([('x',a.dtype),('y',a.dtype),('z',a.dtype)]))[:,0]

def get_meet_triples(hlabels):
    meets = get_meet_matrix(hlabels)
    #plot_matrix(meets)
    comp_sets = [meets[i,:][:,None] > meets[i,:] for i in range(meets.shape[0])]
    triples = np.concatenate([np.insert(np.nonzero(c), 0, i, axis=0).T
        for i,c in enumerate(comp_sets)])
    triples = triples[triples[:,0] != triples[:,2]]
    return to_triples(triples).tolist()

def lmeasure(reference, estimate):
    print("t1")
    reftriples = get_meet_triples(reference)
    print(type(reftriples))
    print("t2")
    esttriples = get_meet_triples(estimate)
    print("sets")
    reftriples, esttriples = set(reftriples), set(esttriples)
    print("isect")
    intersection = reftriples.intersection(esttriples)#snp.intersect(reftriples, esttriples)#set(reftriples).intersection(set(esttriples))
    print("rest")
    if len(intersection) > 0:
        precision = len(intersection) / len(esttriples)
        recall = len(intersection) / len(reftriples)
        hmean = 2 / (1/precision + 1/recall)
        print(precision, recall, hmean)
        return precision, recall, hmean
    return 0, 0, 0

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

def test():
    ref_i = [[[0, 30], [30, 60]], [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]]]
    ref_l = [['A', 'A'], ['a', 'b', 'c', 'a', 'b', 'd']]
    est_i = [[[0, 30], [30, 60]], [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50], [50, 60]]]
    est_l = [['A', 'A'], ['a', 'a', 'a', 'a', 'a', 'b']]
    ref = np.vstack((np.repeat(0, 60),
        np.concatenate((np.repeat(1, 10), np.repeat(2, 10), np.repeat(3, 10),
            np.repeat(1, 10), np.repeat(2, 10), np.repeat(4, 10)))))
    est = np.vstack((np.repeat(0, 60),
        np.concatenate((np.repeat(1, 50), np.repeat(2, 10)))))
    print(lmeasure(ref, est))

# get_relative_meet_triples(np.array([[271,0,0,0],[271,201,182,1],[271,201,182,2],
# [271,201,182,3],[271,201,201,4],[271,201,201,5],[271,201,201,6],[271,201,201,7],
# [271,201,201,8],[271,201,201,9],[271,201,182,1],[271,201,182,2],[271,201,182,3],
# [271,201,201,4],[271,201,201,5],[271,201,201,6],[271,201,201,7],[271,201,201,8],
# [271,201,201,9],[271,201,182,1]]))
# print(lmeasure(np.array([[0,0,0,0,0,0], [1,2,3,1,2,4]]),
#     np.array([[0,0,0,0,0,0], [1,1,1,1,1,2]])))
# 
# print(lmeasure(np.array([[0,0,0,0,0,0], [1,2,3,1,2,4]]),
#     np.array([[0,0,0,0,0,0], [1,2,3,1,2,4]])))
# print(lmeasure(np.array([[0,0,0,0,0,0], [1,2,3,1,2,4]]),
#     np.array([[0,0,0,0,0,0]])))
#profile(lambda: lmeasure(np.array(load_json('hlabels.json')), np.array(load_json('g0labels.json'))))
#test()