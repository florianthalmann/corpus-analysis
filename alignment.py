import timeit
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.ndimage import median_filter
from librosa.segment import timelag_filter

def get_affinity_matrix(a, b, equality, smoothing=0):
    matrix = np.all(a[:, None] == b[None, :], axis=2).astype(int) if equality \
        else 1-pairwise_distances(a, b, metric="cosine")
    if np.array_equal(a, b): np.fill_diagonal(matrix, 0)
    return timelag_filter(median_filter)(matrix, size=(1, (smoothing*2)+1))

#returns a list of arrays of index pairs
def get_diagonal_indices(A):
    ij = np.dstack(np.indices(A.shape))
    return [ij.diagonal(i).T for i in range(-ij.shape[0]+1, ij.shape[1])]

def extract_alignments(A):
    #A = np.array([[1,1,0],[0,0,1],[1,0,1]])
    #get arrays of nonzero diagonal elements
    paths = [d[A[tuple(d.T)] != 0] for d in get_diagonal_indices(A)]
    paths = [p for p in paths if len(p) > 0]
    #find gaps between diagonal elements
    diffs = [np.diff(p, axis=0)[:,0] for p in paths]
    gaps = [np.nonzero(d > 1)[0]+1 for d in diffs]
    #split paths at gaps, flatten and return
    segments = [np.split(p,g) for p,g in zip(paths, gaps)]
    return [s for segs in segments for s in segs]

def get_alignment(a, b, min_len):
    print("affinity")
    matrix = get_affinity_matrix(np.array(a), np.array(b), True, 0)
    alignments = extract_alignments(matrix)
    print(len(alignments))
    alignments = [a for a in alignments if len(a) >= min_len]
    print(len(alignments))
    points = np.concatenate(alignments)
    print(len(points.T))
    matrix2 = np.zeros(matrix.shape)
    matrix2[points.T[0], points.T[1]] = 1
    #print(timeit.timeit(lambda: extract_alignments(matrix), number=1000))
    return matrix2
