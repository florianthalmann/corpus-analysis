import timeit
import numpy as np
from sklearn.metrics import pairwise_distances
from util import median_filter, symmetric

def to_diagonals(A):
    return [A.diagonal(i).T for i in range(-A.shape[0]+1, A.shape[1])]

def from_diagonals(d, shape):
    get_element = lambda i, j: d[j+shape[0]-i-1][min(i,j)]
    return np.fromfunction(np.vectorize(get_element), shape, dtype=int)

def get_affinity_matrix(a, b, equality, smoothing=0):
    #create affinity or equality matrix
    matrix = np.all(a[:, None] == b[None, :], axis=2).astype(int) if equality \
        else 1-pairwise_distances(a, b, metric="cosine")
    #remove main diagonal in symmetric case
    if np.array_equal(a, b): np.fill_diagonal(matrix, 0)
    #smooth with a median filter
    if smoothing > 0:
        diagonals = to_diagonals(matrix)
        smoothed = [median_filter(d, smoothing) for d in diagonals]
        matrix = from_diagonals(smoothed, matrix.shape)
    return matrix

#returns a list of arrays of index pairs
def get_diagonal_indices(A):
    return to_diagonals(np.dstack(np.indices(A.shape)))

def split_at_gaps(path):
    diffs = np.diff(path, axis=0)[:,0]
    gaps = np.nonzero(diffs > 1)[0]+1
    return np.split(path, gaps)

def extract_alignment_segments(A):
    #get arrays of nonzero diagonal elements
    paths = [d[A[tuple(d.T)] != 0] for d in get_diagonal_indices(A)]
    paths = [p for p in paths if len(p) > 0]
    #split at gaps and flatten
    segments = [split_at_gaps(p) for p in paths]
    return [s for segs in segments for s in segs]

def get_padded_area(a, padding):
    size = len(a)
    if size == 0: return a
    pad = min(padding, size-1)
    diags = get_diagonal_indices(np.empty((size,size)))[size-pad-1:size+pad]
    return np.concatenate([d+a[0] for d in diags])

def difference(a, b):
    av = a.view([('', a.dtype)] * a.shape[1]).ravel()
    bv = b.view([('', b.dtype)] * b.shape[1]).ravel()
    return np.setdiff1d(av, bv).view(a.dtype).reshape(-1, a.shape[1])

#moves element of l at i before the first element with len < len(l[i])
def move_before_shorter(l, i):
    j = next((k for k in range(i+1, len(l)) if len(a) < len(l[i])), len(l))
    l.insert(j, l.pop(i))

#removes the area of width 2*padding+1 around the given ref and rearranges
def remove_filter_and_sort(segments, ref, padding, min_len):
    #remove overlaps with areas around ref segment
    if len(ref) > 0:
        area = get_padded_area(ref, padding)
        nooverlaps = [difference(s, area) for s in segments]
        # changed = [i for i in range(len(segments))
        #     if len(nooverlaps[i]) >= min_len
        #     and len(nooverlaps[i]) != len(segments[i])][::-1]
        # print([len(s) for s in nooverlaps])
        # [move_before_shorter(nooverlaps, i) for i in changed]
        # print([len(s) for s in nooverlaps])
        segments = nooverlaps
    #remove short segments
    segments = [s for s in segments if len(s) >= min_len]
    segments = sorted(segments, key=lambda s: (len(s), max(s[0]), min(s[0])), reverse=True)
    return segments

def get_best_segments(segments, min_len, min_dist, symmetric, shape):
    padding = min_dist-1
    segments = [s for s in segments if len(s) >= min_len]
    segments = sorted(segments, key=lambda s: (len(s), max(s[0]), min(s[0])), reverse=True)
    #remove area around diagonal if symmetric
    diagonal = np.dstack((np.arange(shape[0]), np.arange(shape[0])))[0] \
        if symmetric else np.array([])
    remaining = remove_filter_and_sort(segments, diagonal, padding, min_len)
    selected = []
    #iteratively take longest segment and remove overlaps
    while len(remaining) > 0:
        best = remaining.pop(0)
        selected.append(best)
        remaining = remove_filter_and_sort(remaining, best, padding, min_len)
    return selected

def get_alignment(a, b, min_len, min_dist, max_gap_size=10):
    print("affinity")
    print(get_padded_area([[0,0],[1,1],[2,2]], 1))
    matrix = get_affinity_matrix(np.array(a), np.array(b), True, max_gap_size)
    print(symmetric(matrix))
    segments = extract_alignment_segments(matrix)
    print(len(segments))
    segments = get_best_segments(segments, min_len, min_dist, a == b, matrix.shape)
    print(len(segments))
    print([len(s) for s in segments])
    points = np.concatenate(segments)
    print(len(points.T))
    matrix2 = np.zeros(matrix.shape)
    print(symmetric(matrix2))
    matrix2[points.T[0], points.T[1]] = 1
    print(symmetric(matrix2))
    #print(timeit.timeit(lambda: extract_alignments(matrix), number=1000))
    return matrix2
