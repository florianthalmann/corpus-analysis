import timeit
import numpy as np
from sklearn.metrics import pairwise_distances
from util import median_filter, symmetric

def to_diagonals(A):
    return [A.diagonal(i).T for i in range(-A.shape[0]+1, A.shape[1])]

def from_diagonals(ds, shape):
    size = min(shape)
    ds = np.vstack([np.resize(d, size) for d in ds])
    dia_indices = np.flip(np.arange(shape[0]))[:, None] + np.arange(shape[1])
    ele_indices = np.minimum(np.tile(np.arange(shape[1]), (shape[0], 1)),
        np.tile(np.arange(shape[0])[:,None], (1, shape[1])))
    return ds[dia_indices, ele_indices]

def get_affinity_matrix(a, b, equality, smoothing=0):
    symmetric = np.array_equal(a, b)
    #create affinity or equality matrix
    matrix = np.all(a[:, None] == b[None, :], axis=2).astype(int) if equality \
        else 1-pairwise_distances(a, b, metric="cosine")
    #only keep upper triangle in symmetric case
    if symmetric: matrix = np.triu(matrix, k=1)
    #smooth with a median filter
    if smoothing > 0:
        diagonals = to_diagonals(matrix)
        if symmetric: #only smooth upper triangle
            smoothed = [median_filter(d, smoothing) if i >= len(a) else d
                for i, d in enumerate(diagonals)]
        else:
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
    return [d+a[0] for d in diags]

#returns a list of disjoint subsegments of a with all points in area removed
def difference2(a, area):
    a_offset = a[0][1] - a[0][0]
    area_offsets = [d[0][1] - d[0][0] for d in area]
    if a_offset < area_offsets[0] or area_offsets[-1] < a_offset: return [a]
    matching = area[a_offset-area_offsets[0]]
    if a[-1][0] < matching[0][0] or matching[-1][0] < a[0][0]: return [a]
    result = []
    if a[0][0] < matching[0][0]:
        result.append(a[:matching[0][0]-a[0][0]])
    if matching[-1][0] < a[-1][0]:
        result.append(a[matching[-1][0]-a[0][0]+1:])
    return result

#moves element of l at i before the first element with len < len(l[i])
def move_before_shorter(l, i):
    j = next((k for k in range(i+1, len(l)) if len(a) < len(l[i])), len(l))
    l.insert(j, l.pop(i))

#removes the area of width 2*padding+1 around the given ref and rearranges
def remove_filter_and_sort(segments, ref, padding, min_len):
    #remove overlaps with areas around ref segment
    if len(ref) > 0:
        area = get_padded_area(ref, padding)
        segments = [d for s in segments for d in difference2(s, area)]
        # changed = [i for i in range(len(segments))
        #     if len(nooverlaps[i]) >= min_len
        #     and len(nooverlaps[i]) != len(segments[i])][::-1]
        # [move_before_shorter(nooverlaps, i) for i in changed]
    #remove short segments
    segments = [s for s in segments if len(s) >= min_len]
    segments = sorted(segments, key=lambda s: (len(s), max(s[0]), min(s[0])), reverse=True)
    return segments

def filter_segments(segments, min_len, min_dist, symmetric, shape):
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

def get_alignment_segments(a, b, min_len, min_dist, max_gap_size):
    symmetric = np.array_equal(a, b)
    matrix = get_affinity_matrix(a, b, True, max_gap_size)
    if symmetric: matrix = np.triu(matrix)
    segments = extract_alignment_segments(matrix)
    return filter_segments(segments, min_len, min_dist, symmetric, matrix.shape)

def segments_to_matrix(segments, shape):
    points = np.concatenate(segments)
    matrix = np.zeros(shape)
    matrix[points.T[0], points.T[1]] = 1
    return matrix

def get_alignment_matrix(a, b, min_len, min_dist, max_gap_size):
    segments = get_alignment_segments(a, b, min_len, min_dist, max_gap_size)
    return segments_to_matrix(segments, (len(a), len(b)))
