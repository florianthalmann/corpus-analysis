import math
import numpy as np
from sklearn.metrics import pairwise_distances
from .util import median_filter, symmetric

def fill_gaps(a, gap_size, gap_ratio):
    diffs = np.diff(a)
    gaps = np.nonzero(diffs)[0]+1
    segs = np.split(a, gaps)
    return np.concatenate([np.repeat(1, len(s))
        if i > 0 and i < len(segs)-1 and s[0] == 0 and len(s) <= gap_size
            and len(s)/(len(segs[i-1])+len(s)+len(segs[i+1])) <= gap_ratio
        else s
        for i,s in enumerate(segs)])

def smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio):
    func = lambda d: median_filter(d, max_gaps)
    #func = lambda d: fill_gaps(d, max_gaps, max_gap_ratio)
    diagonals = to_diagonals(matrix)
    if symmetric: #only smooth upper triangle
        smoothed = [func(d) if i >= matrix.shape[0] else d
            for i, d in enumerate(diagonals)]
    else:
        smoothed = [func(d) for d in diagonals]
    return from_diagonals(smoothed, matrix.shape)

def to_diagonals(A):
    return [A.diagonal(i).T for i in range(-A.shape[0]+1, A.shape[1])]

def from_diagonals(ds, shape):
    size = min(shape)
    ds = np.vstack([np.resize(d, size) for d in ds])
    dia_indices = np.flip(np.arange(shape[0]))[:, None] + np.arange(shape[1])
    ele_indices = np.minimum(np.tile(np.arange(shape[1]), (shape[0], 1)),
        np.tile(np.arange(shape[0])[:,None], (1, shape[1])))
    return ds[dia_indices, ele_indices]

def get_equality(a, b):
    if len(a.shape) > 1:
        return np.all(a[:, None] == b[None, :], axis=2).astype(int)
    return a[:, None] == b[None, :]

def get_affinity_matrix(a, b, equality, max_gaps, max_gap_ratio, K_FACTOR=10):
    symmetric = np.array_equal(a, b)
    #create affinity or equality matrix
    if equality:
        matrix = get_equality(a, b)
    else:
        matrix = 1-pairwise_distances(a, b, metric="cosine")
        k = 1+K_FACTOR*int(math.log(len(matrix), 2))
        conns = np.zeros(matrix.shape)
        knn = [np.argpartition(m, -k)[-k:] for m in matrix]
        for i,k in enumerate(knn):
            conns[i][k] = 1
        matrix = conns
    unsmoothed = matrix
    #only keep upper triangle in symmetric case
    if symmetric: matrix = np.triu(matrix, k=1)
    #smooth with a median filter
    if max_gaps > 0:
        matrix = smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio)
    return matrix, unsmoothed

#returns a list of arrays of index pairs
def get_diagonal_indices(A):
    return to_diagonals(np.dstack(np.indices(A.shape)))

def split_at_gaps(path):
    diffs = np.diff(path, axis=0)[:,0]
    gaps = np.nonzero(diffs > 1)[0]+1
    return np.split(path, gaps)

def matrix_to_segments(A):
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

def get_min_dist(s, refs, ref_locs, ref_starts, ref_ends):
    dists = np.abs(ref_locs-(s[0][1]-s[0][0]))[
        np.where(np.logical_and(s[-1][0]-ref_starts > 0, ref_ends-s[0][0] > 0))]
    return np.min(dists) if len(refs) > 0 and len(dists) > 0 else 1

#sort by length and minimum distance from refs
def sort_segments(segments, refs):
    ref_locs = np.array([r[0][1]-r[0][0] for r in refs])
    ref_starts = np.array([r[0][0] for r in refs])
    ref_ends = np.array([r[-1][0] for r in refs])
    #min distance from ref segments that overlap with s
    min_dist = lambda s: get_min_dist(s, refs, ref_locs, ref_starts, ref_ends)
    return sorted(segments, key=lambda s: len(s)*min_dist(s), reverse=True)

#removes the area of width 2*padding+1 around the given ref and rearranges
def remove_filter_and_sort(segments, refs, padding, min_len):
    #remove overlaps with areas around last added ref segment
    if len(refs) > 0 and len(refs[-1]) > 0:
        area = get_padded_area(refs[-1], padding)
        segments = [d for s in segments for d in difference2(s, area)]
    #remove short segments
    segments = [s for s in segments if len(s) >= min_len]
    return sort_segments(segments, refs)

def filter_segments(segments, count, min_len, min_dist, symmetric, shape):
    padding = max(min_dist-1, 0)
    segments = sort_segments(segments, [])
    selected = []
    #remove area around diagonal if symmetric
    if symmetric: 
        selected.append(np.dstack((np.arange(shape[0]), np.arange(shape[0])))[0])
        count = count+1 if count > 0 else count
    diapad = max(padding, min_len-1)#too close to diagonal means small transl vecs
    remaining = remove_filter_and_sort(segments, selected, diapad, min_len)
    if 0 < count < len(remaining):
        #iteratively take longest segment and remove area around it
        while len(selected) < count and len(remaining) > 0:
            selected.append(remaining.pop(0))
            remaining = remove_filter_and_sort(remaining, selected, padding, min_len)
    else: selected += remaining
    #print([(len(s), s[0][1]-s[0][0]) for s in selected])
    return selected[1:] if symmetric else selected #remove diagonal if symmetric

def get_alignment_segments(a, b, count, min_len, min_dist, max_gap_size, max_gap_ratio):
    symmetric = np.array_equal(a, b)
    equality = issubclass(a.dtype.type, np.integer)
    matrix, unsmoothed = get_affinity_matrix(a, b, equality, max_gap_size, max_gap_ratio)
    if symmetric: matrix = np.triu(matrix)
    segments = matrix_to_segments(matrix)
    #keep only segments longer than min_len and with a gap ratio below max_gap_ratio
    segments = [s for s in segments if len(s) >= min_len]
    if max_gap_size > 0:
        segments = [s for s in segments
            if np.sum(unsmoothed[tuple(s.T)]) >= (1-max_gap_ratio)*len(s)]
    return filter_segments(segments, count, min_len, min_dist, symmetric, matrix.shape)

def segments_to_matrix(segments, shape=None, sum=False):
    points = np.concatenate(segments)
    if not shape: shape = tuple(np.max(points, axis=0)+1)
    matrix = np.zeros(shape)
    if sum:
        np.add.at(matrix, tuple(points.T), 1)
    else:
        matrix[points.T[0], points.T[1]] = 1
    return matrix

def get_alignment_matrix(a, b, min_len, min_dist, max_gap_size, max_gap_ratio):
    segments = get_alignment_segments(a, b, min_len, min_dist, max_gap_size, max_gap_ratio)
    return segments_to_matrix(segments, (len(a), len(b)))