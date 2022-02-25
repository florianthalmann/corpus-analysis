from math import ceil, sqrt, log
from itertools import zip_longest
import numpy as np
import librosa
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d, find_peaks
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from .util import median_filter, symmetric, strided, strided2D
from ..util import plot_matrix, plot_hist
from ..structure.novelty import peak_picking_MSAF

def fill_gaps(a, gap_size, gap_ratio):
    diffs = np.diff(a)
    gaps = np.nonzero(diffs)[0]+1
    segs = np.split(a, gaps)
    return np.concatenate([np.repeat(1, len(s))
        if i > 0 and i < len(segs)-1 and s[0] == 0 and len(s) <= gap_size
            and len(s)/(len(segs[i-1])+len(s)+len(segs[i+1])) <= gap_ratio
        else s
        for i,s in enumerate(segs)])

#smooth with a median filter
def smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio=None):
    func = lambda d: median_filter(d, max_gaps)
    #func = lambda d: fill_gaps(d, max_gaps, max_gap_ratio)
    diagonals = to_diagonals(matrix)
    if symmetric: #only smooth upper triangle
        smoothed = [func(d) if i >= matrix.shape[0] else d
            for i, d in enumerate(diagonals)]
    else:
        smoothed = [func(d) for d in diagonals]
    return from_diagonals(smoothed, matrix.shape)

#resmoothing sum keeps beginnings followed by gaps
def double_smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio):
    unsmoothed = matrix
    if max_gaps > 0:
        matrix = smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio)
        matrix = smooth_matrix(np.logical_or(matrix, unsmoothed),
            symmetric, max_gaps, max_gap_ratio)
    return matrix

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

def k_factor(shape, emphasis, width=1):#used strength 10 before....
    #k = 1+k_factor*int(log(len(matrix), 2))
    return round(2 * ceil(sqrt(((shape[0]+shape[1])/2*emphasis) - 2 * width + 1)))

def knn_threshold(matrix, emphasis):
    k = k_factor(matrix.shape, emphasis)
    conns = np.zeros(matrix.shape)
    knn = [np.argpartition(m, -k)[-k:] for m in matrix]
    for i,k in enumerate(knn):
        conns[i][k] = 1
    return conns

def peak_threshold(matrix, median_len=16, sigma=0.25):
    result = np.zeros(matrix.shape)
    for i,r in enumerate(matrix):
        #result[i][peak_picking_MSAF(r, median_len=median_len, sigma=sigma)[0].astype(int)] = 1
        result[i][find_peaks(r, height=sigma)[0].astype(int)] = 1
        #result[i][libfmp.c6.peak_picking_roeder(r)[0].astype(int)] = 1
        #result[i][librosa.util.peak_pick(r, pre_max=5, post_max=5, pre_avg=5, post_avg=5, delta=0.01, wait=5)] = 1
    return result

#if factor <= 10, use knn, otherwise percentile
def threshold_matrix(matrix, threshold):
    if threshold == 0:
        return peak_threshold(matrix)
    elif threshold <= 10:
        return knn_threshold(matrix, threshold)
        #plot_matrix(matrix, 'est-.png')
    else:
        matrix = MinMaxScaler().fit_transform(matrix)
        matrix[matrix < np.percentile(matrix, threshold)] = 0
        matrix[matrix != 0] = 1
        # k = k_factor(matrix.shape, emphasis) * len(matrix) #really??
        # thresh = np.partition(matrix.flatten(), -k)[-k]
        # matrix = np.where(matrix >= thresh, 1, 0)
        #plot_matrix(matrix, 'est-.png')
        return matrix

def averages(a, window_length):
    windows = strided(a, window_length)
    # mean = np.mean(windows, axis=1)
    # return mean / (np.std(windows, axis=1)/mean)
    #return 1 / np.std(windows, axis=1)
    return np.mean(windows, axis=1)

#sliding averages: position x window size
def avg_matrix(a, min, max, matrix):
    d = matrix[tuple(a.T)]
    size = np.min(list(matrix.shape))#max([len(i) for i in diagonals])
    m = np.zeros((size, max-min+1))
    for l in range(min, max+1):
        if l <= len(a):
            avs = averages(d, l) #/
            if avs is not None:
                m[:,l-min] = np.pad(avs, (0,size-len(avs)))# * l**0.01
    return m

def convolve(matrix, kernel):
    #convolve, only keep where kernel fully contained, and convert to diagonals
    c = to_diagonals(convolve2d(matrix, kernel, mode='valid'))
    #zip and pad into a matrix (diagonal index x position) 0 if no value
    k = len(kernel)-1
    return np.pad(np.array(list(zip_longest(*c, fillvalue=0))).T, ((k,k),(0,k)))

def avg_filter(matrix, size):
    #return convolve(matrix, np.ones((size,size))/size**2)
    return convolve(matrix, (np.ones((size,size))-np.identity(size))/(size**2-size))

def dia_avg_filter(matrix, size):
    return convolve(matrix, np.identity(size)/size)

def ratings(matrix, min_size, max_size):
    dav = np.dstack([dia_avg_filter(matrix, s) for s in range(min_size, max_size+1)])
    avg = np.dstack([avg_filter(matrix, s) for s in range(min_size, max_size+1)])
    #print(dav)
    #print(avg)
    return np.divide(dav, avg, out=np.zeros(dav.shape), where=avg!=0)
    #return dav-avg
    #return dav

def ssm(a, b):
    return 1-pairwise_distances(a, b, metric="cosine")

def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return None

def avgs1(diagonals, min_len, max_len, matrix):
    #avgs = ratings(matrix, min_len, max_len)
    return np.stack([avg_matrix(d, min_len, max_len, matrix) for d in diagonals])

def avgs2(diagonals, min_len, max_len, matrix, len_emph):
    diagonals = [matrix[tuple(d.T)] for d in diagonals]
    diagonals = np.column_stack((zip_longest(*diagonals, fillvalue=0)))
    result = []
    for l in range(min_len, max_len+1):
        windows = np.concatenate(strided2D(diagonals, l))
        avgs = np.reshape(np.mean(windows, axis=1), (diagonals.shape[0],-1))
        avgs *= l**len_emph
        mask = np.tril(np.ones(avgs.shape), -l+1)
        mask = np.logical_and(mask, np.flip(mask, axis=0))
        result.append(np.pad(avgs*mask, ((0,0),(0,l-min_len))))
    return np.dstack(result)

#new method for unthresholded unsmoothed matrix!
def get_best_segments(matrix, min_len=20, max_len=44, min_dist=1, threshold=0,#99.5,
        len_emph=0, min_val=.6, ignore_overlaps=False, max_gap_len=5):
    #min_len=2
    diagonals = get_diagonal_indices(matrix)
    
    #avgs is organized as: (diagonal index, position, length)
    #avgs = ratings(matrix, min_len, max_len)
    #avgs = avgs1(diagonals, min_len, max_len, matrix)
    avgs = avgs2(diagonals, min_len, max_len, matrix, len_emph)
    
    # #TODO: divide by medians of adjacent lengths not same lengths (square)
    # #divide by local environment (avg relative to surrounding diagonals..)
    # envs = np.dstack([np.dstack([median_filter(avgs[:,i,j], j+min_len)
    #     for i in range(avgs.shape[1])])[0] for j in range(avgs.shape[2])])
    # avgs = np.divide(avgs, envs, out=np.zeros(avgs.shape), where=envs!=0)
    # print(avgs.shape)
    # print('done')
    
    flatavgs = np.reshape(avgs, -1)
    maxorder = np.flip(np.argsort(flatavgs))#order of max avgs
    maxpos = np.argsort(maxorder)
    maxavgs = flatavgs[maxorder]
    maxindices = np.transpose(np.unravel_index(maxorder, avgs.shape))#indices of maxes in original
    ignored = np.zeros(avgs.shape)#ignored avgs, in original shape
    maxignored = np.reshape(ignored, -1)[maxorder]#ignored, in order of maxes
    best = []
    i = 0
    #total = 0.02*matrix.shape[0]**2
    threshold = np.percentile(avgs, threshold)#0
    while maxavgs[i] > max(threshold, min_val):# and sum([b[2]+min_len for b in best]) < total:#len(best) < N:
        #print(i, maxavgs[i], maxindices[i], maxignored[i])
        b = maxindices[i]
        best.append(b)
        
        if ignore_overlaps:#now ignore all ratings that overlap with chosen
            l = b[2]+min_len
            ixs = [np.mgrid[
                    max(b[0]-min_dist+1, 0) : min(b[0]+min_dist, avgs.shape[0]),
                    max(b[1]-(min_len+k)+1, 0) : min(b[1]+l, avgs.shape[1]),
                    k:k+1]
                for k in range(max_len-min_len+1)]
            ixs = np.hstack([np.ravel_multi_index(x, ignored.shape) for x in ixs])
            maxignored[maxpos[ixs]] = 1
        else:#ignore ratings that are contained by chosen
            l = b[2]+min_len
            ixs = [np.mgrid[
                    max(b[0]-min_dist+1, 0) : min(b[0]+min_dist, avgs.shape[0]),
                    max(b[1], 0) : min(b[1]+l-(min_len+k)+1, avgs.shape[1]),
                    k:k+1]
                for k in range(b[2]+1)]
            ixs = np.hstack([np.ravel_multi_index(x, ignored.shape) for x in ixs])
            maxignored[maxpos[ixs]] = 1
        
        i += index(maxignored[i:], 0)[0]#np.argmax(maxignored[i:]==0)
        
    #print(len(best))
    #print(indices)
    #print([[matrix[tuple(ii)] for ii in i] for i in indices])
    segs = [diagonals[b[0]][b[1]:b[1]+b[2]+min_len] for b in best]
    segs = [remove_outer_gaps(s, matrix) for s in segs]
    segs = [s for s in segs if len(s) >= min_len]
    if max_gap_len > 0:
        segs = [s for s in segs if max_gap_length(s, matrix) <= max_gap_len]
    #print(sum([len(s) for s in segs]), matrix.shape[0]**2)
    return segments_to_matrix(segs, matrix.shape)


def get_affinity_matrix(a, b, equality, max_gaps, max_gap_ratio, threshold=1):
    symmetric = np.array_equal(a, b)
    #create affinity or equality matrix
    if equality:
        matrix = get_equality(a, b)
    else:
        matrix = 1-pairwise_distances(a, b, metric="cosine")
        matrix = threshold_matrix(matrix, threshold)
    #only keep upper triangle in symmetric case
    if symmetric: matrix = np.triu(matrix, k=1)
    
    smoothed = double_smooth_matrix(matrix, symmetric, max_gaps, max_gap_ratio)
    #plot_matrix(matrix, 'est-3.png')
    return smoothed, matrix

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

def max_gap_length(segment, unsmoothed):
    gaps = np.nonzero(1 - unsmoothed[tuple(segment.T)])[0]
    return max([len(c) for c in consecutive(gaps)])

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def remove_outer_gaps(segment, unsmoothed):
    true_matches = np.nonzero(unsmoothed[tuple(segment.T)])[0]
    #print(true_matches, segment)
    if len(true_matches) > 1:
        return segment[true_matches[0]:true_matches[-1]+1]
    return []

def get_segments_from_matrix(matrix, symmetric, count, min_len, min_dist, max_gap_size, max_gap_ratio, unsmoothed=None):
    if symmetric: matrix = np.triu(matrix)
    segments = matrix_to_segments(matrix)
    #remove gaps at beginning and end
    if unsmoothed is not None:
        segments = [remove_outer_gaps(s, unsmoothed) for s in segments]
    #keep only segments longer than min_len and with a gap ratio below max_gap_ratio
    segments = [s for s in segments if len(s) >= min_len]
    if max_gap_size > 0 and max_gap_ratio > 0 and unsmoothed is not None:
        segments = [s for s in segments
            if np.sum(unsmoothed[tuple(s.T)]) >= (1-max_gap_ratio)*len(s)]
    return filter_segments(segments, count, min_len, min_dist, symmetric, matrix.shape)

def get_alignment_segments(a, b, count, min_len, min_dist, max_gap_size, max_gap_ratio):#, k_factor=10):
    symmetric = np.array_equal(a, b)
    equality = issubclass(a.dtype.type, np.integer)
    matrix, unsmoothed = get_affinity_matrix(a, b, equality, max_gap_size, max_gap_ratio)
    return get_segments_from_matrix(matrix, symmetric, count, min_len, min_dist, max_gap_size, max_gap_ratio, unsmoothed)

def get_longest_segment(a, b, count, min_len, min_dist, max_gap_size, max_gap_ratio):#, k_factor=10):
    symmetric = np.array_equal(a, b)
    equality = issubclass(a.dtype.type, np.integer)
    matrix, unsmoothed = get_affinity_matrix(a, b, equality, max_gap_size, max_gap_ratio)
    segments = matrix_to_segments(matrix)
    return segments[np.argmax([len(s) for s in segments])]

def segments_to_matrix(segments, shape=None, sum=False):
    if len(segments) > 0:
        points = np.concatenate(segments)
        if not shape: shape = tuple(np.max(points, axis=0)+1)
        matrix = np.zeros(shape)
        if sum:
            np.add.at(matrix, tuple(points.T), 1)
        else:
            matrix[points.T[0], points.T[1]] = 1
        return matrix
    elif shape:
        return np.zeros(shape)

def get_alignment_matrix(a, b, count, min_len, min_dist, max_gap_size, max_gap_ratio):
    segments = get_alignment_segments(a, b, count, min_len, min_dist, max_gap_size, max_gap_ratio)
    return segments_to_matrix(segments, (len(a), len(b)))

#get_best_segments([[1,2],[3,3],[3,4],[3,1],[5,0]], [[4,1],[2,2],[2,3],[3,4],[4,5],[5,6]], 2, 3)
#get_best_segments(ssm([[0,0],[1,0],[2,1],[3,2],[5,0]], [[1,0],[2,1],[3,2],[4,0],[4,5],[5,6]]), 2, 3, threshold=50)