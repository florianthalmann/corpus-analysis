import timeit
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from scipy.ndimage import median_filter
from librosa.segment import timelag_filter
from librosa.segment import recurrence_to_lag

def get_affinity_matrix(a, b, equality, smoothing=0):
    matrix = np.all(a[:, None] == b[None, :], axis=2).astype(int) if equality \
        else 1-pairwise_distances(a, b, metric="cosine")
    if np.array_equal(a, b): np.fill_diagonal(matrix, 0)
    return timelag_filter(median_filter)(matrix, size=(1, (smoothing*2)+1))

def get_alignment(a, b):
    matrix = get_affinity_matrix(np.array(a), np.array(b), True, 0)
    print(timeit.timeit(lambda: get_affinity_matrix(np.array(a), np.array(b), True, 0), number=1))
    #print([matrix.diagonal(i) for i in range(-matrix.shape[0]+1,matrix.shape[1])])
    return matrix