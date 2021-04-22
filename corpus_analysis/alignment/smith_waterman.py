#adopted from https://tiefenauer.github.io/blog/smith-waterman/
from itertools import product
import numpy as np

def matrix(a, b, match_score, gap_cost, ignore):
    H = np.zeros((len(a) + 1, len(b) + 1), np.int)
    for i, j in product(range(1, H.shape[0]), range(1, H.shape[1])):
        sign = 1 if (a[i-1] == b[j-1] or a[i-1] in ignore or b[j-1] in ignore) else -1
        match = H[i-1, j-1] + (sign * match_score)
        delete = H[i-1, j] - gap_cost
        insert = H[i, j-1] - gap_cost
        H[i, j] = max(match, delete, insert, 0)
    return H

def traceback(H):
    # flip H to get index of **last** occurrence of H.max() with np.argmax()
    H_flip = np.flip(np.flip(H, 0), 1)
    i_, j_ = np.unravel_index(H_flip.argmax(), H_flip.shape)
    i, j = np.subtract(H.shape, (i_ + 1, j_ + 1))  # (i, j) are **last** indexes of H.max()
    if H[i, j] == 0: return []
    return traceback(H[0:i, 0:j]) + [[i-1, j-1]]

def smith_waterman(a, b, match_score=3, gap_cost=5, ignore=[]):
    H = matrix(a, b, match_score, gap_cost, ignore)
    return traceback(H), H

#print(smith_waterman([3,4,6,7],[3,4,5,6,7]))