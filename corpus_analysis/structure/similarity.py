import numpy as np
import sortednp as snp
from ..alignment.smith_waterman import smith_waterman


def similarity(match, p1, p2, parsim_diff=None):
    maxlen = max(len(p1), len(p2))
    if parsim_diff: return 1 if abs(match - maxlen) == parsim_diff else 0
    return match / maxlen

def equality(p1, p2, ignore=[]):
    if len(p1) == len(p2):
        return all(p1[i] == p2[i] or p1[i] in ignore or p2[i] in ignore
            for i in range(len(p1)))

def sw_similarity(p1, p2, ignore=[], parsim_diff=None):
    sw = smith_waterman(p1, p2, ignore=ignore)[0]
    return similarity(len(sw), p1, p2, parsim_diff)

#p1 and p2 need to be sorted!!
def isect_similarity(p1, p2, parsim_diff=None):
    if len(p1) == 0 or len(p2) == 0: return 0
    return similarity(len(snp.intersect(p1, p2,
        duplicates=snp.KEEP_MIN_N)), p1, p2, parsim_diff)

def multi_jaccard(p1, p2):
    isect = snp.intersect(p1, p2, duplicates=snp.KEEP_MIN_N)
    #TODO: make KEEP_MAX_N!
    union = snp.merge(p1, p2, duplicates=snp.KEEP_MAX_N)
    return len(isect)/len(union)

#finds the longest overlap between any two occurrences of the patterns
def cooc_similarity(p1, p2, occmat1, occmat2, parsim_diff=None):
    both = np.logical_and(occmat1, occmat2)
    diffs = np.transpose(np.diff(np.where(both == 1)))
    overlaps = np.split(diffs, np.where(diffs != [0,1])[0]+1)
    return similarity(max([len(o) for o in overlaps]), p1, p2, parsim_diff)