import math
from functools import reduce
from collections import OrderedDict
import numpy as np
import sortednp as snp
from patterns import Pattern, segments_to_patterns, patterns_to_segments
from util import argmax

# returns the min dist between p and any parents of p in patterns
def min_dist_from_parents(p, patterns):
    parents = [q for q in patterns if q.contains(p)]
    if len(parents) > 0:
        return min([p.distance(q) for q in parents])
    return math.inf

# filter and sort list of patterns based on given params
def filter_and_sort_patterns(patterns, min_len=0, min_dist=0, parents=[]):
    #filter patterns that are too short or too close to parents
    min_dists = [min_dist_from_parents(p, parents) for p in patterns]
    filtered = [p for i,p in enumerate(patterns)
        if p.l >= min_len and min_dists[i] >= min_dist]
    #sort by position and smallest vector
    secondary = sorted(filtered, key=lambda p: (p.p, min(p.t)))
    #reverse sort by min(dist from parents, length)
    return sorted(secondary, key=
        lambda p: min(min_dists[patterns.index(p)], p.l), reverse=True)

# removes any pattern overlaps, starting with longest pattern,
# adjusting shorter ones to fit within limits
def remove_overlaps(patterns, min_len, min_dist):
    result = []
    patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result)
    while len(patterns) > 0:
        next = patterns.pop(0)
        result.append(next)
        new_boundaries = next.to_boundaries()
        for b in new_boundaries:
            patterns = [q for p in patterns for q in p.divide_at_absolute(b)]
        patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result)
    print(result)
    return result

def remove_contradictions(patterns, min_dist):#possible before transitivity??
    return

def add_transitivity(patterns):
    patterns = filter_and_sort_patterns(patterns)
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            #find absolute positions of p in q and add translations of q to p
            pos = [q.p+r for r in q.internal_positions(p)]
            new_t = [p+t for t in q.t for p in pos]
            #if len(new_t) > 0: print(q)
            p.add_new_translations(new_t)
    return list(OrderedDict.fromkeys(patterns))

def filter_out_dense_infreq(numbers, min_dist, freqs):
    #groups adjacent numbers if within min_dist
    areas = reduce(lambda s,t: s+[[t]] if (len(s) == 0 or t-s[-1][-1] > min_dist)
        else s[:-1]+[s[-1]+[t]], numbers, [])
    #keep only highest frequency number in each area
    return np.array([a[argmax([freqs[t] for t in a])] for a in areas])

def remove_dense_areas(patterns, min_dist=1):
    translations = np.concatenate([p.t for p in patterns])
    unique, counts = np.unique(translations, return_counts=True)
    freqs = dict(zip(unique, counts))
    #keep only most common vector in dense areas within pattern
    for p in patterns:
        p.t = filter_out_dense_infreq(p.t, min_dist, freqs)
    #delete occurrences in dense areas between patterns
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            if q.first_occ_overlaps(p):
                t_union = np.unique(snp.merge(p.t, q.t))
                sparse = filter_out_dense_infreq(t_union, min_dist, freqs)
                if not np.array_equal(t_union, sparse):
                    p.t = snp.intersect(p.t, sparse)
                    q.t = snp.intersect(q.t, sparse)
    return filter_and_sort_patterns([p for p in patterns if len(p.t) > 1])#filter out rudiments

def merge_patterns(patterns):
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            if q.first_occ_overlaps(p) and np.array_equal(q.t, p.t): #patterns can be merged
                new_p = min(p.p, q.p)
                q.l = max(p.p+p.l, q.p+q.l) - new_p
                q.p = new_p
                p.p = -1 #mark for deletion
    return filter_and_sort_patterns([p for p in patterns if p.p >= 0]) #filter out marked

def make_hierarchical(segments, min_len, min_dist):
    patterns = segments_to_patterns(segments)
    patterns = add_transitivity(remove_overlaps(patterns, min_len, min_dist))
    print(patterns)
    patterns = remove_dense_areas(patterns, min_dist)
    print(patterns)
    patterns = merge_patterns(patterns)
    print(patterns)
    return patterns_to_segments(patterns)

#print(add_transitivity([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))

