import math
from functools import reduce
from collections import OrderedDict, defaultdict
import numpy as np
import sortednp as snp
from patterns import Pattern, segments_to_patterns, patterns_to_segments
from util import argmax, ordered_unique

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

def group_adjacent(numbers, max_dist=1):#groups adjacent numbers if within max_dist
    return np.array(reduce(
        lambda s,t: s+[[t]] if (len(s) == 0 or t-s[-1][-1] > max_dist)
            else s[:-1]+[s[-1]+[t]], numbers, []))

def filter_out_dense_infreq(numbers, min_dist, freqs):
    areas = group_adjacent(numbers, min_dist)
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

def get_most_frequent_pair(sequence, overlapping=False):
    pairs = np.dstack([sequence[:-1], sequence[1:]])[0]
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    if not overlapping:#could be optimized..
        unique = unique[counts > 1]
        counts = [len(get_locs_of_pair(sequence, p)) for p in unique]
    index = np.argmax(counts)
    if counts[index] > 1:
        return unique[index]

def thin_out(a, min_dist=2):
    return np.array(reduce(lambda r,i:
        r+[i] if len(r)==0 or abs(i-r[-1]) >= min_dist else r, a, []))

def get_locs_of_pair(sequence, pair, overlapping=False):
    pairs = np.dstack([sequence[:-1], sequence[1:]])[0]
    indices = np.where(np.all(pairs == pair, axis=1))[0]
    return indices if overlapping else thin_out(indices)

def replace_pairs(sequence, indices, replacement):
    sequence[indices] = replacement
    return np.delete(sequence, indices+1)

def replace_in_tree(tree, element, replacement):
    return [replace_in_tree(t, element, replacement) if type(t) == list
        else replacement if t == element else t for t in tree]

def to_hierarchy(sequence, new_types):
    hierarchy = sequence.tolist()
    for t in reversed(list(new_types.keys())):
        hierarchy = replace_in_tree(hierarchy, t, new_types[t].tolist())
    return hierarchy

def flatten(hierarchy):
    if type(hierarchy) == list:
        return [a for h in hierarchy for a in flatten(h)]
    return [hierarchy]

#groupings on top
def to_labels(sequence, new_types):
    layers = []
    type_lengths = {k:len(flatten((to_hierarchy(np.array([k]), new_types))))
        for k in new_types.keys()}
    print(type_lengths)
    while len(np.intersect1d(sequence, list(new_types.keys()))) > 0:
        layers.append(np.concatenate([np.repeat(s, type_lengths[s])
            if s in new_types else [s] for s in sequence]))
        sequence = np.concatenate([new_types[s]
            if s in new_types else [s] for s in sequence])
    layers.append(sequence)
    return np.dstack(layers)[0]

#leaves at bottom
def to_labels2(sequence, new_types):
    labels = to_labels(sequence, new_types)
    numlevels = labels.shape[1]
    uniques = [ordered_unique(l) for l in labels]
    main = np.max(sequence)+1#overarching main section
    return np.array([np.hstack([[main], np.repeat(u[0], numlevels-len(u)), u])
        for u in uniques])

def build_hierarchy_bottom_up(sequence):
    pair = get_most_frequent_pair(sequence)
    next_index = int(np.max(sequence)+1)
    new_types = dict()
    #group recurring adjacent pairs into types
    while pair is not None:
        locations = get_locs_of_pair(sequence, pair)
        new_types[next_index] = pair
        sequence = replace_pairs(sequence, locations, next_index)
        next_index += 1
        pair = get_most_frequent_pair(sequence)
    #merge types that always cooccur
    to_delete = []
    for t in new_types.keys():
        parents = [k for (k,v) in new_types.items() if t in v]
        if len(parents) == 1:
            parent = parents[0]
            occs = np.count_nonzero(np.concatenate([sequence, new_types[parent]]) == t)
            if occs <= 1:
                new_types[parent] = np.concatenate(
                    [new_types[t] if p == t else [p] for p in new_types[parent]])
                to_delete.append(t)
    #delete merged types
    for t in to_delete:
        del new_types[t]
    print(new_types)
    #make hierarchy
    print(to_hierarchy(sequence, new_types))
    return to_labels2(sequence, new_types)

#print(add_transitivity([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))

