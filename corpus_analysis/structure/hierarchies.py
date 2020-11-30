from math import sqrt
from functools import reduce
from collections import OrderedDict, defaultdict
import numpy as np
import sortednp as snp
from .patterns import Pattern, segments_to_patterns, patterns_to_segments
from ..alignment.affinity import segments_to_matrix
from ..util import argmax, ordered_unique, plot_matrix, group_adjacent

# filter and sort list of patterns based on given params
def filter_and_sort_patterns(patterns, min_len=0, min_dist=0, refs=[], occs_length=False):
    #filter out patterns that are too short
    patterns = [p for p in patterns if p.l >= min_len]
    #remove translations that are too close to references
    ref_segs = [s for r in refs for s in r.to_segments()]
    min_dists = [p.remove_close_occs(ref_segs, min_dist) for p in patterns]
    #remove patterns with no remaining translations
    patterns = [p for p in patterns if len(p.t) > 1]
    #sort by position and smallest vector
    secondary = sorted(patterns, key=lambda p: (p.p, min(p.t)))
    #reverse sort by min(dist from refs, length/occs_length)
    return sorted(secondary, key=lambda p:
        p.l*sqrt(len(p.t)) if occs_length else p.l,
        #min(min_dists[patterns.index(p)], p.l*len(p.t) if occs_length else p.l),
        reverse=True)

# removes any pattern overlaps, starting with longest pattern,
# adjusting shorter ones to fit within limits
def remove_overlaps(patterns, min_len, min_dist, size):
    result = []
    patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result, True)
    i = 0
    while len(patterns) > 0:
        next = patterns.pop(0)
        #print(next)
        result.append(next)
        #plot_matrix(segments_to_matrix(patterns_to_segments(result), (size,size)), 'olap'+str(i)+'.png')
        new_boundaries = next.to_boundaries()
        #print(new_boundaries)
        for b in new_boundaries:
            patterns = [q for p in patterns for q in p.divide_at_absolute(b)]
        patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result, True)
        i += 1
    return result

def add_transitivity(patterns, proportion=1):
    patterns = filter_and_sort_patterns(patterns)
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            #find absolute positions of p in q and add translations of q to p
            pos = [q.p+r for r in q.internal_positions(p, proportion)]
            new_t = [p+t for t in q.t for p in pos]
            #if len(new_t) > 0: print(q, new_t)
            p.add_new_translations(new_t)
    return list(OrderedDict.fromkeys(patterns)) #unique patterns

#adds transitivity for full or partial overlaps
def add_transitivity2(patterns):
    patterns = filter_and_sort_patterns(patterns)
    new_patterns = []
    for i,p in enumerate(patterns):
        #print(p)
        for q in patterns[:i]:
            #find absolute positions of p in q and add translations of q to p
            apps = q.partial_appearances(p)
            #full appearances: update p
            pos = [q.p+a[0] for a in apps if a[2] == p.l]
            new_t = [p+t for t in q.t for p in pos]
            p.add_new_translations(new_t)
            #partial appearances: add new patterns
            for a in [a for a in apps if a[2] < p.l]:
                new_p = Pattern(p.p+a[1], a[2], p.t)
                new_p.add_new_translations([q.p+a[0]+t for t in q.t])
                new_patterns.append(new_p)
    return list(OrderedDict.fromkeys(patterns + new_patterns)) #unique patterns

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

def integrate_patterns(patterns):
    to_del = []
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            if q.first_occ_contained(p):
                if q.l == p.l: #p can be removed
                    q.t = np.unique(np.concatenate([q.t, p.t]))
                    to_del.append(i)
                else: #p is updated with ts of q
                    p.t = np.unique(np.concatenate([q.t, p.t]))
    patterns = [p for i,p in enumerate(patterns) if i not in to_del]
    return filter_and_sort_patterns(patterns)

def merge_patterns(patterns):
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            if q.first_occ_overlaps(p) and np.array_equal(q.t, p.t): #patterns can be merged
                new_p = min(p.p, q.p)
                q.l = max(p.p+p.l, q.p+q.l) - new_p
                q.p = new_p
                p.p = -1 #mark for deletion
    return filter_and_sort_patterns([p for p in patterns if p.p >= 0]) #filter out marked

def make_segments_hierarchical(segments, min_len, min_dist, size=None, path=None):
    patterns = filter_and_sort_patterns(segments_to_patterns(segments))
    if path: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t1.png')
    #print(patterns)
    #patterns = add_transitivity2(patterns)
    #patterns = add_transitivity(patterns, 1)#0.9)
    patterns = integrate_patterns(patterns)
    if path: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t2.png')
    #print(patterns)
    patterns = merge_patterns(patterns)
    if path: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t3.png')
    #print(patterns)
    patterns = remove_overlaps(patterns, min_len, min_dist, size)
    if path: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t4.png')
    #print(patterns)
    #patterns = add_transitivity2(patterns)
    patterns = add_transitivity(patterns, 1)#0.9)
    if path: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t5.png')
    #print(patterns)
    #plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)))
    #only return segments that fit into size (transitivity proportion < 1 can introduce artifacts)
    return [s for s in patterns_to_segments(patterns) if np.max(s) < size]

def get_most_frequent_pair(sequence, overlapping=False):
    pairs = np.dstack([sequence[:-1], sequence[1:]])[0]
    unique, counts = np.unique(pairs, axis=0, return_counts=True)
    if not overlapping:#could be optimized..
        unique = unique[counts > 1]
        counts = [len(get_locs_of_pair(sequence, p)) for p in unique]
    if len(counts) > 0:
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

def to_hierarchy(sequence, sections):
    hierarchy = sequence.tolist()
    for t in reversed(list(sections.keys())):
        hierarchy = replace_in_tree(hierarchy, t, sections[t].tolist())
    return hierarchy

def flatten(hierarchy):
    if type(hierarchy) == list:
        return [a for h in hierarchy for a in flatten(h)]
    return [hierarchy]

def reindex(array):
    uniques = np.unique(array)
    new_ids = np.zeros(np.max(uniques)+1, dtype=array.dtype)
    for i,u in enumerate(uniques):
        new_ids[u] = i
    return new_ids[array]

def pad(array, value, target_length, left=True):
    width = target_length-len(array)
    width = (width if left else 0, width if not left else 0)
    return np.pad(array, width, constant_values=(value, value))

#only packToBottom=False really makes sense. otherwise use to_labels2
def to_labels(sequence, sections, packToBottom=False):
    layers = []
    #iteratively replace sections and stack into layers
    section_lengths = {k:len(flatten((to_hierarchy(np.array([k]), sections))))
        for k in sections.keys()}
    while len(np.intersect1d(sequence, list(sections.keys()))) > 0:
        layers.append(np.concatenate([np.repeat(s, section_lengths[s])
            if s in sections else [s] for s in sequence]))
        sequence = np.concatenate([sections[s]
            if s in sections else [s] for s in sequence])
    layers.append(sequence)
    #add overarching main section
    layers.insert(0, [max(list(sections.keys()))+1] * len(sequence))
    #pack to bottom or top and remove leaf sequence values
    labels = np.array(layers).T
    uniques = [ordered_unique(l)[:-1] for l in labels]
    num_levels = labels.shape[1]-1
    labels = np.array([pad(uniques[i],
        uniques[i][0] if packToBottom else uniques[i][-1],
        num_levels, packToBottom) for i,l in enumerate(labels)])
    #back to layers and reindex
    return reindex(labels.T)

def replace_lowest_level(hierarchy, sections):
    return [h if isinstance(h, int) else
        sections[tuple(h)] if all([isinstance(e, int) for e in h])
        else replace_lowest_level(h, sections) for h in hierarchy]

def to_labels2(sequence, sections):
    hierarchy = to_hierarchy(sequence, sections)
    section_lengths = {k:len(flatten((to_hierarchy(np.array([k]), sections))))
        for k in sections.keys()}
    sections_ids = {tuple(v):k for k,v in sections.items()}
    layers = []
    layers.append(np.array(flatten(hierarchy)))
    while not all([isinstance(h, int) for h in hierarchy]):
        hierarchy = replace_lowest_level(hierarchy, sections_ids)
        layers.insert(0, np.concatenate([np.repeat(h, section_lengths[h])
            if h in sections else [h] for h in flatten(hierarchy)]))
    #add overarching main section
    layers.insert(0, np.repeat(max(list(sections.keys()))+1, len(layers[0])))
    print(np.array(layers).shape)
    #replace sequence-level labels
    labels = np.array(layers).T
    uniques = [ordered_unique(l) for l in labels]
    labels = np.array([[uniques[i][-2] if u == uniques[i][-1] else u for u in l]
        for i,l in enumerate(labels)])
    #back to layers and reindex
    return reindex(labels.T[:-1])

def to_sections(sections):
    sections = []
    keys = list(sections.keys())
    for k in keys:
        section = sections[k]
        while len(np.intersect1d(section, keys)) > 0:
            section = np.concatenate([sections[s]
                if s in sections else [s] for s in section])
        sections.append(section)
    return sections

def build_hierarchy_bottom_up(sequence):
    sequence = np.copy(sequence)
    pair = get_most_frequent_pair(sequence)
    next_index = int(np.max(sequence)+1)
    sections = dict()
    #group recurring adjacent pairs into sections
    while pair is not None:
        locations = get_locs_of_pair(sequence, pair)
        sections[next_index] = pair
        sequence = replace_pairs(sequence, locations, next_index)
        next_index += 1
        pair = get_most_frequent_pair(sequence)
    #merge sections that always cooccur
    to_delete = []
    for t in sections.keys():
        parents = [k for (k,v) in sections.items() if t in v]
        if len(parents) == 1:
            parent = parents[0]
            occs = np.count_nonzero(np.concatenate([sequence, sections[parent]]) == t)
            if occs <= 1:
                sections[parent] = np.concatenate(
                    [sections[t] if p == t else [p] for p in sections[parent]])
                to_delete.append(t)
    #delete merged sections
    for t in to_delete:
        del sections[t]
    #add sections for remaining adjacent surface objects
    ungrouped = np.where(np.isin(sequence, list(sections.keys())) == False)[0]
    groups = np.split(ungrouped, np.where(np.diff(ungrouped) != 1)[0]+1)
    for g in reversed(groups):
        if len(g) > 1:
            sections[next_index] = sequence[g]
            sequence[g[0]] = next_index
            sequence = np.delete(sequence, g[1:])
            next_index += 1
    #make hierarchy
    print(sections)
    print(to_hierarchy(sequence, sections))
    return sequence, sections

def get_hierarchy(sequence):
    sequence, sections = build_hierarchy_bottom_up(sequence)
    return to_hierarchy(sequence, sections)

def get_hierarchy_labels(sequence):
    sequence, sections = build_hierarchy_bottom_up(sequence)
    return to_labels2(sequence, sections)

def get_hierarchy_sections(sequence):
    sequence, sections = build_hierarchy_bottom_up(sequence)
    return to_sections(sequence, sections)

# print(add_transitivity([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(4, 3, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(1, 3, [0,10,18])]))
# remove_overlaps([Pattern(2, 4, [0,10,30]), Pattern(1, 3, [0,10,18])], 0, 1)
# remove_overlaps([Pattern(31, 71, [0, 92, 260, 350]), Pattern(196, 95, [0, 256]),
#     Pattern(16, 15, [0, 260, 516]), Pattern(86, 16, [0, 92, 348])], 0, 3)
# print(reindex(np.array([3,1,5])))