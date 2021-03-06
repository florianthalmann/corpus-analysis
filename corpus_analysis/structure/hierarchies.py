from math import sqrt
from functools import reduce
from collections import OrderedDict, defaultdict, Counter
import numpy as np
import sortednp as snp
from graph_tool.topology import transitive_closure
from .patterns import Pattern, segments_to_patterns, patterns_to_segments
from .sections import segments_to_sections, remove_contained, merge_overlapping
from .graphs import graph_from_matrix, segments_to_matrix, matrix_to_segments,\
    adjacency_matrix
from ..alignment.affinity import segments_to_matrix, smooth_matrix
from ..util import argmax, ordered_unique, plot_matrix, group_adjacent,\
    indices_of_subarray, plot

# filter and sort list of patterns based on given params
def filter_and_sort_patterns(patterns, min_len=0, min_dist=0, refs=[], occs_length=True):
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
        p.l*len(p.t) if occs_length else p.l,#p.l*sqrt(len(p.t)) if occs_length else p.l,
        #min(min_dists[patterns.index(p)], p.l*len(p.t) if occs_length else p.l),
        reverse=True)

# removes any pattern overlaps, starting with longest pattern,
# adjusting shorter ones to fit within limits
def remove_overlaps(patterns, min_len, min_dist, size, occs_length):
    result = []
    patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result, occs_length)#[:6]#[:10]
    i = 0
    boundaries = []
    while len(patterns) > 0:
        next = patterns.pop(0)
        #print(tuple(np.concatenate(next.to_segments()).T))
        #print(next)
        result.append(next)
        #result = add_transitivity(result)#add_transitivity_graph(result, size)
        #plot_matrix(segments_to_matrix(patterns_to_segments(result), (size,size)), 'oolap'+str(i)+'-.png')
        result = add_transitivity_graph(result, size)
        matrix = segments_to_matrix(patterns_to_segments(result), (size,size))
        #plot_matrix(matrix, 'olap'+str(i)+'-.png')
        # additions = [p for p in patterns if overlapping_prop(p, matrix) >= 0.95]
        # [result.append(p) for p in additions]
        # result = add_transitivity_graph(result, size)
        # patterns = [p for p in patterns if p not in additions]
        # matrix = segments_to_matrix(patterns_to_segments(result), (size,size))
        #plot_matrix(matrix, 'olap'+str(i)+'.png')
        # prev_bounds = boundaries
        new_boundaries = np.unique(np.concatenate([r.to_boundaries() for r in result]))#next.to_boundaries()
        #new_boundaries = np.setdiff1d(boundaries, prev_bounds)
        #print(new_boundaries)
        for b in new_boundaries:
            patterns = [q for p in patterns for q in p.divide_at_absolute(b)]
        patterns = [p for p in patterns if not fully_contained(p, matrix)]
        patterns = filter_and_sort_patterns(patterns, min_len, min_dist, result, occs_length)
        i += 1
    return result

def make_segments_hierarchical(segments, min_len, min_dist, size, target=None, path=None):
    segments = segments.copy()#since we're removing from it
    if target is None:
        target = segments_to_matrix(segments, (size,size))#replace with raw or intermediary
    target += target.T
    np.fill_diagonal(target, 1)
    #plot_matrix(target, 'new0.png')
    #plot_matrix(segments_to_matrix(segments, (size,size)), 'new00.png')
    improvement = 1
    matrix = np.zeros((size,size))
    distance = len(np.nonzero(target-matrix > 0)[0])
    #print(distance)
    i=1
    while improvement > 0 and len(segments) > 0:
        # matsegs = matrix_to_segments(np.triu(matrix, k=1))
        # starts = np.unique([s[0][0] for s in matsegs])
        # ends = np.unique([s[-1][1] for s in matsegs])
        # print(starts, ends)
        #candidates = 
        def best_transitive(matrices, last_best=False):
            matrices = [add_transitivity_to_matrix(m) for m in matrices]
            dists = [len(np.nonzero(target-m > 0)[0])+(len(np.nonzero(m-target > 0)[0])**1) for m in matrices]
            best = len(dists)-np.argmin(dists[::-1])-1 if last_best else np.argmin(dists)
            #print(dists[best], best, dists)
            return best, matrices[best], dists[best]
        
        matrices = [(matrix+segments_to_matrix([s], (size,size))) for s in segments]
        best, mat, dist = best_transitive(matrices)
        #plot_matrix(mat, 'new'+str(i)+'-.png')
        
        def get_start_variations(s, length=10):
            l = min(s[0][0], s[0][1], length)
            r = np.arange(s[0][0]-l, s[0][0])
            p = np.vstack((r, r+(s[0][1]-s[0][0]))).T
            return [np.concatenate((p[i:], s)) for i in range(l)] + [s] \
                + [s[i:] for i in range(1, min(l+1, len(s)))]
        
        def get_end_variations(s, length=10):
            l = min(size-s[-1][0]-1, size-s[-1][1]-1, length)
            r = np.arange(s[-1][0]+1, s[-1][0]+l+1)
            p = np.vstack((r, r+(s[0][1]-s[0][0]))).T
            return [np.concatenate((s, p[:l-i])) for i in range(l)] + [s] \
                + [s[:l-i] for i in range(1, min(l+1, len(s)))]
        
        vars = get_start_variations(segments[best])
        matrices = [(matrix+segments_to_matrix([v], (size,size))) for v in vars]
        best, mat, dist = best_transitive(matrices, True)#last min (shortest possible)
        
        vars = get_end_variations(vars[best])
        matrices = [(matrix+segments_to_matrix([v], (size,size))) for v in vars]
        best, mat, dist = best_transitive(matrices, True)#last min (shortest possible)
        
        if dist < distance:
            matrix = mat
            #keep only parts of segments not covered by current matrix
            #segments = [s for s in segments if np.sum(matrix[tuple(s.T)]) / len(s) < 1]
            segments = [s[np.nonzero(matrix[tuple(s.T)] == 0)] for s in segments]
            segments = [s for s in segments if len(s) > 0]
            i+=1
        improvement = distance-dist
        distance = dist
        #print(distance)
    # unsmoothed = matrix
    # matrix = smooth_matrix(matrix, True, 5, .4)
    # matrix = smooth_matrix(matrix+unsmoothed, True, 5, .4)
    # matrix = add_transitivity_to_matrix(matrix)
    # plot_matrix(matrix, 'new'+str(i)+'.png')
    #print(len(np.nonzero(target-matrix > 0)[0])+(len(np.nonzero(matrix-target > 0)[0])**1))
    return matrix_to_segments(matrix)

def fully_contained(pattern, matrix):
    return overlapping_prop(pattern, matrix) == 1

def overlapping_prop(pattern, matrix):
    points = np.concatenate(pattern.to_segments())
    return np.sum(matrix[tuple(points.T)]) / len(points)

def remove_overlaps2(patterns, min_len, min_dist, size, occs_length):
    result = []
    boundaries = np.bincount(np.hstack([r.to_boundaries() for r in patterns]))
    boundaries = boundaries[:-2]+boundaries[1:-1]+boundaries[2:]
    print(boundaries)
    plot(boundaries)

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

def add_transitivity_graph(patterns, size):
    m = segments_to_matrix(patterns_to_segments(patterns), (size,size))
    return segments_to_patterns(matrix_to_segments(add_transitivity_to_matrix(m)))

def add_transitivity_to_matrix(matrix):
    g, w = graph_from_matrix(matrix+matrix.T, True)
    return adjacency_matrix(transitive_closure(g))

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

#update translations of patterns contained by others
def integrate_patterns(patterns):
    to_del = []
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            if q.first_occ_contained(p):
                if q.l == p.l: #p can be removed
                    q.t = np.unique(np.concatenate([q.t, p.t]))
                    to_del.append(i)
                else: #p is updated with ts of q (p.l < q.l)
                    p.t = np.unique(np.concatenate([q.t, p.t]))
    patterns = [p for i,p in enumerate(patterns) if i not in to_del]
    return filter_and_sort_patterns(patterns)

#merge overlapping patterns with same translations
def merge_patterns(patterns):
    for i,p in enumerate(patterns):
        for q in patterns[:i]:
            #if q.first_occ_overlap(p) > 0.8 and len(set(q.t).intersection(set(p.t))) >= max(len(p.t), len(q.t))-1:
            if q.first_occ_overlaps(p) and np.array_equal(q.t, p.t): #patterns can be merged
                new_p = min(p.p, q.p)
                q.l = max(p.p+p.l, q.p+q.l) - new_p
                q.p = new_p
                #q.t = np.unique(np.hstack((p.t, q.t)))
                p.p = -1 #mark for deletion
    return filter_and_sort_patterns([p for p in patterns if p.p >= 0]) #filter out marked

def make_segments_hierarchical2(segments, min_len, min_dist, size=None, path=None):
    patterns = filter_and_sort_patterns(segments_to_patterns(segments))
    if path != None: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t1.png')
    #print(patterns)
    #patterns = add_transitivity2(patterns)
    #patterns = add_transitivity(patterns, 1)#0.9)
    patterns = integrate_patterns(patterns)
    if path != None: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t2.png')
    #print(patterns)
    patterns = merge_patterns(patterns)
    if path != None: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t3.png')
    #print(patterns)
    patterns = remove_overlaps(patterns, min_len, min_dist, size, occs_length=True)
    if path != None: plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)), path+'t4.png')
    #print(patterns)
    #patterns = add_transitivity2(patterns)
    #patterns = add_transitivity(patterns, 1)#0.9)
    patterns = add_transitivity_graph(patterns, size)
    segments = [s for s in patterns_to_segments(patterns) if len(s) >= min_len]
    if path != None: plot_matrix(segments_to_matrix(segments, (size,size)), path+'t5.png')
    #print(patterns)
    #plot_matrix(segments_to_matrix(patterns_to_segments(patterns), (size,size)))
    #only return segments that fit into size (transitivity proportion < 1 can introduce artifacts)
    #return [s for s in patterns_to_segments(patterns) if np.max(s) < size]
    return segments

def make_segments_hierarchical3(segments, min_len, min_dist, size=None, path=None):
    sections = segments_to_sections(segments)
    print(sections)
    sections = remove_contained(sections)
    print(sections)
    sections = merge_overlapping(sections, 0.8)
    print(sections)
    print(hey)

def thin_out2(pairs):
    #get locations of repetitions 
    diff = np.diff(pairs, axis=0)
    same = np.all(diff == 0, axis=1)
    notsame = np.where(~same)
    #get the heights of the plateaus at their initial positions
    plateaus = np.diff(np.concatenate(([0], np.cumsum(same)[notsame])))
    #subtract plateau values from series to be summed
    addition = same.astype(int)
    addition[notsame] = -plateaus
    return pairs[np.where(np.cumsum(addition)%2==0)]

def get_most_frequent_pair(sequences, ignore=[], overlapping=False):
    #find all valid pairs (no element in ignore)
    pairs = [np.dstack([s[:-1], s[1:]])[0] for s in sequences]
    uneq = [thin_out2(ps) for ps in pairs]
    #print(uneq[0][:20])
    valid = [np.where(np.all(~np.isin(ps, ignore), axis=1))[0] for ps in uneq]
    valid = np.concatenate([ps[valid[i]] for i,ps in enumerate(uneq)])
    #print(valid[:5])
    counts = Counter(valid.view(dtype=np.dtype([('x',valid.dtype),('y',valid.dtype)]))[:,0].tolist())
    #print(counts)
    counts = sorted(counts.items(), key=lambda c: c[1], reverse=True)
    if counts[0][1] > 1:
        locs = [get_locs_of_pair(s, counts[0][0], overlapping) for s in sequences]
        return counts[0][0], [(i,j) for i,l in enumerate(locs) for j in l]
    return None, None

def thin_out(a, min_dist=2):
    return np.array(reduce(lambda r,i:
        r+[i] if len(r)==0 or abs(i-r[-1]) >= min_dist else r, a, []))

def get_locs_of_pair(sequence, pair, overlapping=False):
    pairs = np.dstack([sequence[:-1], sequence[1:]])[0]
    indices = np.where(np.all(pairs == pair, axis=1))[0]
    return indices if overlapping else thin_out(indices)

def replace_pairs(sequence, indices, replacement):
    if len(indices) > 0:
        sequence[indices] = replacement
        return np.delete(sequence, indices+1)
    return sequence

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

def reindex(arrays):
    uniques = np.unique(np.concatenate(arrays))
    new_ids = np.zeros(np.max(uniques)+1, dtype=arrays[0].dtype)
    for i,u in enumerate(uniques):
        new_ids[u] = i
    return [new_ids[a] for a in arrays]

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
    return reindex([labels.T])

def replace_lowest_level(hierarchy, sections):
    return [h if isinstance(h, int) else
        sections[tuple(h)] if all([isinstance(e, int) for e in h])
        else replace_lowest_level(h, sections) for h in hierarchy]

def to_labels2(sequences, sections, section_lengths):
    #merge sequences and treat together
    hierarchy = to_hierarchy(np.hstack(sequences), sections)
    sections_ids = {tuple(v):k for k,v in sections.items()}
    layers = []
    layers.append(np.array(flatten(hierarchy)))
    while not all([isinstance(h, int) for h in hierarchy]):
        hierarchy = replace_lowest_level(hierarchy, sections_ids)
        layers.insert(0, np.concatenate([np.repeat(h, section_lengths[h])
            if h in sections else [h] for h in flatten(hierarchy)]))
    #add overarching main section
    layers.insert(0, np.repeat(max(list(sections.keys()))+1, len(layers[0])))
    #print(np.array(layers).shape)
    labels = np.array(layers).T
    #replace sequence-level labels with next higher section
    uniques = [ordered_unique(l) for l in labels]#uniques for each time point
    labels = np.array([[uniques[i][-2] if u == uniques[i][-1] else u for u in l]
        for i,l in enumerate(labels)])
    #back to layers and reindex
    reindexed = reindex2(labels.T[:])
    #now cut at sequence boundaries to get original sequence lengths
    seqlens = [sum([section_lengths[c] if c in section_lengths else 1 for c in s])
        for s in sequences]
    indices = np.cumsum(seqlens)[:-1]
    return np.split(reindexed, indices, axis=1)

#fancy reindexing based on section contents (similar contents = similar label)
def reindex2(labels):
    newlabels = np.zeros(np.max(labels)+1).astype(float)
    #map bottom level to integers
    uniq = np.unique(labels[-1]) #ordered by original values
    uniq = labels[-1][ #order of appearance
        np.sort(np.array([np.argmax(u == labels[-1]) for u in uniq]))]
    newlabels[uniq] = np.arange(len(uniq))
    bottom = newlabels[labels[-1]]
    #higher levels become averages of contained bottom-level ints
    for l in labels[:-1]:
        for u in np.unique(l):
            newlabels[u] = np.mean(bottom[np.where(l == u)])
    #map new labels to integers
    uniq = np.unique(newlabels)
    newlabels = np.array([np.argmax(uniq == l) for l in newlabels])
    return newlabels[labels]

def to_sections(sections):
    sects = []
    keys = list(sections.keys())
    for k in keys:
        section = sections[k]
        while len(np.intersect1d(section, keys)) > 0:
            section = np.concatenate([sections[s]
                if s in sections else [s] for s in section])
        sects.append(section)
    return sects

def find_sections_bottom_up(sequences, ignore=[]):
    sequences = [np.copy(s) for s in sequences]
    seq_indices = [np.arange(len(s)) for s in sequences]
    pair, locs = get_most_frequent_pair(sequences, ignore)
    next_index = int(np.max(np.hstack(sequences))+1)
    sections = dict()
    occurrences = dict()
    #group recurring adjacent pairs into sections
    while pair is not None:
        sections[next_index] = np.array(list(pair))
        occurrences[next_index] = [(l[0], seq_indices[l[0]][l[1]]) for l in locs]
        for i,s in enumerate(sequences):
            slocs = np.array([l[1] for l in locs if l[0] == i])
            seq_indices[i] = np.delete(seq_indices[i], slocs+1)
            sequences[i] = replace_pairs(s, slocs, next_index)
        pair, locs = get_most_frequent_pair(sequences, ignore)
        next_index += 1
    #print(to_hierarchy(np.array(sequences[0]), sections))
    #merge sections that always cooccur (nested)
    to_delete = []
    for t in sections.keys():
        parents = [k for (k,v) in sections.items() if t in v]
        if len(parents) == 1:
            parent = parents[0]
            #doesn't occur in top sequences (outside of parent) and only once in parent
            occs = np.count_nonzero(np.concatenate(sequences+[sections[parent]]) == t)
            if occs <= 1: #no need to update occurrence dict since same number
                sections[parent] = np.concatenate(
                    [sections[t] if p == t else [p] for p in sections[parent]])
                to_delete.append(t)
    #delete merged sections
    for t in to_delete:
        del sections[t]
        del occurrences[t]
    #print(to_hierarchy(np.array(sequences[0]), sections))
    #add sections for remaining adjacent surface objects
    for i,s in enumerate(sequences):
        ungrouped = np.where(np.isin(s, list(sections.keys())+ignore) == False)[0]
        groups = np.split(ungrouped, np.where(np.diff(ungrouped) != 1)[0]+1)
        groups = [g for g in groups if len(g) > 1]
        for g in reversed(groups):
            sections[next_index] = s[g]
            occurrences[next_index] = [(i, seq_indices[i][g[0]])]
            s[g[0]] = next_index
            s = np.delete(s, g[1:])
            next_index += 1
        sequences[i] = s
    #print(to_hierarchy(np.array(sequences[0]), sections))
    #make hierarchy
    #print(sections)
    #print(to_hierarchy(sequence, sections))
    return sequences, sections, occurrences

def get_hierarchies(sequences):
    sequences, sections, occs = find_sections_bottom_up(sequences)
    return [to_hierarchy(s, sections) for s in sequences]

def get_hierarchy_labels(sequences):
    sequences, sections, occs = find_sections_bottom_up(sequences)
    section_lengths = {k:len(flatten((to_hierarchy(np.array([k]), sections))))
        for k in sections.keys()}
    return to_labels2(sequences, sections, section_lengths)

def get_recurring_subseqs(sequences):
    sequences, sections, occs = find_sections_bottom_up(sequences)
    seqs = {k:flatten(to_hierarchy(np.array([k]), sections))
        for k in sections.keys()}
    seqs = [(seqs[k], len(occs[k])) for k in seqs.keys()]
    return sorted(seqs, key=lambda s: s[1], reverse=True)

#finds the best non-overlapping labels for the given set of sequences
def get_most_salient_labels(sequences, count=0, ignore=[], min_len=2):
    seqs, sections, occs = find_sections_bottom_up(sequences, ignore)
    flatsecs = {k:flatten(to_hierarchy(np.array([k]), sections))
        for k in sections.keys()}
    seclens = {k:len(flatsecs[k]) for k in sections.keys()}
    #only keep patterns longer than min_len
    occs = {s:o for s,o in occs.items() if seclens[s] >= min_len}
    #sections = {s:o for s,o in sections.items() if seclens[s] >= min_len}
    #sort by coverage and occurrences
    most_salient = []
    outseqs = [np.repeat(-1, len(s)) for s in sequences]
    remaining = list(occs.items())
    while len(remaining) > 0 and (count == 0 or len(most_salient) < count):
        coverages = [seclens[o[0]]*len(o[1]) for o in remaining]
        current_best = remaining.pop(np.argmax(coverages))
        #update sequences
        for o in current_best[1]:
            outseqs[o[0]][o[1]:o[1]+seclens[current_best[0]]] = current_best[0]#len(most_salient)
        most_salient.append(current_best)
        #remove overlaps
        remaining = [(r[0], [o for o in r[1]
            if np.all(outseqs[o[0]][o[1]:o[1]+seclens[r[0]]] == -1)])
            for r in remaining]
        remaining = [r for r in remaining if len(r[1]) > 0]
    return outseqs, sections, dict(most_salient)

def get_top_level_labels(sequences, count=0, ignore=[], min_len=2):
    seqs, sections, occs = find_sections_bottom_up(sequences, ignore)
    flatsecs = {k:flatten(to_hierarchy(np.array([k]), sections))
        for k in sections.keys()}
    seclens = {k:len(flatsecs[k]) for k in sections.keys()}
    #only keep patterns longer than min_len
    occs = {s:o for s,o in occs.items() if seclens[s] >= min_len}
    #sections = {s:o for s,o in sections.items() if seclens[s] >= min_len}
    #sort by coverage and occurrences
    most_salient = []
    outseqs = [np.repeat(-1, len(s)) for s in sequences]
    remaining = list(occs.items())
    while len(remaining) > 0 and (count == 0 or len(most_salient) < count):
        coverages = [seclens[o[0]]*len(o[1]) for o in remaining]
        current_best = remaining.pop(np.argmax(coverages))
        #update sequences
        for o in current_best[1]:
            outseqs[o[0]][o[1]:o[1]+seclens[current_best[0]]] = current_best[0]#len(most_salient)
        most_salient.append(current_best)
        #remove overlaps
        remaining = [(r[0], [o for o in r[1]
            if np.all(outseqs[o[0]][o[1]:o[1]+seclens[r[0]]] == -1)])
            for r in remaining]
        remaining = [r for r in remaining if len(r[1]) > 0]
    return outseqs, sections, dict(most_salient)

def contract_sections(seqs, sections, occs):
    seclens = {k:len(flatten(to_hierarchy(np.array([k]), sections)))
        for k in occs.keys()}
    contracted = []
    for s in seqs:
        contracted.append([])
        i = 0
        while i < len(s):
            contracted[-1].append(s[i])
            if s[i] in occs:
                i += seclens[s[i]]
            else: i += 1
    return [np.array(c) for c in contracted]

def get_longest_sections(sequences, ignore):
    seqs, sections, occs = find_sections_bottom_up(sequences, ignore)
    flatsecs = [(flatten(to_hierarchy(np.array([k]), sections)), len(occs[k]))
        for k in sections.keys()]
    #sort by length * sqrt of occurrences
    return sorted(flatsecs, key=lambda s: len(s[0])*s[1], reverse=True)

def get_hierarchy_sections(sequences):
    sequences, sections, occs = find_sections_bottom_up(sequences)
    return [to_sections(s, sections) for s in sequences]

# print(add_transitivity([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(3, 2, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(4, 3, [0,10,18])]))
# print(add_transitivity2([Pattern(2, 4, [0,10,30]), Pattern(1, 3, [0,10,18])]))
# remove_overlaps([Pattern(2, 4, [0,10,30]), Pattern(1, 3, [0,10,18])], 0, 1)
# remove_overlaps([Pattern(31, 71, [0, 92, 260, 350]), Pattern(196, 95, [0, 256]),
#     Pattern(16, 15, [0, 260, 516]), Pattern(86, 16, [0, 92, 348])], 0, 3)
# print(reindex([np.array([3,1,5])]))
# print(indices_of_subarray(np.array([1,2,1,2]), np.array([1,2])))
# reindex2(np.array([[0,0,1,1,1,1],[2,2,4,4,5,5],[7,6,7,8,10,9]]))