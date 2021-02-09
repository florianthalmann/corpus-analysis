import datetime, math, statistics, tqdm
from itertools import groupby, product
from collections import Counter, defaultdict
from heapq import merge
import numpy as np
import sortednp as snp
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import graph_tool.all as gt
from graph_tool.all import Graph, GraphView, graph_draw
from .graphs import graph_from_matrix
from .hierarchies import get_hierarchy_labels, get_most_salient_labels,\
    get_longest_sections
from ..util import plot_sequences, mode, flatten, group_by, plot_matrix,\
    multiprocess, split
from ..clusters.histograms import freq_hist_clusters, trans_hist_clusters,\
    freq_trans_hist_clusters
from ..alignment.smith_waterman import smith_waterman

MIN_VERSIONS = 0.2 #how many of the versions need to contain the patterns
PARSIM = True
PARSIM_DIFF = 0 #the largest allowed difference in parsimonious mode (full containment == 0)
MIN_SIM = 0.9 #min similarity for non-parsimonious similarity
COMPS_NOT_BLOCKS = True #use connected components instead of community blocks

def similarity(match, p1, p2, parsim):
    minlen = min(len(p1), len(p2))
    if parsim: return 1 if abs(match - minlen) == PARSIM_DIFF else 0
    return match / minlen

def sw_similarity(p1, p2, parsim=PARSIM):
    return similarity(len(smith_waterman(p1, p2)[0]), p1, p2, parsim)

def isect_similarity(p1, p2, parsim=PARSIM):
    return similarity(len(snp.intersect(p1, p2)), p1, p2, parsim)

def get_occ_matrix(pattern, dict, num_seqs, max_len):
    matrix = np.zeros((num_seqs, max_len))
    for occ in dict[pattern]:
        matrix[occ[0], np.arange(occ[1], occ[1]+len(pattern))] = 1
    return matrix

#finds the longest overlap between any two occurrences of the patterns
def cooc_similarity(p1, p2, occmat1, occmat2, parsim=PARSIM):
    both = np.logical_and(occmat1, occmat2)
    diffs = np.transpose(np.diff(np.where(both == 1)))
    overlaps = np.split(diffs, np.where(diffs != [0,1])[0]+1)
    return similarity(max([len(o) for o in overlaps]), p1, p2, parsim)

def group_by_comps(patterns, adjacency_matrix):
    comps = connected_components(adjacency_matrix)[1]
    groups = group_by(range(len(patterns)), lambda i: comps[i])
    return [[patterns[i] for i in g] for g in groups]

#group by a transitive distance
def group_by_sim2(patterns, simfunc, minsim):
    sims = np.array([[simfunc(p1, p2) >= minsim if j > i else 0
        for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
    return group_by_comps(patterns, sims)

def group_by_sim(groups, simfunc, minsim):
    return [subg for g in groups for subg in group_by_sim2(g, simfunc, minsim)]

def refine_adjacency(adjacency, simfunc, patterns):
    return np.array([[simfunc(p1, p2) if j > i and adjacency[i,j] else 0
        for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])

def cluster(patterns, relative):
    #prepared = np.array([np.array(p)+1 for p in patterns])#to positive integers
    prepared = np.array([np.array([c for c in p if c >= 0]) for p in patterns])#remove -1
    clustered, unclustered = freq_trans_hist_clusters(prepared, relative)
    return [[patterns[i] for i in c] for c in clustered],\
        [patterns[i] for i in unclustered]

#keys: pattern tuples with -1 for blanks, values: list of occurrences (sequence, position)
def create_pattern_dict(sequences, pairings, alignments):
    patterns = dict()
    #what to do with overlapping locations?
    for i,a in enumerate(alignments):
        for s in a:
            j, k = pairings[i]
            s1, s2 = sequences[j][s.T[0]], sequences[k][s.T[1]]
            pattern = tuple(np.where(s1 == s2, s1, -1))
            locations = [(j, s[0][0]), (k, s[0][1])]
            if not pattern in patterns:
                patterns[pattern] = set()
            patterns[pattern].update(locations)
    return patterns

class PatternGraph:
    
    def __init__(self, sequences, pairings, alignments):
        plot_sequences(sequences, 'seqpats1.png')
        self.patterns = create_pattern_dict(sequences, pairings, alignments)
        self.equivalences = {}
        self.print('all')
        #prune pattern dict: keep only ones occurring in a min num of versions
        self.patterns = {k:v for k,v in self.patterns.items()
            if len(np.unique([o[1] for o in v])) >= len(sequences)*MIN_VERSIONS}
        self.print('frequent')
        # #merge cooccurring patterns
        # self.merge_patterns(lambda p, q: len(p) == len(q) and
        #     len(self.patterns[p].intersection(self.patterns[q])) > 0)
        # self.print('merged cooc')
        self.merge_patterns(lambda p, q: len(p) == len(q) and
            all(p[i] == -1 or q[i] == -1 or p[i] == q[i] for i in range(len(p))))
        self.print('merged equiv')
        #print(sorted(list(self.patterns.items()), key=lambda p: len(p[0])*,math.sqrt(len(p[1])), reverse=True)[0:10])
        
        patterns = list(self.patterns.keys())
        groups = [patterns]
        
        # #cluster relative
        # groups = [c for g in groups for c in cluster(g, False)[0]]
        # print('clusters', [len(g) for g in groups])
        # print('clusters', [sum([len(p) for p in g]) for g in groups])
        # groups = self.keep_largest_pattern_groups(groups, 10)
        # #groups = groups[:5]
        # [print(g[:5]) for g in groups]
        # 
        # ps = groups[0]
        # ref = self.patterns[ps[0]]
        # [print(self.patterns[ps[i]]) for i in range(0,len(ps))]
        
        # #cluster absolute
        # groups = [c for g in groups for c in cluster(g, True)[0]]
        # print('reclusters', [len(c) for c in groups])
        # #[print([e for e in g]) for g in groups]
        
        # groups = self.keep_largest_pattern_groups(groups, 2)
        # print('kept largest', [len(c) for c in groups])
        # 
        # [print([e for e in g]) for g in groups]
        
        adjacency = self.get_adjacency_matrix(groups)
        
        #cooccurrence similarity
        num_seqs = len(sequences)
        max_len = max([len(s) for s in sequences])
        occ_matrices = [get_occ_matrix(p, self.patterns, num_seqs, max_len)
            for p in patterns]
        adjacency = np.array([[cooc_similarity(p1, p2, occ_matrices[i], occ_matrices[j])
            if j > i and adjacency[i,j] else 0
            for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
        print('cooc', len(np.nonzero(adjacency)[0]))
        
        #content similarity
        sorted_patterns = list([np.sort(p) for p in patterns])
        adjacency = refine_adjacency(adjacency, isect_similarity, sorted_patterns)
        print('cont', len(np.nonzero(adjacency)[0]))
        
        #sequence similarity
        adjacency = refine_adjacency(adjacency, sw_similarity, patterns)
        print('seqs', len(np.nonzero(adjacency)[0]))
        
        groups = group_by_comps(patterns, adjacency)
        groups = self.keep_most_frequent_pattern_groups(groups, 8)
        
        #make weighted adjacency matrix
        patterns = list(self.patterns.keys())
        adjacency = self.get_adjacency_matrix(groups)
        pattern_counts = np.array([len(self.patterns[p]) for p in patterns])
        adjacency *= np.minimum.outer(pattern_counts, pattern_counts)
        #make graph and find communities
        g, weights = graph_from_matrix(adjacency)
        g = GraphView(g, vfilt=g.get_total_degrees(g.get_vertices()) > 0)
        print(g)
        graph_draw(g, output_size=(1000, 1000), edge_pen_width=weights, output="patterns.png")
        if COMPS_NOT_BLOCKS:
            blocks = gt.label_components(g)[0].a+1
        else:
            state = gt.minimize_blockmodel_dl(g, overlap=True,
                state_args=dict(recs=[weights], rec_types=["discrete-binomial"]))#, B_max=8)
            state.draw(output_size=(1000, 1000), edge_pen_width=weights, output="patterns2.png")
            blocks = state.get_blocks().a+1
        
        #make annotated sequences
        # typeseqs = [np.zeros_like(s)-1 for s in sequences]
        # for p,b in sorted(list(zip(patterns, blocks)), key=lambda pb: pb[1]):
        #     for j,k in self.patterns[p]:
        #         typeseqs[j][k:k+len(p)] = b
        typeseqs = [[[] for e in s] for s in sequences]
        for p,b in sorted(list(zip(patterns, blocks)), key=lambda pb: pb[1]):
            for j,k in self.patterns[p]:
                for l in range(k, k+len(p)):
                    typeseqs[j][l] += [b]
        typeseqs = [[mode(e) if len(e) > 0 else 0 for e in s] for s in typeseqs]
        plot_sequences(typeseqs, 'seqpats2.png')
    
    def print(self, title):
        longest3 = [len(l[1]) for l in sorted(list(self.patterns.items()),
            key=lambda p: len(p[1]), reverse=True)[:3]]
        print(title, len(self.patterns), len(self.equivalences), longest3)
    
    def merge_patterns(self, equiv_func):#lambda p,q: bool
        merged = {}
        for p in list(self.patterns.keys()):
            q = next((q for q in merged.keys() if equiv_func(p, q)), None)
            if q:
                merged[q].update(self.patterns[p])
                if q not in self.equivalences:
                    self.equivalences[q] = set()
                self.equivalences[q].add(p)
            else:
                merged[p] = self.patterns[p]
        self.patterns = merged
    
    def keep_largest_pattern_groups(self, groups, count, length_based=True):
        key = lambda c: sum([len(p) for p in c]) if length_based \
            else lambda c: len(c)
        largest = sorted(groups, key=key, reverse=True)[:count]
        self.patterns = {k:v for k,v in self.patterns.items()
            if k in flatten(largest, 1)}
        return largest
    
    def keep_most_frequent_pattern_groups(self, groups, count, length_based=True):
        key = lambda c: sum([len(self.patterns[p]) for p in c]) if length_based \
            else lambda c: len(c)
        largest = sorted(groups, key=key, reverse=True)[:count]
        self.patterns = {k:v for k,v in self.patterns.items()
            if k in flatten(largest, 1)}
        return largest
    
    def get_adjacency_matrix(self, groups):
        patterns = list(self.patterns.keys())
        indices = {p:i for i,p in enumerate(patterns)}
        adjacency = np.zeros((len(patterns), len(patterns)))
        for g in groups:
            for p, q in product(g, g):
                i, j = indices[p], indices[q]
                if j > i:
                    adjacency[i][j] = 1
        return adjacency


########################################


MIN_DIST = 16
MAX_OCCS = 50000
INIT_MIN_COUNT = 3#7
MIN_COUNT = 1
MAX_MIN_SIZE = 0#20

def super_alignment_graph(song, sequences, pairings, alignments):
    all_patterns = create_pattern_dict(sequences, pairings, alignments)
    print(len(all_patterns))
    all_points = set([(i,j) for i,s in enumerate(sequences) for j in range(len(s))])
    all_patterns = filter_patterns(all_patterns, 0, remove_uniform=True)
    print_status('all', all_patterns)
    print(len(all_points))
    
    all_ordered = sorted(all_patterns.items(),
        key=lambda p: len(np.unique([v for v,t in p[1]])), reverse=True)
    
    # groups = group_patterns(all_patterns, length=True)
    # groups = [c for g in groups for c in cluster(g, True)[0]]
    # groups = sorted(groups, key=lambda g: sum([len(all_patterns[p]) for p in g]), reverse=True)
    # all_ordered = [(p,all_patterns[p]) for p in flatten(groups, 1)]
    # print("clustered", len(groups), statistics.median([len(g) for g in groups]))
    
    #all_ordered = list(all_patterns.items())
    
    #print(by_versions[0])
    comps = []
    locs = {}
    incomp = set()
    remaining = all_patterns.copy()
    ordered = all_ordered.copy()
    remaining_points = all_points
    min_count = INIT_MIN_COUNT
    min_size = 0
    last_adjusted = "min_count"
    while len(remaining) > 0 and len(ordered) > 0:
        #filter patterns
        cutoff = np.argmax(np.cumsum([len(o) for p,o in ordered]) > MAX_OCCS)
        if cutoff == 0:#always at least 5, or all remaining if no cutoff
            cutoff = 5 if len(ordered[0][1]) > MAX_OCCS else len(ordered)
        patterns = {p:o for p,o in ordered[:cutoff]}
        print('selected', len(patterns), 'of', len(ordered))
        
        if len(patterns) > 0:
            
            groups = group_patterns(patterns, length=True, cooccurrence=True, similarity=False)
            
            conns = get_most_common_connections(groups, patterns, sequences, MIN_DIST, min_count=min_count)
            
            print('graph', datetime.datetime.now())
            #build non-ambiguous graph containing strongest connections
            add_to_components(conns, comps, locs, incomp, MIN_DIST)
            #print(len(comps), datetime.datetime.now(), [len(c) for c in comps])
            #print([[s for s in c if s[0] in [1,2,3]] for c in comps])
            
            previous = len(remaining_points)
            inbigcomps = set([p for c in comps if len(c) >= min_size for p in c])
            remaining_points = all_points - inbigcomps#locs.keys()
            print('rempoints', len(remaining_points))
            
            #no improvement or all patterns taken at once
            if len(remaining_points) == previous or len(patterns) == len(ordered):
                if min_size < MAX_MIN_SIZE:
                    min_size += 10
                    remaining = all_patterns #need to widen selection
                    print('increased min size to', min_size)
                    inbigcomps = set([p for c in comps if len(c) >= min_size for p in c])
                    remaining_points = all_points - inbigcomps#locs.keys()
                    print('rempoints', len(remaining_points))
                    remaining = filter_patterns(all_patterns, include=remaining_points)
                    ordered = [p for p in all_ordered if p[0] in remaining]
                    print('remaining', len(remaining))
                elif min_count > MIN_COUNT:
                    min_count -= 1
                    min_size = 0
                    print('reduced min count to', min_count)
                else: break
            else:
                remaining = filter_patterns(remaining, include=remaining_points)
                ordered = [p for p in all_ordered if p[0] in remaining]
                print('remaining', len(remaining))
    
    
    #remove empty comps and sort
    comps = [c for c in comps if len(c) > 0]
    comps = sorted(comps, key=lambda c: np.mean([s[1] for s in c]))
    print(len(comps), datetime.datetime.now(), [len(c) for c in comps])
    
    #print('VALID', [valid(c,MIN_DIST) for c in comps])
    #print('REALLY', [[for i in np.unique([s[0] for s in c])] for c in comps])
    
    #sort by avg first occ
    # comps = sorted(comps, key=lambda c:
    #     np.mean([min([o[1] for o in c if o[0] == i]) for i in np.unique([o[0] for o in c])]))
    # 
    # print(len(comps), datetime.datetime.now(), [len(c) for c in comps])
    
    # #merge compatible adjacent
    # while True:
    #     merged = comps[:1]
    #     for c in comps[1:]:
    #         m = list(merge(merged[-1], c))
    #         if valid(m, MIN_DIST):
    #             merged[-1] = m
    #         else:
    #             merged.append(c)
    #     if len(merged) < len(comps):
    #         comps = merged
    #     else: break
    # print(len(comps), [len(c) for c in comps])
    
    #remove small comps
    # comps = [c for c in comps if len(c) > 10]
    
    # typeseqs = comps_to_seqs(comps, sequences)
    # plot_sequences(typeseqs.copy(), song+'-seqpat...png')
    # 
    # #typeseqs = [l[-2] for l in get_hierarchy_labels(typeseqs)]
    # typeseqs = get_most_salient_labels(typeseqs, 20, [-1])
    # plot_sequences(typeseqs, song+'-seqpat....png')
    
    return comps

def comps_to_seqs(comps, sequences):
    typeseqs = [np.repeat(-1, len(s)) for s in sequences]
    for i,c in enumerate(comps):
        for s in c:
            typeseqs[s[0]][s[1]] = i
    return typeseqs

def smooth_sequences(sequences, min_match=0.8, min_defined=0.6, count=10000):
    max_len = math.inf#30
    min_len = 10
    min_occs = 4
    secs = get_longest_sections(sequences, [-1])
    secs = [s for s in secs if min_len <= len(s[0]) <= max_len
        and s[1] >= min_occs][:count]
    print(len(secs))
    smoothed = [t.copy() for t in sequences]
    smooth_seqs(smoothed, secs, min_match, min_defined)
    #smooth_seqs(smoothed, reversed(secs), min_match, min_defined) #most common have last effect
    
    t, s = np.hstack(sequences), np.hstack(smoothed)
    print(len(np.unique(t)), len(np.unique(s)))
    merged = np.where(np.logical_and(t != -1, t != s))
    merged = Counter([tuple(m) for m in np.vstack((t[merged], s[merged])).T])
    print({e:c for e,c in merged.items() if c >= 5})
    return smoothed

#update sequences with mode of values in comps
def get_mode_sequences(sequences, comps):
    for i,c in enumerate(comps):
        m = mode([sequences[s[0]][s[1]] for s in c])
        for s in c:
            sequences[s[0]][s[1]] = m
    return sequences

def get_grouped_comp_typeseqs(comps, sequences):
    #infer types directly from components
    comp_groups = [[comps[0]]]
    #proj = lambda ts,i: [t[i] for t in ts]
    #compdiff = lambda c,d: [for i in set(proj(c,1)).intersect(set(proj(d,1)))]
    adjmax = get_comp_adjacency(comps, True, 0.5)
    adjmin = get_comp_adjacency(comps, False, 0.8)
    
    comp_groups = group_by_comps(comps, adjmax)
    print([len(g) for g in comp_groups])
    # comp_groups = [g for g in comp_groups if len(g) > 1]\
    #     + [[c for g in comp_groups if len(g) == 1 for c in g]]
    #merge_tiny_comp_groups(comp_groups, adjmax, adjmin, comps)
    print([len(g) for g in comp_groups])
    
    #print([[[s for s in c if s[0] in [4]] for c in g] for g in comp_groups])
    typeseqs = [np.repeat(-1, len(s)) for s in sequences]
    for i,g in enumerate(comp_groups):
        for c in g:
            for s in c:
                typeseqs[s[0]][s[1]] = i
    plot_sequences(typeseqs.copy(), 'seqpat....png')
    
    # #typeseqs = get_most_salient_labels(typeseqs, 30, [-1])
    # typeseqs = get_most_salient_labels(typeseqs, 20, [-1])
    # #typeseqs = [l[1] for l in get_hierarchy_labels(typeseqs)]
    # plot_sequences(typeseqs, 'seqpat.....png')
    
    return [np.array(t) for t in typeseqs]

def merge_tiny_comp_groups(comp_groups, adjmax, adjmin, comps):
    for g in [g for g in comp_groups if len(g) == 1]:
        print()
        candidate = None
        succ = np.nonzero(adjmin[comps.index(g[0])])[0]
        print(succ)
        for s in succ:
            succpred = next((i for i,p in enumerate(adjmax) if p[s] > 0), -1)
            print(succpred)
            if succpred >= 0:
                candidate = comps[succpred]
        pred = next((i for i,p in enumerate(adjmin) if p[comps.index(g[0])] > 0), -1)
        #predgroup = next(i for i,g in enumerate(comp_groups) if comps[pred] in g)
        #print(predgroup)
        print(pred)
        predsucc = np.nonzero(adjmax[pred])[0]
        print(predsucc)
        if len(predsucc) > 0:
            candidate = comps[predsucc[0]]
        print(candidate)
        if candidate:
            candidate.extend(g[0])
            comp_groups.remove(g)

def print_status(title, patterns):
    longest3 = [len(l[1]) for l in sorted(list(patterns.items()),
        key=lambda p: len(p[1]), reverse=True)[:3]]
    print(title, len(patterns), longest3)

#returns the subset of patterns occurring in a given min number of versions
def filter_patterns(patterns, min_versions=0, remove_uniform=False, include=None):
    #remove uniform (all same value or blank)
    if remove_uniform:
        uniq = {k:len(np.unique(list(k))) for k in patterns.keys()}
        patterns = {k:v for k,v in patterns.items()
            if uniq[k] > 2 or (uniq[k] == 2 and -1 not in k)}
        print_status('removed uniform', patterns)
    #prune pattern dict: keep only ones occurring in a min num of versions
    if min_versions > 0:
        patterns = {k:v for k,v in patterns.items()
            if len(np.unique([o[0] for o in v])) >= min_versions}
        print_status('frequent', patterns)
    if include != None:
        include = np.array(list(include))
        pv = set(include[:,0])
        patterns = {p:o for p,o in patterns.items() if contains((p,o), include, pv)}
    return patterns

def contains(pattern, points, points_versions):
    versions = list(set([p[0] for p in pattern[1]]).intersection(points_versions))
    vpoints = points[np.isin(points[:,0], versions)]
    voccs = np.array(list(pattern[1]))
    voccs = voccs[np.isin(voccs[:,0], versions)]
    voccs = np.column_stack((voccs, voccs[:,1]+len(pattern)))
    containspoint = lambda o: np.any(np.logical_and(o[0] == vpoints[:,0],
        np.logical_and(o[1] <= vpoints[:,1], vpoints[:,1] < o[2])))
    return any(containspoint(o) for o in voccs)

def group_patterns(patterns, length=True, cooccurrence=False, similarity=False):
    #group by length
    groups = group_by(patterns.keys(), lambda p: len(p))
    
    #group by cooccurrence (groups not overlapping)
    if cooccurrence:
        eqfunc = lambda p, q: len(patterns[p].intersection(patterns[q])) > 0
        groups = flatten([group_patterns2(eqfunc, g, False) for g in groups], 1)
    
    #group by similarity (groups are overlapping)
    if similarity:
        eqfunc = lambda p, q: all(p[i] == -1 or q[i] == -1 or p[i] == q[i]
            for i in range(len(p)))
        groups = flatten([group_patterns2(eqfunc, g, True) for g in groups], 1)
    
    print('grouped', len(groups), sum([len(g) for g in groups]))
    return groups

def group_patterns2(equiv_func, patterns, cliques_not_comps):
    if len(patterns) > 1:
        matrix = np.array([[1 if equiv_func(p,q) else 0
            for q in patterns] for p in patterns])
        g = graph_from_matrix(matrix)[0]
        if cliques_not_comps:
            equivs = list(gt.max_cliques(g))
        else:
            comps = list(gt.label_components(g)[0].a)
            equivs = group_by(range(len(comps)), lambda v: comps[v])
        union = list(np.hstack(equivs)) if len(equivs) > 0 else []
        return [[patterns[i] for i in e] for e in equivs]\
            +[[patterns[i]] for i in range(len(patterns)) if i not in union]
    return [patterns]

#catalogue all connection counts between equivalent patterns
def get_most_common_connections(groups, patterns, sequences, min_dist, min_count):
    conns = []
    maxlen = max([len(s) for s in sequences])
    total = maxlen*len(sequences)
    #print('conns')
    for g in groups:
        #sort occurrences in group
        o = np.array(sorted(list(set([o for p in g for o in patterns[p]]))))
        #convert to indices of global concatenated sequence
        a = np.array([oo[0]*maxlen+oo[1] for oo in o])
        #get all pairwise connections not within min dist
        i = np.array([[i1,i2] for i1 in range(len(o)) for i2 in range(i1+1, len(o))])
        oo = o[i] #pairs of occurrences
        aa = a[i] #pairs of global indices
        ooo = aa[np.logical_or(oo[:,0,0] != oo[:,1,0],
            np.absolute(oo[:,0,1] - oo[:,1,1]) >= min_dist)]
        #add segment durations
        oooo = np.hstack(ooo[:,:,None] + np.arange(0, len(g[0]))).T
        #to edge indices and append
        ooooo = oooo[:,0]*total + oooo[:,1]
        conns.append(ooooo)
    #concat and count
    c = np.concatenate(conns)
    #print('count', datetime.datetime.now())
    #counts = np.bincount(c)
    edges = Counter(c.tolist())
    print(len(edges))
    edges = {e:c for e,c in edges.items() if c >= min_count}
    print(len(edges))
    #print('sort', datetime.datetime.now())
    bestconns = sorted(edges.items(), key=lambda c: c[1], reverse=True)
    bestconns = np.array(bestconns)[:,0]
    #print('post', datetime.datetime.now())
    pairs = np.vstack((np.floor(bestconns/total), bestconns % total)).T
    #print('post2', datetime.datetime.now())
    return np.dstack((np.floor(pairs/maxlen), pairs % maxlen)).astype(int)

def valid(comp, min_dist):
    diffs = np.diff([c for c in comp], axis=0)
    diffs = np.array([d[1] for d in diffs if d[0] == 0])#same version
    if len(diffs) > 0:
        return np.min(diffs[diffs >= 0]) >= min_dist
    return True

#s1 needs to be pred of s2
def at_min_dist(s1, s2, min_dist):
    return s1[0] != s2[0] or s2[1]-s1[1] >= min_dist

def compatible(comp, s, pos, min_dist):
    return (pos == 0 or at_min_dist(comp[pos-1], s, min_dist))\
        and (pos >= len(comp)-1 or at_min_dist(s, comp[pos], min_dist))

#locs keeps track of which components each segment is in
#incomp keeps track of incompatible component combinations
def add_to_components(edges, comps, locs, incomp, min_dist):
    for pair in edges:
        pair = (pair[0][0], pair[0][1]), (pair[1][0], pair[1][1])
        loc1 = locs[pair[0]] if pair[0] in locs else None
        loc2 = locs[pair[1]] if pair[1] in locs else None
        if loc1 != None and loc2 != None:
            if loc1 != loc2 and (loc1, loc2) not in incomp:
                #print(loc1, loc2, comps[loc1], comps[loc2])
                merged = list(merge(comps[loc1], comps[loc2]))
                #merged = sorted(set(comps[loc1] + comps[loc2]))
                if valid(merged, min_dist):
                    comps[loc1] = merged
                    for o in comps[loc2]:
                        locs[o] = loc1
                    comps[loc2] = [] #keep to not have to update indices
                else:
                    incomp.add((loc1, loc2))
                    #print(incomp)
            #else ignore pair
        elif loc1 != None:
            pos = next((i for i,c in enumerate(comps[loc1]) if c > pair[1]), len(comps[loc1]))
            if compatible(comps[loc1], pair[1], pos, min_dist):
                comps[loc1].insert(pos, pair[1])
                locs[pair[1]] = loc1
        elif loc2 != None:
            pos = next((i for i,c in enumerate(comps[loc2]) if c > pair[0]), len(comps[loc2]))
            if compatible(comps[loc2], pair[0], pos, min_dist):
                comps[loc2].insert(pos, pair[0])
                locs[pair[0]] = loc2
        else:
            locs[pair[0]] = locs[pair[1]] = len(comps)
            comps.append(list(pair))#pair is ordered
        # vvv = [valid(c, min_dist) for c in comps]
        # if not all(vvv):
        #     print('VALID', vvv)
        #     print(pair, loc1, loc2)
        #     print(comps)
        #     break
        #print(loc1, loc2, comps)

def adj_proportion(c, d, maximum=True):
    numadj = len(set(c).intersection(set([(s[0],s[1]-1) for s in d])))
    return numadj/max(len(c),len(d)) if maximum else numadj/min(len(c),len(d))

def get_comp_adjacency(comps, max=True):
    adjacency = np.zeros((len(comps),len(comps)))
    for i,c in enumerate(comps):
        for j,d in enumerate(comps):
            adjacency[i][j] = adj_proportion(c, d, max)
    return adjacency

def cleanup_comps(comps, sequences, path):
    print('bf', len(flatten(comps, 1)))
    print([len(c) for c in comps])
    scomps = group_by_maxadj(comps, sequences, path)
    
    #mixed up are removed automatically by individual seq grouping algo
    seqs = get_individual_seqs(scomps, len(sequences))
    #print_individual_seqs(seqs)
    print('bf', len(flatten(scomps, 2)))
    print([len(c) for c in flatten(scomps, 1)])
    #separate sparse segments...
    for i,s in enumerate(seqs):
        if len(flatten(s, 1)) > 0:
            newc = [[] for x in flatten(s, 1)[0]]
            for t in flatten(s, 1):
                defined = [seg for seg in t if seg > ()]
                if len(defined) <= len(t)/2:
                    for j,seg in enumerate(t):
                        if seg > ():
                            scomps[i][j].remove(seg)
                            newc[j].append(seg)
            scomps.extend([[c] for c in newc if len(c) > 0])
    #add missing
    locs = {o:(i,j) for i,s in enumerate(scomps) for j,t in enumerate(s) for o in t}
    for i,s in sorted(zip(range(len(seqs)), seqs), key=lambda s: len(s[1][0])):
        for t in flatten(s, 1):
            defined = [seg for seg in t if seg > ()]
            if len(defined) > len(t)/2:
                offsets = [seg[1]-j for j,seg in enumerate(t) if seg > ()]
                index = next(seg[0] for seg in t if seg > ())
                if len(set(offsets)) <= 1: #all offsets the same
                    for j,seg in enumerate(t):
                        if seg == () and j+offsets[0] < len(sequences[index]):
                            scomps[i][j].append((index, j+offsets[0]))
                            #remove from old comp
                            if t[j] in locs:
                                l = locs[t[j]]
                                #print(t[j], l, offsets, j, t)
                                scomps[l[0]][l[1]].remove(t[j])
                                locs[t[j]] = (i,j)
    scomps = [s for s in scomps if len(s) > 0]
    
    # print('af', len(flatten(scomps, 2)))
    # print([len(c) for c in flatten(scomps, 1)])
    # scomps = group_by_maxadj(flatten(scomps, 1), sequences, path+'r')
    # print('af', len(flatten(scomps, 2)))
    # print([len(c) for c in flatten(scomps, 1)])
    return scomps
    
    # seqs = get_individual_seqs(scomps, len(sequences))
    # print_individual_seqs(seqs)
    # print(len(flatten(scomps, 2)))
    # print_scomp_gaps(scomps, len(sequences))
    # plot_seq_x_comps(flatten(scomps, 1), sequences, path)
    # 
    # adjmax = get_comp_adjacency(flatten(scomps, 1), True)
    # plot_matrix(adjmax, path+'-max3.png')
    # print_offdiasum(adjmax)

def group_by_maxadj(comps, sequences, path):
    #sort by adjacency
    adjmax = get_comp_adjacency(comps, True)
    plot_matrix(adjmax, path+'-max.png')
    #print_offdiasum(adjmax)
    # adjmax = get_comp_adjacency(comps, True)
    # plot_matrix(adjmax, 'maxl.png')
    #maxes = list(zip(*np.nonzero(adjmax)))
    maxes = list(zip(*np.where(adjmax > 0.5)))
    maxes = sorted(maxes, key=lambda ij: adjmax[ij], reverse=True)
    #print(len(list(zip(*np.nonzero(adjmax)))))
    #print(len(maxes), maxes[:10])
    #print([(comps[i], comps[j]) for i,j in maxes[:5]])
    scomps = []
    for i,j in maxes:
        ii = next((k for k,s in enumerate(scomps) if comps[i] in s), -1)
        jj = next((k for k,s in enumerate(scomps) if comps[j] in s), -1)
        #print(i, j, ii, jj)
        if ii >= 0 and jj >= 0:
            if ii != jj and scomps[ii][-1] == comps[i] and scomps[jj][0] == comps[j]:
                scomps[ii].extend(scomps[jj])
                scomps.pop(jj)
            else:
                iii, jjj = scomps[ii].index(comps[i]), scomps[jj].index(comps[j])
                li, lj = len(scomps[ii]), len(scomps[jj])
                r = range(-min(iii, jjj)-1, min(li-iii, lj-jjj)-1)
                merged = [list(merge(scomps[ii][iii+k+1], scomps[jj][jjj+k])) for k in r]
                v = [valid(m, MIN_DIST) for m in merged]
                #print(v, all(v))
                if all(v):
                    print(len(scomps[ii]), len(scomps[jj]), iii, jjj, r, len(merged),
                        len(scomps[ii][:iii+1] + merged + scomps[jj][jjj+len(r):]))
                    scomps[ii] = scomps[ii][:iii+1] + merged + scomps[jj][jjj+len(r):]
                    scomps.pop(jj)
                # else:
                # 
                #if not valid: split
        elif ii >= 0:
            if scomps[ii][-1] == comps[i]:
                scomps[ii].append(comps[j])
        elif jj >= 0:
            if scomps[jj][0] == comps[j]:
                scomps[jj].insert(0, comps[i])
        else:
            scomps.append([comps[i], comps[j]])
        #print([[comps.index(c) for c in s] for s in scomps])
    #scomps = flatten(scomps, 1)
    #comps = scomps+[c for c in comps if c not in scomps]#add missing
    #print('VALID', [valid(c, MIN_DIST) for c in flatten(scomps, 1)])
    #scomps = scomps+[[c for c in comps if c not in scomps]]#add missing
    scomps = sorted(scomps, key=lambda c: np.mean([s[1] for s in c[0]]))
    
    #print(len(flatten(scomps, 1)), datetime.datetime.now(), [len(c) for c in flatten(scomps, 1)])
    adjmax = get_comp_adjacency(flatten(scomps, 1), True)
    plot_matrix(adjmax, path+'-max2.png')
    #print_offdiasum(adjmax)
    
    # adjmin = get_comp_adjacency(flatten(scomps, 1), False)
    # plot_matrix(adjmin, 'min.png')
    
    return [s for s in scomps if len(s) > 0]

def print_offdiasum(matrix):
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    print('offdia', np.sum(matrix))

def diff(c, d, num_seqs):
    df = []
    for i in range(num_seqs):
        ci = [o[1] for o in c if o[0] == i]
        di = [o[1] for o in d if o[0] == i]
        if len(ci) > 0 and len(di) > 0:
            df.append(min([abs(b-a) for a in ci for b in di]))
    return mode(df)

def get_individual_seqs(scomps, num_seqs):
    seqs = []
    for sc in scomps:
        seqs.append([])
        for i in range(num_seqs):
            occs = [[o for o in c if o[0] == i] for c in sc]
            seq = []
            prev = (), -1
            while len(flatten(occs, 1)) > 0:
                for j in range(len(occs)):
                    nexx = occs[j].pop(0) if len(occs[j]) > 0 \
                        and occs[j][0] == min(flatten(occs, 1)) else ()
                    #jump to next level if gap too large (> 2)
                    if prev[0] > () and nexx > () \
                            and np.subtract(nexx, prev[0])[1] > len(seq)-prev[1]+1:
                        seq.extend([() for j in range(len(occs))])
                    seq.append(nexx)
                    if nexx > ():
                        prev = nexx, len(seq)-1
            seq = split(seq, len(sc))
            seqs[-1].append([s for s in seq if any(e > () for e in s)])
    return seqs

def print_individual_seqs(seqs):
    for s in seqs:
        print('------------')
        for i in s[:1]:
            print('')
            [print(s) for s in i]

def print_scomp_gaps(scomps, num_seqs):
    for s in scomps:
        print([diff(c, d, num_seqs) for c, d in zip(s[:-1], s[1:])])
        # for i, (c,d) in enumerate(zip(s[:-1], s[1:])):
        #     if diff(c,d) > 1:
        # 
    

def plot_seq_x_comps(comps, sequences, path):
    def plot_seq(k):
        alo = np.zeros((len(comps), len(sequences[k])))
        for j in range(len(sequences[k])):
            i = next((i for i,c in enumerate(comps) if (k,j) in c), -1)
            if i >= 0:
                alo[i][j] = 1
        plot_matrix(alo, path+'-maxs'+str(k)+'.png')
    plot_seq(0)
    plot_seq(1)
    plot_seq(2)
    plot_seq(3)

#returns matching proportion (need to be same length)
def sim(s1, s2):
    blank = np.logical_or(s1 == -1, s2 == -1)
    same_or_blank = np.nonzero(np.logical_or(blank, s1 == s2))
    return len(same_or_blank[0]) / len(s1), len(np.nonzero(blank)[0]) / len(s1)

def smooth_seqs(sequences, sections, min_match, min_defined):
    sections = [s[0] for s in sections]
    #print(sections)
    for s in tqdm.tqdm(sequences, desc='smoothing'):
        matches = []
        for i,c in enumerate(sections):
            #print(c)
            r = range(len(s)-len(c))
            sims = [sim(c, s[j:j+len(c)]) for j in r]
            if len(sims) > 0:
                matched, blank = zip(*sims)
                matched, blank = np.array(list(matched)), np.array(list(blank))
                matched[np.where(np.logical_or(
                    matched < min_match, (1-blank) < min_defined))] = 0
                m = len(matched)
                matched = np.vstack((matched, np.repeat(i, m), np.arange(m))).T
                matches.append(matched[np.where(matched[:,0] > 0)])
        matches = np.concatenate(matches)
        matches = matches[matches[:,0].argsort()]
        for p,c,j in matches:
            cc = sections[int(c)]
            s[int(j):int(j)+len(cc)] = cc

# adjmax = np.array([[0,1,0,0],[0,0,0,1],[0,0,1,0],[0,0,0,1]])
# comps = [0,1,2,3]
# plot_matrix(adjmax, 'max.png')
# remaining = comps.copy()
# scomps = []
# next = 0
# while len(remaining) > 0:
#     current = comps[next] if next > 0 else remaining[0]
#     print(next, current)
#     scomps.append(current)
#     remaining.remove(current)
#     next = np.argmax(adjmax[comps.index(current)])
# comps = scomps
# adjmax = get_comp_adjacency(comps, True, 0.5)
# plot_matrix(adjmax, 'max2.png')
# print(len(comps))