import datetime, math, statistics, tqdm, os, psutil, sys
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
from ..alignment.affinity import segments_to_matrix, get_alignment_segments
from ..util import plot_sequences, mode, flatten, group_by, plot_matrix,\
    multiprocess, split
from ..stats.histograms import freq_hist_clusters, trans_hist_clusters,\
    freq_trans_hist_clusters
from .graphs import graph_from_matrix
from .hierarchies import get_hierarchy_labels, get_most_salient_labels,\
    get_longest_sections
from .similarity import sw_similarity, isect_similarity, cooc_similarity

MIN_VERSIONS = 0.2 #how many of the versions need to contain the patterns
PARSIM = True
PARSIM_DIFF = 0 #the largest allowed difference in parsimonious mode (full containment == 0)
MIN_SIM = 0.9 #min similarity for non-parsimonious similarity
COMPS_NOT_BLOCKS = True #use connected components instead of community blocks


def get_occ_matrix(pattern, dict, num_seqs, max_len):
    matrix = np.zeros((num_seqs, max_len))
    for occ in dict[pattern]:
        matrix[occ[0], np.arange(occ[1], occ[1]+len(pattern))] = 1
    return matrix

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
    return {p:o for p,o in
        create_pattern_list(sequences, pairings, alignments, True)}

#tuples with patterns with -1 for blanks and occurrences (sequence index, position)
#all patterns remain separate, duplicates with different occurrences exist
def create_pattern_list(sequences, pairings, alignments, combine_similar=False,
        min_versions=0, remove_uniform=False, include=None):
    patterns = []
    id = 0
    for i,a in enumerate(alignments):
        for s in a:
            j, k = pairings[i]
            s1, s2 = sequences[j][s.T[0]], sequences[k][s.T[1]]
            if isinstance(s1[0], np.integer):
                pattern = tuple(np.where(s1 == s2, s1, -1))
            else:
                pattern = tuple(np.repeat(id, len(s1)))
            locations = [(j, s[0][0]), (k, s[0][1])]
            patterns.append((pattern, set(locations)))
            id += 1
    if combine_similar:
        pdict = dict()
        for pattern, occs in enumerate(patterns):
            if not pattern in pdict:
                pdict[pattern] = set()
            pdict[pattern].update(occs)
        patterns = list(pict.items())
    return filter_patterns(patterns, min_versions, remove_uniform, include)

#returns the subset of patterns occurring in a given min number of versions
def filter_patterns(patterns, min_versions=0, remove_uniform=False, include=None):
    #remove uniform (all same value or blank)
    if remove_uniform:
        uniq = [len(np.unique(list(p[0]))) for p in patterns]
        patterns = [p for i,p in enumerate(patterns)
            if uniq[i] > 2 or (uniq[i] == 2 and -1 not in p[0])]
        print_status('removed uniform', patterns)
    #prune pattern dict: keep only ones occurring in a min num of versions
    if min_versions > 0:
        patterns = [p for p in patterns
            if len(np.unique([o[0] for o in p[1]])) >= min_versions]
        print_status('frequent', patterns)
    if include != None:
        include = np.array(list(include))
        pv = set(include[:,0])
        patterns = [p for p in patterns if contains(p, include, pv)]
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
MAX_OCCS = 300000
INIT_MIN_COUNT = 5#7
MIN_COUNT = 2
MAX_MIN_SIZE = 0#20

def super_alignment_graph(song, sequences, pairings, alignments):
    print(psutil.Process(os.getpid()).memory_info())
    all_patterns = create_pattern_dict(sequences, pairings, alignments, remove_uniform=True)
    all_points = set([(i,j) for i,s in enumerate(sequences) for j in range(len(s))])
    print_status('all', all_patterns)
    print(len(all_points))
    print(psutil.Process(os.getpid()).memory_info())
    
    groups = group_patterns(all_patterns, length=True, cooccurrence=True, similarity=False)
    #groups to new patterns
    all_patterns = groups_to_patterns(groups)
    
    #sort by num different versions...
    # all_ordered = sorted(all_patterns,
    #     key=lambda p: len(np.unique([v for v,t in p[1]])), reverse=False)
    
    #sort by average first time point
    avg_first = lambda p: np.mean([min([o[1] for o in p[1] if o[0] == i])
        for i in np.unique([o[0] for o in p[1]])])
    all_ordered = sorted(all_patterns, key=avg_first)
    
    #shortest first
    #all_ordered = sorted(all_patterns, key=lambda p: len(p[0]))
    
    
    # groups = group_patterns(all_patterns, length=True)
    # groups = [c for g in groups for c in cluster(g, True)[0]]
    # groups = sorted(groups, key=lambda g: sum([len(all_patterns[p]) for p in g]), reverse=True)
    # all_ordered = [(p,all_patterns[p]) for p in flatten(groups, 1)]
    # print("clustered", len(groups), statistics.median([len(g) for g in groups]))
    
    #all_ordered = all_patterns
    
    #print(by_versions[0])
    comps = []
    locs = {}
    incomp = set()
    remaining = all_patterns.copy()
    ordered = all_ordered.copy()
    remaining_points = all_points
    min_count = INIT_MIN_COUNT
    min_size = 0
    scanning = True
    while len(remaining) > 0 and len(ordered) > 0:
        #filter patterns
        cutoff = np.argmax(np.cumsum([len(o)*len(p) for p,o in ordered]) > MAX_OCCS)
        if cutoff == 0:#always at least 5, or all remaining if no cutoff
            cutoff = 5 if len(ordered[0][1]) > MAX_OCCS else len(ordered)
        patterns = [p for p in ordered[:cutoff]]
        print('selected', len(patterns), 'of', len(ordered))
        
        if len(patterns) > 0:
            
            groups = group_patterns(patterns, length=True, cooccurrence=True, similarity=False)
            print(psutil.Process(os.getpid()).memory_info())
            
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
            if not scanning and (len(remaining_points) == previous or len(patterns) == len(ordered)):
                if min_size < MAX_MIN_SIZE:
                    min_size += 10
                    #remaining = all_patterns #need to widen selection
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
                scanning = True
                remaining = [p for p in remaining if p not in patterns]#try scanning through
                if len(remaining) == 0:
                    scanning = False
                    remaining = filter_patterns(all_patterns, include=remaining_points)
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

def super_alignment_graph2(song, sequences, pairings, alignments):
    all_patterns = create_pattern_list(sequences, pairings, alignments)#, remove_uniform=True)
    print_status('all', all_patterns)
    
    groups = group_patterns(all_patterns, length=False, cooccurrence=False, similarity=False)
    all_patterns = groups_to_patterns(groups)
    
    matrix = get_connection_matrix(sequences, all_patterns)
    
    return matrix_to_components(matrix, sequences)

def alignment_csr_matrix(sequences, pairings, alignments):
    lens = [len(s) for s in sequences]
    locs = np.cumsum([0]+lens)
    size = sum(lens)
    conns = []
    for p,a in zip(pairings[1:], alignments[1:]):
        a0, b0 = locs[p[0]], locs[p[1]]
        [conns.append(s + [a0, b0]) for s in a]
    matrix = conns_to_matrix(conns, size)
    return matrix+matrix.T #make symmetric for fast queries

def super_alignment_graph3(song, sequences, pairings, alignments):
    print(len(np.concatenate(flatten(alignments, 1))))
    matrix = alignment_csr_matrix(sequences, pairings, alignments)
    print(len(matrix.data))
    matrix = [matrix.getrow(i) for i in range(matrix.shape[0])]
    #print(matrix.data.nbytes, sys.getsizeof(matrix))
    #try all longest segments first!
    #mutual = [a for i,a in enumerate(alignments) if pairings[i][0] != pairings[i][1]]
    #for m in mutual:
    patterns = create_pattern_list(sequences, pairings, alignments)
    patterns = sorted(patterns, key=lambda p: len(p[0]), reverse=True)
    
    seqlens = [len(s) for s in sequences]
    seqlocs = np.cumsum([0]+seqlens)
    seqid = np.hstack([np.repeat(i,l) for i,l in enumerate(seqlens)])
    size = sum([len(s) for s in sequences])
    # matrix = csr_matrix((size, size), dtype='int8')
    comps = []
    locs = np.repeat(-1, size)
    
    aligned = conns_to_matrix([pairings], len(sequences)).toarray()
    aligned = aligned+aligned.T
    
    #returns the proportion of ps aligned with p
    connectedness = lambda p, ps: np.sum(matrix[p].toarray()[0][ps])\
        / np.sum(aligned[seqid[p],seqid[ps]])
    
    incomp = set()#buffer incompatible components
    
    for p in tqdm.tqdm(patterns):#[:100000]):
        o = list(p[1])
        #print(o)
        v1, v2, o1, o2 = o[0][0], o[1][0], o[0][1], o[1][1]
        
        ids = np.arange(len(p[0]))
        pos = np.vstack((seqlocs[v1] + o1 + ids, seqlocs[v2] + o2 + ids)).T
        loc = locs[pos]
        
        conns = np.zeros((len(p[0]),2))
        unequal = loc[:,0] != loc[:,1]
        diff = np.absolute(loc[:,0] - loc[:,1])
        #NARROW DOWN STEP BY STEP TO SPEED UP!!!!!!! HOW ELSE??????
        check = unequal & (diff > 2)#diff > 1 speeds up significantly (although not perfectly sound)
        
        dists = np.zeros((len(p[0])))
        for i in range(len(p[0])):
            if check[i]:
                l1, l2 = loc[i]
                if l1 >= 0 and l2 >= 0 and (l1, l2) in incomp:
                    dists[i] = 0
                else:
                    c1 = comps[l1] if l1 >= 0 else np.array([pos[i,0]])
                    c2 = comps[l2] if l2 >= 0 else np.array([pos[i,1]])
                    dists[i] = np.min(np.absolute(np.diff(snp.merge(c1, c2))))
                    if dists[i] < MIN_DIST and l1 >= 0 and l2 >= 0:
                        incomp.add((l1, l2))
        
        check = check & (dists >= MIN_DIST)
        check1 = check & (loc[:,0] >= 0)
        check2 = check & (loc[:,1] >= 0)
        # l1, l2, l3 = len(np.nonzero(unequal)[0]), len(np.nonzero(check1)[0]), len(np.nonzero(check2)[0])
        # if sum([l1,l2,l3]) > 0:
        #     print(l1,l2,l3, loc[:3,0], loc[:3,1])
        #print(len(np.where(unequal)[0]), len(np.where(check1)[0]), len(np.where(check2)[0]))
        conns[check1, 0] = np.array([connectedness(pos[i,1], comps[loc[i,0]])
            for i in np.where(check1)[0]])
        conns[check2, 1] = np.array([connectedness(pos[i,0], comps[loc[i,1]])
            for i in np.where(check2)[0]])
        #print(conns[:,0][np.nonzero(conns[:,0])])
        avgconns = [
            np.mean(conns[:,0][np.nonzero(conns[:,0])]),
            np.mean(conns[:,1][np.nonzero(conns[:,1])])]
        #print(conns)
        
        if any([a >= 0.5 for a in avgconns]) or np.all(loc == -1):
            for i in range(len(p[0])):
            #if True:#np.any(conns[i] > 0) or np.all(loc[i] == -1):
                loc = locs[pos]
                p1, p2, l1, l2 = pos[i,0], pos[i,1], loc[i,0], loc[i,1]
                d1, d2 = loc[:,0] >= 0, loc[:,1] >= 0
                if d1[i] and not d2[i]:
                    #if conns[i,0] == 1:
                    if dists[i] >= MIN_DIST:
                        union = snp.merge(comps[l1], np.array([p2]), duplicates=snp.DROP)
                        if np.min(np.absolute(np.diff(union))) >= MIN_DIST:#check again in case several merges in one iteration
                            locs[p2] = l1
                            comps[l1] = union
                elif d2[i] and not d1[i]:
                    #if conns[i,1] == 1:
                    if dists[i] >= MIN_DIST:
                        union = snp.merge(comps[l2], np.array([p1]), duplicates=snp.DROP)
                        if np.min(np.absolute(np.diff(union))) >= MIN_DIST:#check again in case several merges in one iteration
                            locs[p1] = l2
                            comps[l2] = union
                elif not (d1[i] or d2[i]): #both -1
                    locs[p1] = locs[p2] = len(comps)
                    comps.append(np.unique(np.array([p1, p2], dtype=int)))
                elif l1 != l2:
                    #if np.all(conns[i] == 1):
                    if dists[i] >= MIN_DIST:
                        union = snp.merge(comps[l1], comps[l2], duplicates=snp.DROP)
                        if np.min(np.absolute(np.diff(union))) >= MIN_DIST: #check again in case several merges in one iteration
                            comps[l1] = union
                            comps[l2] = np.array([], dtype=int)
                            locs[comps[l1]] = l1
    
    to_point = lambda id: (seqid[id], id-seqlocs[seqid[id]])
    comps = [[to_point(p) for p in c] for c in comps if len(c) > 0]
    comps = sorted(comps, key=lambda c: np.mean([s[1] for s in c]))
    #print(comps)
    return comps

def get_connection_matrix(sequences, all_patterns):
    lengths = [len(s) for s in sequences]
    #total = sum(lengths)
    #print(total**2)
    maxlen = max(lengths)
    size = maxlen*len(sequences)
    #print(size**2)
    versions = np.insert(np.cumsum(lengths), 0, 0)
    conns = []
    matrix = csr_matrix((size, size), dtype='int8')
    for pattern,occs in tqdm.tqdm(all_patterns, desc='creating matrix'):
        #sort occurrences in group and convert to matrix indices
        occs = np.array(sorted(occs))
        locs = np.array([(o[0]*maxlen)+o[1] for o in occs])
        #get all pairwise connections that are not within min dist
        i = np.dstack(np.nonzero(np.triu(np.ones((len(occs),len(occs))), k=1)))[0]
        occs = occs[i] #pairs of occurrences
        locs = locs[i] #pairs of indices
        locs = locs[np.logical_or(occs[:,0,0] != occs[:,1,0],
            np.absolute(occs[:,0,1] - occs[:,1,1]) >= MIN_DIST)]
        #add all connections within segment durations
        #conns.append(np.hstack(locs[:,:,None] + np.arange(0, len(pattern))).T)
        locs = locs[:,:,None] + np.arange(0, len(pattern))
        conns.append(np.reshape(np.transpose(locs, (0,2,1)), (-1,2)))
        if len(conns) >= 500: #dump every 500 patterns to save memory
            matrix += conns_to_matrix(conns, size)
            conns = []
    matrix += conns_to_matrix(conns, size)
    return matrix

def conns_to_matrix(conns, size):
    conns = np.concatenate(conns)
    data = (np.repeat(1, len(conns)), (conns[:,0], conns[:,1]))
    return csr_matrix(data, (size, size), 'int8')

def matrix_to_components(matrix, sequences):
    maxlen = max([len(s) for s in sequences])
    comps = []
    locs = {}
    incomp = set()
    for i in tqdm.tqdm(range(np.max(matrix), MIN_COUNT-1, -1), desc='creating components'):
        conns = np.vstack(np.nonzero(matrix == i)).T
        edges = np.dstack((np.floor(conns/maxlen), conns % maxlen)).astype(int)
        add_to_components(edges, comps, locs, incomp, MIN_DIST)
    comps = [c for c in comps if len(c) > 0]
    comps = sorted(comps, key=lambda c: np.mean([s[1] for s in c]))
    return comps

def comps_to_seqs(comps, sequences):
    typeseqs = [np.repeat(-1, len(s)) for s in sequences]
    for i,c in enumerate(comps):
        for s in c:
            typeseqs[s[0]][s[1]] = i
    return typeseqs

def alignment_segs(sequences):
    return get_alignment_segments(sequences[0], sequences[1], 0, 16, 1, 4, .2)

def realign_gaps(sequences, labelseqs, comps):
    locs = {p:i for i,c in enumerate(comps) for p in c}
    gaps = get_gaps_by_length(labelseqs)
    #print(gaps[:5])
    for i,g in enumerate(gaps):
        gapseq = sequences[g[0]][g[1]:g[1]+g[2]]
        segs = multiprocess('', alignment_segs, [(gapseq,s) for s in sequences])
        conns = [[] for p in range(g[2])]
        [conns[p[0]].append(locs[(i,p[1])])
            for i,s in enumerate(segs) for a in s for p in a if (i,p[1]) in locs]
        #conns = [mode(c) if len(c) > 0 else -1 for c in conns]
        for j,c in enumerate(conns):
            if len(c) > 0:
                labelseqs[g[0]][g[1]+j] = mode(c, strict=True)
        #print(conns)
        #plot_matrix(segments_to_matrix(segs[1]))
        #print(np.concatenate(flatten(s, 1))
    return labelseqs

def realign_gaps_comps(sequences, labelseqs, comps):
    #comps = flatten(group_by_maxadj(comps, sequences), 1)
    modeseq = get_comp_modes(sequences, comps)
    plot_matrix(segments_to_matrix(alignment_segs((sequences[0], modeseq))))
    gaps = get_gaps_by_length(labelseqs)
    print(gaps[:5])
    bestsec = get_best_minlen(get_longest_sections(labelseqs, [-1]), 500)[0]
    #print(bestsec)
    modeseq = np.array([mode([sequences[s[0]][s[1]] for s in comps[c]]) for c in bestsec])
    #print(alignment_segs((sequences[0], modeseq)))
    #plot_matrix(segments_to_matrix(alignment_segs((sequences[0], modeseq))))
    for i,g in enumerate(gaps):
        gapseq = sequences[g[0]][g[1]:g[1]+g[2]]
        segs = alignment_segs((gapseq, modeseq))
        conns = [[] for p in range(g[2])]
        [conns[p[0]].append(bestsec[p[1]]) for s in segs for p in s]
        for j,c in enumerate(conns):
            if len(c) > 0:
                labelseqs[g[0]][g[1]+j] = mode(c, strict=True)
    return labelseqs

#returns a list of gaps (sequence index, position, length), longest first
def get_gaps_by_length(sequences):
    gaps = [get_gaps(l) for l in sequences]
    gaps = [(i,g[0],len(g)) for i,gs in enumerate(gaps) for g in gs if len(g) > 0]
    return sorted(gaps, key=lambda g: g[2], reverse=True)#longest first

def get_gaps(sequence):
    return split_at_jumps(np.where(sequence == -1)[0])

def split_at_jumps(sequence):
    continuous = np.nonzero(np.diff(sequence) > 1)[0]+1
    return np.split(sequence, continuous)

def smooth_sequences(sequences, min_match=0.8, min_defined=0.6, min_len=10, min_occs=2, count=10000):
    max_len = math.inf#30
    secs = get_longest_sections(sequences, [-1])
    secs = [s for s in secs if min_len <= len(s[0]) <= max_len
        and s[1] >= min_occs][:count]
    #secs = sorted(secs, key=lambda s: len(s[0]), reverse=True)
    print(len(secs))
    smoothed = [t.copy() for t in sequences]
    smoothed = smooth_seqs2(smoothed, secs, min_match, min_defined)
    #smooth_seqs(smoothed, secs, min_match, min_defined)
    
    t, s = np.hstack(sequences), np.hstack(smoothed)
    print(len(np.unique(t)), len(np.unique(s)))
    merged = np.where(np.logical_and(t != -1, t != s))
    merged = Counter([tuple(m) for m in np.vstack((t[merged], s[merged])).T])
    print({e:c for e,c in merged.items() if c >= 5})
    return smoothed

def smooth_seqs(sequences, sections, min_match, min_defined):
    for s in tqdm.tqdm(sequences, desc='smoothing'):
        matches = []
        for i,c in enumerate(sections):
            #print(c)
            r = range(len(s)-len(c[0]))
            sims = [sim(c[0], s[j:j+len(c[0])]) for j in r]
            if len(sims) > 0:
                matched, blank = zip(*sims)
                matched, blank = np.array(list(matched)), np.array(list(blank))
                matched[np.where(np.logical_or(
                    matched < min_match, (1-blank) < min_defined))] = 0
                m = len(matched)
                matched = np.vstack((matched, np.repeat(i, m), np.arange(m))).T
                matches.append(matched[np.where(matched[:,0] > 0)])
        #sort by increasing certainty and importance and apply in this order (best have last say)
        matches = np.concatenate(matches)
        #matches = matches[np.lexsort((len(sections)-matches[:,1], matches[:,0]))]#longest of best
        matches = matches[np.lexsort((len(sections)-matches[:,1], len(sections)-matches[:,1]))]#just longest
        for p,c,j in matches:
            cc = sections[int(c)][0]
            s[int(j):int(j)+len(cc)] = cc

def smooth_seqs2(sequences, sections, min_match, min_defined):
    #sections = sorted(sections, key=lambda s: len(s[0]), reverse=True)
    smoothlocs = [np.zeros(len(s), dtype=int) for s in sequences]
    avglen = np.mean([len(s) for s in sequences])
    for l in [0.75, 0.5, 0.4, 0.3, 0.2]:
        best = get_best_minlen(sections, l*avglen)
        if best and best[1] >= 5:
            print(l, avglen, len(best[0]), best[1])
            smooth(smoothlocs, sequences, best)
    return sequences

def get_best_minlen(sections, minlen):
    sections = [s for s in sections if len(s[0]) >= minlen]
    if len(sections) > 0:
        return sections[np.argmax([s[1] for s in sections])]

def smooth(smoothlocs, sequences, section):
    for k,s in enumerate(tqdm.tqdm(sequences[:], desc='smoothing')):
        # r = range(len(s)+len(section[0])-1)
        # b = [section[0][max(len(section[0])-j-1, 0):] for j in r]
        # m = [s[max(j+1-len(section[0]), 0):] for j in r]
        r = range(len(s)-1)
        w = min(round(len(section[0])/2), round(len(s)/2))
        b = [section[0][max(w-j-1, 0):] for j in r]
        m = [s[max(j+1-w, 0):] for j in r]
        # print(b[:2], m[:2])
        # print(b[-2:], m[-2:])
        sims = [sim(p[0], p[1]) for p in zip(b, m)]
        if len(sims) > 0:
            matched, blank, pos = zip(*sims)
            matched = np.array(list(matched))
            matched[:w] *= (w+np.arange(w))/len(section[0])
            matched[-w:] *= (w+np.flip(np.arange(w)))/len(section[0])
            maxx = np.argmax(matched)
            #print(k, matched[maxx])
            if matched[maxx] >= .95:
                update_sequence(smoothlocs[k], s, section, w, maxx)
            else:
                for i in np.flip(np.argsort(matched))[:50]:
                    #print(maxx, i)
                    segs = split_at_jumps(pos[i])
                    #print(matched[i], blank[i])
                    #print(pos[i])
                    #print(len(segs))
                    #print([(i,j) for i in range(len(segs)) for j in range(i+1, len(segs)+1)])
                    segs = [np.concatenate(segs[i:j])
                        for i in range(len(segs)) for j in range(i+1, len(segs)+1)]
                    #print(len(segs))
                    segs = [s for s in segs if len(s) > 0 and len(s)/(s[-1]-s[0]+1) >= 0.95]
                    if len(segs) > 0:
                        #best = sorted(segs, key=lambda s: len(s), reverse=True)
                        best = sorted(segs, key=lambda s: len(s)*(len(s)/(s[-1]-s[0]+1)), reverse=True)
                        #print([len(b)/(b[-1]-b[0]+1) for b in best])
                        #print(best[0], len(best[0])/(best[0][-1]-best[0][0]+1))
                        start, end = best[0][0], best[0][-1]
                        #print(i, i+start, end-start+1, w)
                        update_sequence(smoothlocs[k], s, section, w, i, start, end-start+1)
                
                #NOW SEARCH FOR MATCHES FOR REST OF SEQ...
            #print(np.flip(np.sort(matched))[:10])
            #print(np.sum(matched))

def update_sequence(updated, sequence, section, w, offset, start=0, length=None):
    sec_offset = max(w-offset-1, 0)+start
    sec_end = sec_offset+length if length else None
    #print(sec_offset, sec_end)
    sec = section[0][sec_offset:sec_end]
    seq_offset = max(offset+1-w, 0)+start
    seqlen = len(sequence[seq_offset:seq_offset+len(sec)])
    #print(sec_offset, len(sec), seq_offset, seqlen,
    #    len(np.nonzero(updated[seq_offset:seq_offset+len(sec)])[0]) == 0)
    # if len(np.nonzero(updated[seq_offset:seq_offset+len(sec)])[0]) == 0:
    #     sequence[seq_offset:seq_offset+len(sec)] = sec[:seqlen]
    #     updated[seq_offset:seq_offset+len(sec)] = np.repeat(1, seqlen)
    s, e = seq_offset, seq_offset+len(sec)
    sequence[s:e] = np.where(updated[s:e] == 0, sec[:seqlen], sequence[s:e])
    updated[s:e] = np.repeat(1, seqlen)

#returns matching proportion (match prop, gap prop)
def sim(s1, s2):
    s1 = s1[:min(len(s1), len(s2))]
    s2 = s2[:min(len(s1), len(s2))]
    blank = np.logical_or(s1 == -1, s2 == -1)
    same_or_blank = np.nonzero(np.logical_or(blank, s1 == s2))
    return len(same_or_blank[0]) / len(s1), len(np.nonzero(blank)[0]) / len(s1),\
        same_or_blank[0]

#update sequences with mode of values in comps
def get_mode_sequences(sequences, comps):
    modes = get_comp_modes(sequences, comps)
    for i,c in enumerate(comps):
        for s in c:
            sequences[s[0]][s[1]] = modes[i]
    return sequences

def get_comp_modes(sequences, comps):
    return np.array([mode([sequences[s[0]][s[1]] for s in c]) for c in comps])

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
    longest3 = [len(l[1]) for l in sorted(patterns,
        key=lambda p: len(p[1]), reverse=True)[:3]]
    print(title, len(patterns), longest3)

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
    if length:
        groups = group_by(patterns, lambda p: len(p[0]))
        print('grouped', len(groups), sum([len(g) for g in groups]))
    else:
        groups = [[p] for p in patterns]
    
    #group by cooccurrence (groups not overlapping)
    if cooccurrence:
        # eqfunc = lambda p, q: len(patterns[p].intersection(patterns[q])) > 0
        # groups = flatten([group_patterns2(eqfunc, g, False) for g in groups], 1)
        groups = flatten([fast_group_by_cooc(g) for g in groups], 1)
    
    #group by similarity (groups are overlapping)
    if similarity:
        eqfunc = lambda p, q: all(p[i] == -1 or q[i] == -1 or p[i] == q[i]
            for i in range(len(p)))
        groups = flatten([group_patterns2(eqfunc, g, True) for g in groups], 1)
    
    print('grouped', len(groups), sum([len(g) for g in groups]))
    return groups

def group_patterns2(equiv_func, patterns, cliques_not_comps):
    if len(patterns) > 1:
        matrix = np.array([[1 if equiv_func(p[0],q[0]) else 0
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

def groups_to_patterns(groups):
    return [(g[0][0], set().union(*[p[1] for p in g])) for g in groups]

def fast_group_by_cooc(patterns):
    #group index for each pattern (first all different)
    locs = np.arange(len(patterns))
    #dict with patterns by occurrence
    occs = defaultdict(list)
    for i,p in enumerate(patterns):
        for o in p[1]:
            occs[o].append(i)
    #iteratively set all group indices of cooccurring patterns to min index
    for ps in occs.values():
        ids = np.unique(locs[ps]) #find all involved group ids
        if len(ids) > 1:
            locs[np.isin(locs, ids)] = ids[0] #unify group ids
    #make groups and return
    groups = defaultdict(list)
    for i,l in enumerate(locs):
        groups[l].append(patterns[i])
    return list(groups.values())

#catalogue all connection counts between equivalent patterns
def get_most_common_connections(groups, patterns, sequences, min_dist, min_count):
    conns = []
    maxlen = max([len(s) for s in sequences])
    total = maxlen*len(sequences)
    print("conns1", psutil.Process(os.getpid()).memory_info())
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
    print("conns2", psutil.Process(os.getpid()).memory_info())
    #concat and count
    c = np.concatenate(conns)
    #print('count', datetime.datetime.now())
    #counts = np.bincount(c)
    print(len(c))
    edges = Counter(c.tolist())
    print(c.nbytes, sys.getsizeof(c), sys.getsizeof(edges))
    print(len(edges))
    print("conns3", psutil.Process(os.getpid()).memory_info())
    edges = {e:c for e,c in edges.items() if c >= min_count}
    print(len(edges))
    #print('sort', datetime.datetime.now())
    bestconns = sorted(edges.items(), key=lambda c: c[1], reverse=True)
    bestconns = np.array(bestconns)[:,0]
    print("conns4", psutil.Process(os.getpid()).memory_info())
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
    print('bf', len(flatten(comps)))
    #print([len(c) for c in comps])
    scomps = group_by_maxadj(comps, sequences, path)
    
    #mixed up are removed automatically by individual seq grouping algo
    seqs = get_individual_seqs(scomps, len(sequences))
    #print_individual_seqs(seqs)
    seqs = [flatten(s, 1) for s in seqs]
    
    print('bf', len(flatten(scomps)))
    #print([len(c) for c in flatten(scomps, 1)])
    #add missing, start with longest seq
    locs = {o:(i,j) for i,s in enumerate(scomps) for j,t in enumerate(s) for o in t}
    seqlocs = {o:(i,j) for i,s in enumerate(seqs) for j,t in enumerate(s) for o in t}
    #numdefs = {(i,j):len([seg for seg in t if seg > ()]) for i,s in enumerate(seqs) for j,t in enumerate(s) for o in t}
    numdef = lambda t: len([seg for seg in t if seg > ()])
    moves = []
    for i,s in tqdm.tqdm(sorted(zip(range(len(seqs)), seqs), key=lambda s: len(s[1][0]), reverse=True), desc='adding'):
        for k,t in enumerate(s):
            offset = set([seg[1]-j for j,seg in enumerate(t) if seg > ()])
            if len(offset) == 1: #all offsets the same
                offset = next(iter(offset))
                index = next(seg[0] for seg in t if seg > ())
                first = next(i for i,s in enumerate(t) if s > ())
                last = len(t)-1-next(i for i,s in enumerate(reversed(t)) if s > ())
                gaps = [1 if first < i < last and s == () else 0 for i,s in enumerate(t)]
                for j,seg in enumerate(t):
                    if seg == () and 0 <= j+offset < len(sequences[index]):
                        seg = (index, j+offset)
                        loc = locs[seg] if seg in locs else None
                        loc2 = seqlocs[seg] if seg in seqlocs else None
                        #move if nowhere or in smaller scomp and this one quite complete
                        if seg in seqlocs:
                            sl = seqlocs[seg]
                        #len(t) > len(seqs[sl[0]][sl[1]]): #
                        if not loc or gaps[j]:#numdef(t) > numdef(seqs[sl[0]][sl[1]]):# or (len(scomps[loc[0]]) < len(t) and numdef > len(t)/2):
                            scomps[i][j].append(seg)
                            seqs[i][k][j] = seg
                            locs[seg] = (i,j)
                            seqlocs[seg] = (i,k)
                            if loc:#remove from old comp
                                scomps[loc[0]][loc[1]].remove(seg)
                                seqs[loc2[0]][loc2[1]] = [() if c == seg else c for c in seqs[loc2[0]][loc2[1]]]
                                moves.append((loc[0], loc[1], i, j))
    
    # seqs = get_individual_seqs(scomps, len(sequences))
    # #print_individual_seqs(seqs)
    # seqs = [flatten(s, 1) for s in seqs]
    
    # #separate sparse segments...
    # for i,s in enumerate(seqs):
    #     if len(s) > 0:
    #         newc = [[] for x in s[0]]
    #         for t in s:
    #             defined = [seg for seg in t if seg > ()]
    #             if len(defined) <= len(t)/2:#remove ones that are less than half full
    #                 for j,seg in enumerate(t):
    #                     if seg > ():
    #                         scomps[i][j].remove(seg)
    #                         newc[j].append(seg)
    #         scomps.extend([[c] for c in newc if len(c) > 0])
    
    print('bf', len(flatten(scomps)))
    nummoves = Counter(moves)
    print([len(scomps[c[0][0]][c[0][1]]) for c in list(nummoves.items())[:20]])
    
    # seqs = get_individual_seqs(scomps, len(sequences))
    #print_individual_seqs(seqs)
    #return
    
    
    
    scomps = [[c for c in s if len(c) > 0] for s in scomps]
    scomps = [s for s in scomps if len(s) > 0]
    
    #print('af', len(flatten(scomps, 2)))
    #print([len(c) for c in flatten(scomps, 1)])
    scomps = group_by_maxadj(flatten(scomps, 1), sequences, path+'r')
    #print('af', len(flatten(scomps, 2)))
    #print([len(c) for c in flatten(scomps, 1)])
    return [c for c in flatten(scomps, 1) if len(c) > 1]
    
    # seqs = get_individual_seqs(scomps, len(sequences))
    # print_individual_seqs(seqs)
    # print(len(flatten(scomps, 2)))
    # print_scomp_gaps(scomps, len(sequences))
    # plot_seq_x_comps(flatten(scomps, 1), sequences, path)
    # 
    # adjmax = get_comp_adjacency(flatten(scomps, 1), True)
    # plot_matrix(adjmax, path+'-max3.png')
    # print_offdiasum(adjmax)

def group_by_maxadj(comps, sequences, path=None):
    #sort by adjacency
    adjmax = get_comp_adjacency(comps, True)
    if path: plot_matrix(adjmax, path+'-max.png')
    #print_offdiasum(adjmax)
    # adjmax = get_comp_adjacency(comps, True)
    # plot_matrix(adjmax, 'maxl.png')
    #maxes = list(zip(*np.nonzero(adjmax)))
    maxes = list(zip(*np.where(adjmax >= 0.5)))
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
                    # print(len(scomps[ii]), len(scomps[jj]), iii, jjj, r, len(merged),
                    #     len(scomps[ii][:iii+1] + merged + scomps[jj][jjj+len(r):]))
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
    if path: plot_matrix(adjmax, path+'-max2.png')
    #print_offdiasum(adjmax)
    
    # adjmin = get_comp_adjacency(flatten(scomps, 1), False)
    # plot_matrix(adjmin, 'min.png')
    
    #add remaining comps
    in_scomps = set(flatten(scomps, 2))
    comps = [[n for n in c if n not in in_scomps] for c in comps]
    return [s for s in scomps if len(s) > 0] + [[c] for c in comps if len(c) > 0]

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
            occs = [sorted([o for o in c if o[0] == i]) for c in sc]
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
        for i in s[5:6]:
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