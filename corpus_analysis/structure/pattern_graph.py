from itertools import groupby, product
import numpy as np
import sortednp as snp
import pandas as pd
from scipy.sparse.csgraph import connected_components
import graph_tool.all as gt
from graph_tool.all import Graph, GraphView, graph_draw
from .graphs import graph_from_matrix
from ..util import plot_sequences, mode, flatten, group_by
from ..clusters.histograms import freq_hist_clusters, trans_hist_clusters,\
    freq_trans_hist_clusters
from ..alignment.smith_waterman import smith_waterman

MIN_VERSIONS = .03 #how many of the versions need to contain the patterns
PARSIM = True
MIN_SIM = 0.9 #min similarity for non-parsimonious similarity
MAX_LEN_DIFF = None #max diff for size-based prefilter. None deactivates it
COMPS_NOT_BLOCKS = True #use connected components instead of community blocks

def similarity(match, p1, p2, parsim):
    minlen = min(len(p1), len(p2))
    if parsim: return 1 if abs(match - minlen) <= 0 else 0
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

def cooc_similarity(p1, p2, occmat1, occmat2, parsim=PARSIM):
    both = np.logical_and(occmat1, occmat2)
    diffs = np.transpose(np.diff(np.where(both == 1)))
    overlaps = np.split(diffs, np.where(diffs != [0,1])[0]+1)
    return similarity(max([len(o) for o in overlaps]), p1, p2, parsim)

#group by a transitive distance
def group_by_sim2(patterns, simfunc, minsim):
    sims = np.array([[simfunc(p1, p2) >= minsim if j > i else 0
        for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
    return group_by(patterns, connected_components(sims))

def group_by_sim(groups, simfunc, minsim):
    return [subg for g in groups for subg in group_by_sim2(g, simfunc, minsim)]

def group_by_lengths(patterns, max_len_diff):
    sizes = np.array([len(p) for p in patterns])
    dists = np.absolute(sizes[:,None] - sizes) <= max_len_diff
    return group_by(patterns, connected_components(compatible))

def cluster(patterns, relative):
    #prepared = np.array([np.array(p)+1 for p in patterns])#to positive integers
    prepared = np.array([np.array([c for c in p if c >= 0]) for p in patterns])#remove -1
    return [[patterns[i] for i in c]
        for c in freq_trans_hist_clusters(prepared, relative)]

class PatternGraph:
    
    def __init__(self, sequences, pairings, alignments, sim_threshold=0.8):
        plot_sequences(sequences, 'seqpats1.png')
        self.patterns = self.create_pattern_dict(sequences, pairings, alignments)
        self.equivalences = {}
        self.print('all')
        #prune pattern dict: keep only ones occurring in a min num of versions
        self.patterns = {k:v for k,v in self.patterns.items()
            if len(np.unique([o[1] for o in v])) >= len(sequences)*MIN_VERSIONS}
        self.print('frequent')
        #merge cooccurring patterns
        self.merge_patterns(lambda p, q: len(p) == len(q) and
            len(self.patterns[p].intersection(self.patterns[q])) > 0)
        self.print('merged cooc')
        self.merge_patterns(lambda p, q: len(p) == len(q) and
            all(p[i] == -1 or q[i] == -1 or p[i] == q[i] for i in range(len(p))))
        self.print('merged equiv')
        
        groups = [list(self.patterns.keys())]
        
        if MAX_LEN_DIFF != None:
            groups = [h for g in groups for h in group_by_lengths(g, MAX_LEN_DIFF)]
        
        #cluster relative
        groups = [c for g in groups for c in cluster(g, True)]
        print('clusters', [len(g) for g in groups])
        
        #cluster absolute
        groups = [c for g in groups for c in cluster(g, False)]
        print('reclusters', [len(c) for c in groups])
        
        groups = self.keep_largest_pattern_groups(groups, 2)
        print('kept largest', [len(c) for c in groups])
        
        # #cooccurrence similarity
        # num_seqs = len(sequences)
        # max_len = max([len(s) for s in sequences])
        # occ_matrices = [get_occ_matrix(p, self.patterns, num_seqs, max_len)
        #     for p in patterns]
        # compatible = np.array([[cooc_similarity(p1, p2, occ_matrices[i], occ_matrices[j])
        #     if j > i and compatible[i,j] else 0
        #     for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
        # print('cooc', len(np.nonzero(compatible)[0]))
        
        # #content similarity
        # sorted_patterns = list([np.sort(p) for p in patterns])
        # compatible = group_by_sim(isect_similarity, sorted_patterns, compatible)
        
        # #sequence similarity
        # compatible = group_by_sim(sw_similarity, patterns, compatible)
        
        #make weighted adjacency matrix
        adjacency = self.get_adjacency_matrix(groups)
        patterns = list(self.patterns.keys())
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
    
    #keys: pattern tuples with -1 for blanks, values: list of occurrences (sequence, position)
    def create_pattern_dict(self, sequences, pairings, alignments):
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
    
    def keep_largest_pattern_groups(self, groups, count):
        largest = sorted(groups, key=lambda c: len(c), reverse=True)[:count]
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
