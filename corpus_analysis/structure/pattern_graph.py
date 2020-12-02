from itertools import groupby
import numpy as np
import sortednp as snp
import pandas as pd
import graph_tool.all as gt
from graph_tool.all import Graph, GraphView, graph_draw
from .graphs import graph_from_matrix
from ..util import group_by, plot_sequences, mode
from ..clusters.histograms import freq_hist_clusters
from ..alignment.smith_waterman import smith_waterman

MIN_VERSIONS = .5 #how many of the versions need to contain the patterns
PARSIM = True
MIN_SIM = 0.9 #min similarity for non-parsimonious similarity
MAX_LEN_DIFF = None #max diff for size-based prefilter. None deactivates it
PRE_CLUSTER = True
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

class PatternGraph:
    
    def __init__(self, sequences, pairings, alignments, sim_threshold=0.8):
        plot_sequences(sequences, 'seqpats1.png')
        self.patterns = self.create_pattern_dict(sequences, pairings, alignments)
        #prune pattern dict: keep only ones occurring in a min num of versions
        longest = sorted(list(self.patterns.items()), key=lambda i: len(i[1]), reverse=True)
        print('patterns', [len(l[1]) for l in longest[:3]], len(self.patterns))
        self.patterns = {k:v for k,v in self.patterns.items()
            if len(np.unique([o[1] for o in v])) >= len(sequences)*MIN_VERSIONS}
        print('frequent', len(self.patterns))
        patterns = list(self.patterns.keys())
        
        #mark pairs that should be compared
        compatible = np.ones((len(patterns), len(patterns)))
        if MAX_LEN_DIFF != None:
            sizes = np.array([len(p) for p in patterns])
            compatible *= np.absolute(sizes[:,None] - sizes) <= 1
            print('sim sizes', len(np.nonzero(compatible)[0]), len(np.hstack(compatible)))
        if PRE_CLUSTER:
            print(len(patterns))
            clusters = self.get_pattern_clusters(patterns)
            print('clusters', [len(c) for c in clusters])
            #compatible *= 
        
        #cooccurrence similarity
        # num_seqs = len(sequences)
        # max_len = max([len(s) for s in sequences])
        # occ_matrices = [get_occ_matrix(p, self.patterns, num_seqs, max_len)
        #     for p in patterns]
        # compatible = np.array([[cooc_similarity(p1, p2, occ_matrices[i], occ_matrices[j])
        #     if j > i and compatible[i,j] else 0
        #     for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
        # print('cooc', len(np.nonzero(compatible)[0]))
        
        #content similarity
        sorted_patterns = list([np.sort(p) for p in patterns])
        compatible = self.narrow_down(isect_similarity, sorted_patterns, compatible)
        
        #sequence similarity
        compatible = self.narrow_down(sw_similarity, patterns, compatible)
        
        #make graph and find communities
        g = graph_from_matrix(compatible)
        g = GraphView(g, vfilt=g.get_total_degrees(g.get_vertices()) > 0)
        print(g)
        graph_draw(g, output_size=(1000, 1000), output="patterns.png")
        state = gt.minimize_blockmodel_dl(g)#, B_max=8)
        state.draw(output_size=(1000, 1000), output="patterns2.png")
        blocks = state.get_blocks().a+1
        if COMPS_NOT_BLOCKS: blocks = gt.label_components(g)[0].a+1
        
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
        typeseqs = [[mode(e) if len(e) > 0 else -1 for e  in s] for s in typeseqs]
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
    
    def get_pattern_clusters(self, patterns):
        pos_patterns = np.array([np.array(p)+1 for p in patterns])#to positive integers
        labels = freq_hist_clusters(pos_patterns)
        return group_by(np.arange(len(patterns)), labels)
    
    def narrow_down(self, func, patterns, compatible):
        sims = np.array([[func(p1, p2) if j > i and compatible[i,j] else 0
            for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
        sims = sims >= MIN_SIM
        print(func.__name__, len(np.nonzero(sims)[0]), '/', len(np.hstack(sims)))
        return sims
