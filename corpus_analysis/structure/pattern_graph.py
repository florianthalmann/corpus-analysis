from itertools import groupby
import numpy as np
import sortednp as snp
import pandas as pd
import graph_tool.all as gt
from graph_tool.all import Graph, GraphView, graph_draw
from .graphs import graph_from_matrix
from ..util import group_by, plot_sequences
from ..clusters.histograms import freq_hist_clusters
from ..alignment.smith_waterman import smith_waterman

MIN_OCCS = 10

def pattern_similarity(p1, p2):
    sw = smith_waterman(p1, p2)[0]
    return len(sw) / min(len(p1), len(p2))
    #return len(np.intersect1d(p1, p2)) / min(len(p1), len(p2))

def sorted_pattern_similarity(p1, p2):
    return len(snp.intersect(p1, p2)) / min(len(p1), len(p2))

class PatternGraph:
    
    def __init__(self, sequences, pairings, alignments, sim_threshold=0.8):
        plot_sequences(sequences, 'seqpats1.png')
        self.patterns = self.create_pattern_dict(sequences, pairings, alignments)
        #prune pattern dict
        longest = sorted(list(self.patterns.items()), key=lambda i: len(i[1]), reverse=True)
        print('patterns', [len(l[1]) for l in longest[:3]], len(self.patterns))
        self.patterns = {k:v for k,v in self.patterns.items() if len(v) >= MIN_OCCS}
        print('frequent', len(self.patterns))
        patterns = list(self.patterns.keys())
        
        #cluster patterns to 
        clusters = self.get_pattern_clusters(patterns)
        print('clusters', [len(c) for c in clusters])
        
        sizes = np.array([len(p) for p in patterns])
        similar_size = np.absolute(sizes[:,None] - sizes) <= 1
        
        #content similarity
        spatterns = list([np.sort(p) for p in patterns])
        print('sim sizes', len(np.nonzero(similar_size)[0]), len(np.hstack(similar_size)))
        sims = np.array([[sorted_pattern_similarity(p1, p2)
            if j > i and similar_size[i,j] else 0
            for j, p2 in enumerate(spatterns)] for i, p1 in enumerate(spatterns)])
        sim_content = sims >= 0.9
        print('sim content', len(np.nonzero(sim_content)[0]), len(np.hstack(sim_content)))
        
        #sequence similarity
        sims = np.array([[pattern_similarity(p1, p2)
            if j > i and sim_content[i,j] else 0
            for j, p2 in enumerate(patterns)] for i, p1 in enumerate(patterns)])
        sim_sequence = sims >= 0.9
        print('sim sequence', len(np.nonzero(sim_sequence)[0]), len(np.hstack(sim_sequence)))
        
        #make graph and find communities
        g = graph_from_matrix(sims)
        g = GraphView(g, vfilt=g.get_total_degrees(g.get_vertices()) > 0)
        print(g)
        graph_draw(g, output_size=(1000, 1000), output="patterns.png")
        state = gt.minimize_blockmodel_dl(g, B_max=8)
        state.draw(output_size=(1000, 1000), output="patterns2.png")
        
        #make annotated sequences
        blocks = state.get_blocks().a+1
        typeseqs = [np.zeros_like(s)-1 for s in sequences]
        for p,b in sorted(list(zip(patterns, blocks)), key=lambda pb: pb[1]):
            for j,k in self.patterns[p]:
                typeseqs[j][k:k+len(p)] = b
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
