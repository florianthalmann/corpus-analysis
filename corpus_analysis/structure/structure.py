from itertools import product
import numpy as np
from ..alignment.affinity import matrix_to_segments, segments_to_matrix
from .graphs import alignment_graph, structure_graph, component_labels,\
    adjacency_matrix, graph_from_matrix, clean_up
from .hierarchies import make_segments_hierarchical,\
    get_hierarchy_labels, get_hierarchies
from ..util import plot_matrix, mode, profile, plot_sequences
from graph_tool.topology import max_cliques

def remove_blocks(alignment, min_len):
    matrix = segments_to_matrix(alignment)
    #plot_matrix(matrix, 'blocks0.png')
    s = matrix.shape
    w = 5#int(min_len/2)
    blocks = np.array([[np.logical_or(
        np.mean(matrix[i, j:min(s[1],j+w)]) == 1,
        np.mean(matrix[i, max(0,j-w+1):j+1]) == 1)
        for j in range(s[1])] for i in range(s[0])])
    #plot_matrix(blocks, 'blocks1.png')
    #print('initial', len(alignment))
    block_segs = [a for a in alignment if np.mean(blocks[tuple(a.T)]) >= 0.9]
    alignment = [a for a in alignment if np.mean(blocks[tuple(a.T)]) < 0.9]
    #print('no blocks', len(alignment))
    #plot_matrix(segments_to_matrix(alignment, s), 'blocks0.9.png')
    return alignment, block_segs

def clean_up_alignment(sequence, self_alignment):
    g, s, i, a, seg  = alignment_graph([len(sequence)], [[0, 0]], [self_alignment])
    g2 = clean_up(g, i, seg)
    #profile(lambda: clean_up(g, i, seg))
    return matrix_to_segments(np.triu(adjacency_matrix(g2)))

def simple_structure(sequence, self_alignment, min_len, min_dist):
    if len(self_alignment) > 0:
        #print(sequence)
        #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'est1.png')
        #clean up and make transitive and hierarchical
        self_alignment, blocks = remove_blocks(self_alignment, min_len)
        #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'est2.png')
        #profile(lambda: clean_up_alignment(sequence, self_alignment))
        self_alignment = clean_up_alignment(sequence, self_alignment)
        #print('cleaned up', len(self_alignment))
        #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'est3.png')
        hierarchy = make_segments_hierarchical(self_alignment, min_len, min_dist, len(sequence))#, 'est4')#, 'yoyy')
        #put blocks back
        hierarchy += blocks
        #connected component labels for each position in sequence
        ag, s, i, a, seg = alignment_graph([len(sequence)], [[0, 0]], [hierarchy])
        comp_labels = component_labels(ag)
        #replace sequence with most frequent value in sequence for each component
        comp_values = np.array([mode(sequence[np.where(comp_labels == l)])
            for l in range(np.max(comp_labels)+1)])
        improved_sequence = comp_values[comp_labels]
        labels = get_hierarchy_labels([comp_labels])[0]
        #plot_sequences(labels, 'hierarchy62.png')
        return labels#np.append(labels, [improved_sequence], axis=0)
        
        #plot_matrix(hierarchy)
        #return sections
    return np.array([np.repeat(0, len(sequence))])

def shared_structure(sequences, pairings, alignments, msa, min_len, min_dist):
    ag, s, i = alignment_graph([len(s) for s in sequences], pairings, alignments)
    #g = transitive_closure(g)
    sg, cm, _ = structure_graph(msa, ag)
    matrix = adjacency_matrix(sg)
    #plot_matrix(matrix, 'results/structure1.png')
    plot_matrix(cm)
    segments = matrix_to_segments(matrix)
    segments = make_segments_hierarchical(segments, min_len, min_dist)
    #plot_matrix(segments_to_matrix(segments, matrix.shape), 'results/structure2.png')
    sg2 = graph_from_matrix(segments_to_matrix(segments, matrix.shape))[0]
    hierarchy = get_hierarchies([component_labels(sg2)])[0]
    #plot_matrix(hierarchy)

def shared_structure2(sequences, pairings, alignments):
    ag, s, i, a, seg  = alignment_graph([len(s) for s in sequences], pairings, alignments)
    print(ag)
    print(len(list(max_cliques(ag))))
