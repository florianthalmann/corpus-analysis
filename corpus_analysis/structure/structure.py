import numpy as np
from ..alignment.affinity import matrix_to_segments, segments_to_matrix
from .graphs import alignment_graph, structure_graph, component_labels,\
    adjacency_matrix, graph_from_matrix, clean_up
from .hierarchies import make_segments_hierarchical,\
    get_hierarchy_sections, get_hierarchy_labels
from ..util import plot_matrix, mode, profile

MIN_LENGTH = 20
MIN_DIST = 4

def clean_up_alignment(sequence, self_alignment):
    print('cleaning up')
    g, s, i, a, seg  = alignment_graph([len(sequence)], [[0, 0]], [self_alignment])
    g2 = clean_up(g, i, seg)
    #profile(lambda: clean_up(g, i, seg))
    return matrix_to_segments(np.triu(adjacency_matrix(g2)))

def illustrate_transitivity(sequence, self_alignment):
    self_alignment = clean_up_alignment(sequence, self_alignment)
    
    profile(lambda: make_segments_hierarchical(self_alignment, MIN_LENGTH, MIN_DIST, len(sequence), 'results/ccccc'))

def simple_structure(sequence, self_alignment):
    #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'm1.png')
    #clean up and make transitive and hierarchical
    #profile(lambda: clean_up_alignment(sequence, self_alignment))
    self_alignment = clean_up_alignment(sequence, self_alignment)
    #plot_matrix(segments_to_matrix(self_alignment, (len(sequence),len(sequence))), 'm2.png')
    
    hierarchy = make_segments_hierarchical(self_alignment, MIN_LENGTH, MIN_DIST, len(sequence))
    #connected component labels for each position in sequence
    ag, s, i, a, seg = alignment_graph([len(sequence)], [[0, 0]], [hierarchy])
    comp_labels = component_labels(ag)
    #replace sequence with most frequent value in sequence for each component
    comp_values = np.array([mode(sequence[np.where(comp_labels == l)])
        for l in range(np.max(comp_labels)+1)])
    improved_sequence = comp_values[comp_labels]
    #print(improved_sequence)
    #plot_matrix(np.vstack([comp_labels, sequence, improved_sequence]))
    labels = get_hierarchy_labels(comp_labels)
    return np.append(labels, [improved_sequence], axis=0)
    
    #plot_matrix(hierarchy)
    #return sections

def shared_structure(sequences, pairings, alignments, msa):
    ag, s, i = alignment_graph([len(s) for s in sequences], pairings, alignments)
    #g = transitive_closure(g)
    sg, cm, _ = structure_graph(msa, ag)
    matrix = adjacency_matrix(sg)
    #plot_matrix(matrix, 'results/structure1.png')
    plot_matrix(cm)
    segments = matrix_to_segments(matrix)
    segments = make_segments_hierarchical(segments, MIN_LENGTH, MIN_DIST)
    #plot_matrix(segments_to_matrix(segments, matrix.shape), 'results/structure2.png')
    sg2 = graph_from_matrix(segments_to_matrix(segments, matrix.shape))
    hierarchy = get_hierarchy(component_labels(sg2))
    #plot_matrix(hierarchy)
