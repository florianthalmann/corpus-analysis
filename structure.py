from alignments import matrix_to_segments, segments_to_matrix
from graphs import alignment_graph, structure_graph, component_labels,\
    adjacency_matrix, graph_from_matrix
from hierarchies import build_hierarchy_bottom_up, make_segments_hierarchical
from util import plot_matrix

MIN_LENGTH = 10
MIN_DIST = 4

def simple_structure(sequence, self_alignment):
    #size = len(sequences[TEST_INDEX])
    #plot_matrix(segments_to_matrix(sas[TEST_INDEX], (size,size)), 'results/transitive1.png')
    hierarchy = make_segments_hierarchical(self_alignment, MIN_LENGTH, MIN_DIST)
    #plot_matrix(segments_to_matrix(hierarchy, (size,size)), 'results/transitive2.png')
    ag, s, i = alignment_graph([len(sequence)], [hierarchy])
    #plot(component_labels(g), 'results/labels1.png')
    hierarchy = build_hierarchy_bottom_up(component_labels(ag))
    #plot_matrix(hierarchy)
    return hierarchy

def shared_structure(sequences, sas, multinomial, msa):
    ag, s, i = alignment_graph([len(s) for s in sequences], sas)
    #g = transitive_closure(g)
    sg, _, _ = structure_graph(msa, ag)
    matrix = adjacency_matrix(sg)
    #plot_matrix(matrix, 'results/structure1.png')
    segments = matrix_to_segments(matrix)
    segments = make_segments_hierarchical(segments, MIN_LENGTH, MIN_DIST)
    #plot_matrix(segments_to_matrix(segments, matrix.shape), 'results/structure2.png')
    sg2 = graph_from_matrix(segments_to_matrix(segments, matrix.shape))
    hierarchy = build_hierarchy_bottom_up(component_labels(sg2))
    #plot_matrix(hierarchy)
