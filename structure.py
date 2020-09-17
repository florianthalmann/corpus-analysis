import numpy as np
from alignments import matrix_to_segments, segments_to_matrix
from graphs import alignment_graph, structure_graph, component_labels,\
    adjacency_matrix, graph_from_matrix
from hierarchies import build_hierarchy_bottom_up, make_segments_hierarchical,\
    get_hierarchy_sections
from util import plot_matrix, mode

MIN_LENGTH = 10
MIN_DIST = 4

def simple_structure(sequence, self_alignment):
    #size = len(sequences[TEST_INDEX])
    #plot_matrix(segments_to_matrix(sas[TEST_INDEX], (size,size)), 'results/transitive1.png')
    hierarchy = make_segments_hierarchical(self_alignment, MIN_LENGTH, MIN_DIST)
    #plot_matrix(segments_to_matrix(hierarchy, (size,size)), 'results/transitive2.png')
    ag, s, i = alignment_graph([len(sequence)], [[0, 0]], [hierarchy])
    #plot(component_labels(g), 'results/labels1.png')
    #print(sequence[:10])
    #connected component label for each position in sequence
    comp_labels = component_labels(ag)
    #most frequent value in sequence for each component
    comp_values = np.array([mode(sequence[np.where(comp_labels == l)])
        for l in range(np.max(comp_labels)+1)])
    #print(comp_values[comp_labels][:10])
    sections = get_hierarchy_sections(comp_labels)
    labeled_sections = [comp_values[s] for s in sections]
    #print(labeled_sections)
    return labeled_sections
    
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
    hierarchy = build_hierarchy_bottom_up(component_labels(sg2))
    #plot_matrix(hierarchy)
