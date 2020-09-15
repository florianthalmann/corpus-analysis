import numpy as np
from graph_tool.all import Graph, graph_draw, GraphView, edge_endpoint_property
from graph_tool.topology import label_components
from graph_tool.spectral import adjacency
from graph_tool.inference.blockmodel import BlockState
from graph_tool.topology import transitive_closure
from util import plot

def adjacency_matrix(graph):
    return adjacency(graph).toarray().T

def prune_isolated_vertices(g):
    not_isolated = g.new_vertex_property("bool")
    not_isolated.a = g.get_total_degrees(g.get_vertices()) > 0
    return GraphView(g, vfilt=not_isolated)

def graph_from_matrix(matrix):
    g = Graph(directed=False)
    g.add_vertex(len(matrix))
    g.add_edge_list(list(zip(*np.nonzero(matrix))))
    #graph_draw(g, output_size=(1000, 1000), output="results/structure.pdf")
    return g

def alignment_graph(lengths=[], self_alignments=[], pairings=[], mutual_alignments=[]):
    print('making graph')
    g = Graph(directed=False)
    seq_index = g.new_vertex_property("int")
    time = g.new_vertex_property("int")
    #add vertices
    g.add_vertex(sum(lengths))
    seq_index.a = np.concatenate([np.repeat(i,l) for i,l in enumerate(lengths)])
    time.a = np.concatenate([np.arange(l) for l in lengths])
    #add self-alignments
    for i,a in enumerate(self_alignments):
        pairs = np.concatenate(a, axis=0)
        indices = (np.arange(lengths[i]) + sum(lengths[:i]))[pairs]
        g.add_edge_list(indices)
    #add mutual alignments
    for i,a in enumerate(mutual_alignments):
        j, k = pairings[i]
        pairs = np.concatenate(a, axis=0)
        indicesJ = (np.arange(lengths[j]) + sum(lengths[:j]))[pairs.T[0]]
        indicesK = (np.arange(lengths[k]) + sum(lengths[:k]))[pairs.T[1]]
        g.add_edge_list(np.vstack([indicesJ, indicesK]).T)
    #g.add_edge_list([(b, a) for (a, b) in g.edges()])
    print('created alignment graph', g)
    #g = prune_isolated_vertices(g)
    #print('pruned alignment graph', g)
    #g = transitive_closure(g)
    #graph_draw(g, output_size=(1000, 1000), output="results/box3.pdf")
    return g, seq_index, time

def structure_graph(msa, alignment_graph, mask_threshold=.5):
    msa = [[int(m[1:]) if len(m) > 0 else -1 for m in a] for a in msa]
    matches = alignment_graph.new_vertex_property("int")
    matches.a = np.concatenate(msa)
    num_partitions = np.max(matches.a)+1
    #create connection matrix
    edge_ps = matches.a[alignment_graph.get_edges()] #partition memberships for edges
    edge_ps = edge_ps[np.where(np.all(edge_ps != -1, axis=1))] #filter out non-partitioned
    conn_matrix = np.zeros((num_partitions, num_partitions), dtype=int)
    np.add.at(conn_matrix, tuple(edge_ps.T), 1)
    np.add.at(conn_matrix, tuple(np.flip(edge_ps, axis=1).T), 1)
    conn_matrix = np.triu(conn_matrix, k=1)
    max_conn = np.max(conn_matrix)
    conn_matrix[np.where(conn_matrix < mask_threshold*max_conn)] = 0
    #create graph
    g = graph_from_matrix(conn_matrix)
    print('created structure graph', g)
    return g, conn_matrix, matches

def component_labels(g):
    labels, hist = label_components(g)
    return labels.a
