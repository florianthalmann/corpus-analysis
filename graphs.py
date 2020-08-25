import numpy as np
from graph_tool.all import Graph, graph_draw, GraphView

def to_alignment_graph(lengths=[], self_alignments=[], multi_alignments=[]):
    print('making graph')
    g = Graph()
    seq_index = g.new_vertex_property("int")
    time = g.new_vertex_property("int")
    #add vertices
    g.add_vertex(sum(lengths))
    seq_index.a = np.concatenate([np.repeat(i,l) for i,l in enumerate(lengths)])
    time.a = np.concatenate([np.arange(l) for l in lengths])
    #add self_alignments
    for i,a in enumerate(self_alignments):
        pairs = np.concatenate(a, axis=0)
        indices = (np.arange(lengths[i]) + sum(lengths[:i]))[pairs]
        g.add_edge_list(indices)
    print('created alignment graph', g)
    not_isolated = g.new_vertex_property("bool")
    not_isolated.a = g.get_total_degrees(g.get_vertices()) > 0
    g = GraphView(g, vfilt=not_isolated)
    print('created alignment graph', g)
    #graph_draw(g, output_size=(1000, 1000), output="results/box3.pdf")
    return g, seq_index, time

def get_connected_components():
    return 0